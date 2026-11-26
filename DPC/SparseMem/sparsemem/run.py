import argparse, json, os, re, resource, sys, time
from pathlib import Path
from collections import Counter
from . import BASE
from .history import project, stamp, segment
from .dependency import prepare
from .construct import DEFAULT, MODES, PROMPTS, Ledger, construct, messages
from .recall import keys, retrieve
from dpc_mem.client import (
    ChatClient,
    HTTPTransport,
    Embedder,
    FixtureEmbedder,
    endpoint,
)
from dpc_mem.tokenization import FixtureTokenizer, HFTokenizer
from dpc_mem.metrics import Recorder
from dpc_mem.io import RunLock, code_hash, digest, file_hash, read, save
from dpc_mem.retrieval import Index


class CountingEncoder:
    def __init__(self, backend):
        self.backend = backend
        self.identity = backend.identity
        self.tokens = 0
        self.seconds = 0.0
        self.hits = 0

    def embed(self, texts, role="document", **context):
        start = time.perf_counter()
        vectors = self.backend.embed(texts, role=role, **context)
        self.seconds += time.perf_counter() - start
        if texts:
            if isinstance(self.backend, FixtureEmbedder):
                self.tokens += sum(FixtureTokenizer().count(s) for s in texts)
            else:
                # Include special tokens, report logical encoder workload including cache reads.
                from transformers import AutoTokenizer

                if not hasattr(self, "tokenizer"):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.backend.config["model_path"], local_files_only=True
                    )
                for text in texts:
                    self.tokens += min(
                        len(self.tokenizer.encode(text, verbose=False)), self.limit
                    )
        self.hits += getattr(self.backend, "last_cache_hits", 0)
        return vectors


def mock(msg, limit, stage):
    tok = FixtureTokenizer()
    body = msg[1]["content"]
    text = "Offline fixture response."
    if stage == "construct":
        line = body.split("TARGET (source IDs, roles, dates, text):\n")[1].splitlines()[
            0
        ]
        k = line[1:].split(" |")[0]
        text = json.dumps(
            dict(
                facts=[
                    dict(
                        text=line.split("] ", 1)[1],
                        source_ids=[k],
                        entities=[],
                        event_start=None,
                        event_end=None,
                        time_expression=None,
                        precision="unknown",
                        assertion="reported",
                    )
                ],
                unresolved=[],
            )
        )
    return text, dict(
        prompt_tokens=tok.messages(msg), completion_tokens=tok.count(text)
    )


def write_lines(path, rows):
    tmp = path.with_suffix(".writing")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def usage(rows):
    result = {}
    for stage in ("construct", "query"):
        sel = [x for x in rows if x["stage"] == stage]
        unknown = sum("input" not in x for x in sel)
        result.update(
            {
                stage + "_calls": len(sel),
                stage + "_unknown_usage_calls": unknown,
                stage + "_known_input_tokens": sum(x.get("input", 0) for x in sel),
                stage + "_known_output_tokens": sum(x.get("output", 0) for x in sel),
                stage + "_charged": sum(x["charged"] for x in sel),
                stage + "_reserved_sum": sum(x["reserved"] for x in sel),
                stage + "_generation_seconds": sum(x["seconds"] or 0 for x in sel),
            }
        )
    result["total_generation_tokens"] = (
        None
        if any("input" not in x for x in rows)
        else sum(x["input"] + x["output"] for x in rows)
    )
    return result


def build_memory(root, turns, mode, cfg, seed, tok, encoder, call):
    start = time.perf_counter()
    recorder = Recorder(root)
    ledger = Ledger(root / "ledger.json")
    archive, blocks = segment(turns, tok, encoder, cfg, mode)
    records = prepare(archive, blocks, tok, cfg, mode)
    segment_time = time.perf_counter() - start
    save(root / "sources.json", archive)
    candidate_count = construct(records, archive, cfg, mode, seed, tok, ledger, call)
    save(root / "memories.json", records)
    index_start = time.perf_counter()
    entries = keys(records, archive, key_sentences=(
        cfg.get("key_sentences_per_record", 2) if cfg.get("key_sentence_index", False) else 0
    ))
    index = Index.build(root / "index.json", entries, encoder, recorder)
    index_time = time.perf_counter() - index_start
    rows = list(ledger.rows.values())
    cost = usage(rows)
    refs = [k for x in rows for k in x["refs"]]
    unique = set(refs)
    rawcounts = {k: tok.count(archive[k]["text"]) for k in unique}
    counts = Counter(r["status"] for r in records)
    stats = dict(
        cost,
        raw_history_tokens=sum(tok.count(t["text"]) for t in turns),
        levels=dict(Counter(r["level"] for r in records)),
        primary_fragment_tokens_by_level={
            level: sum(
                tok.count(archive[k]["text"])
                for r in records
                if r["level"] == level
                for k in r["primary"]
            )
            for level in ("L0", "L1", "L2")
        },
        source_token_coverage=(
            (
                sum(rawcounts.values())
                / sum(tok.count(v["text"]) for v in archive.values())
            )
            if any(v["text"].strip() for v in archive.values())
            else 0
        ),
        block_count=len(blocks),
        candidate_count=candidate_count,
        status_counts=dict(counts),
        fallback_count=sum(
            counts[k]
            for k in [
                "failed",
                "invalid_json",
                "invalid_schema",
                "empty_facts",
                "interrupted_unknown",
            ]
        ),
        unresolved_count=sum(
            bool(r["flags"] or r.get("model_unresolved")) for r in records
        ),
        raw_sources=len(archive),
        refine_source_tokens_unique=sum(rawcounts.values()),
        refine_source_tokens_repeated=sum(rawcounts[k] for k in refs)
        - sum(rawcounts.values()),
        segment_link_seconds=segment_time,
        index_seconds=index_time,
        construct_wall_seconds=time.perf_counter() - start,
        index_size_bytes=(root / "index.json").stat().st_size,
        embedding_tokens=encoder.tokens,
        embedding_seconds=encoder.seconds,
        embedding_cache_hits=encoder.hits,
        budget=cfg["budget"],
        budget_overrun=max(0, cost["construct_charged"] - cfg["budget"]),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    return archive, records, index, stats


def pack_question(root, memory, question, qdate, cfg, tok, encoder, reranker):
    archive, records, index, _ = memory
    query_start = time.perf_counter()
    trace = retrieve(
        index, encoder, records, archive, question, qdate, cfg, tok, reranker
    )
    recall_time = time.perf_counter() - query_start
    save(root / "retrieval.json", trace)
    msg = messages(
        "answer", participant_dialogue=any(t.get("participant_dialogue") for t in archive.values()),
        question=question, question_date=qdate, evidence=trace["evidence"],
        aggregate_query=trace.get("aggregate", False),
    )
    return trace, msg, recall_time


def answer_question(root, memory, packed, cfg, tok, call, ledger=None, retry_limit=None):
    archive, records, _, _ = memory
    trace, msg, recall_time = packed
    query_start = time.perf_counter()
    ledger = ledger or Ledger(root / "ledger.json")
    response = ledger.generate(
        "reader", "query", msg, cfg["answer_max"], cfg["margin"], None, tok, call
    )
    if retry_limit is not None and response.get("finish_reason") == "length":
        response = ledger.generate(
            f"reader_recovery_{retry_limit}", "query", msg, retry_limit,
            cfg["margin"], None, tok, call,
        )
    query_usage = usage(list(ledger.rows.values()))
    session = lambda keys: sorted({archive[k]["session"] for k in keys})
    result = dict(
        answer=response.get("content", ""),
        status=response["status"],
        stats=dict(query_usage, evidence_tokens=trace["tokens"],
                   reranker_pairs=trace["reranker_pairs"], recall_seconds=recall_time,
                   query_wall_seconds=recall_time + time.perf_counter() - query_start),
        packed_turn_ids=sorted({archive[k]["id"] for k in trace["packed"]}),
        packed_sessions=session(trace["packed"]),
        retrieved_sessions=session(
            [k for m in trace["ranked"] for k in records[int(m[1:])]["primary"]]
        ),
    )
    return result


def sample_run(root, turns, question, qdate, mode, cfg, seed, tok, encoder, call, reranker):
    start = time.perf_counter()
    memory = build_memory(root, turns, mode, cfg, seed, tok, encoder, call)
    packed = pack_question(root, memory, question, qdate, cfg, tok, encoder, reranker)
    ledger = Ledger(root / "ledger.json")
    result = answer_question(root, memory, packed, cfg, tok, call, ledger)
    result["stats"] = dict(memory[3], **result["stats"])
    result["stats"].update(usage(list(ledger.rows.values())),
        wall_seconds=time.perf_counter() - start, embedding_tokens=encoder.tokens,
        embedding_seconds=encoder.seconds, embedding_cache_hits=encoder.hits)
    save(root / "result.json", result)
    return result


def chat_call(root, live, original, tok):
    transport = HTTPTransport(
        root,
        Recorder(root),
        True,
        original["timeout_seconds"],
        dict(
            max_retries_per_request=0,
            max_retries_per_run=0,
            backoff_seconds=[],
        ),
    )
    client = ChatClient(live, transport, tok)

    def call(msg, max_output, stage):
        transport.last_call = None
        try:
            answer = client.complete(
                msg[0]["content"],
                msg[1]["content"],
                stage,
                max_output,
                json_object_output=stage == "construct",
            )
        except Exception as exc:
            exc.usage = (transport.last_call or {}).get("usage")
            raise
        return answer, client.last_completion["usage"]
    return call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, default="sparse")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget", type=int)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--ids", type=Path, help="JSON list of frozen question IDs")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--reranker", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = DEFAULT.copy()
    overrides = read(args.config) if args.config else {}
    unknown = set(overrides) - set(cfg)
    if unknown:
        p.error("Unknown config keys " + str(unknown))
    cfg.update(overrides)
    if args.budget is not None:
        cfg["budget"] = args.budget
    if (
        args.limit < 1
        or cfg["budget"] < 0
        or any(
            cfg[k] < 1
            for k in (
                "block_tokens",
                "lookback",
                "channel_k",
                "final_k",
                "answer_max",
                "write_max",
                "evidence_tokens",
            )
        )
        or any(cfg[k] < 0 for k in ("margin", "max_links", "link_tokens", "expand_k"))
        or type(cfg["aggregate_mode"]) is not bool
        or type(cfg["key_sentence_index"]) is not bool
        or type(cfg["key_sentences_per_record"]) is not int
        or not 0 <= cfg["key_sentences_per_record"] <= 2
        or set(cfg["weights"]) != set("ECTU")
        or any(w < 0 for w in cfg["weights"].values())
        or abs(sum(cfg["weights"].values()) - 1) > 1e-9
    ):
        p.error("Invalid limits or weights")
    original = read(BASE / "configs/compact_v3.json")
    data = read(args.data)
    ids = read(args.ids) if args.ids else None
    if ids is not None and (
        len(set(ids)) != len(ids) or set(ids) - {s["question_id"] for s in data}
    ):
        p.error("Invalid question IDs")
    selected = [
        (i, s) for i, s in enumerate(data) if ids is None or s["question_id"] in ids
    ][: args.limit]
    del data
    if not selected or len({s["question_id"] for _, s in selected}) != len(selected):
        p.error("Missing/duplicate samples")
    out = args.out
    modeldir = BASE / "models"
    original["embedding"]["model_path"] = str(modeldir / "all-MiniLM-L6-v2")
    tok = (
        FixtureTokenizer()
        if args.mock
        else HFTokenizer(str(modeldir / "Qwen3-30B-A3B-Instruct-2507-tokenizer"))
    )
    with RunLock(out):
        rec = Recorder(out)
        backend = (
            FixtureEmbedder()
            if args.mock
            else Embedder(original["embedding"], None, rec, out)
        )
        reranker = None
        rerank_hash = None
        if args.reranker:
            from sentence_transformers import CrossEncoder
            from dpc_mem.topic_segmentation import model_files_identity

            rerank_hash = model_files_identity(str(args.reranker), "Reranker")
            reranker = CrossEncoder(
                str(args.reranker), local_files_only=True, device="cpu"
            )
        live = original["reader"]
        base_url = None if args.mock else endpoint(os.environ[live["base_url_env"]])
        if not args.mock and not os.getenv(live["key_env"]):
            p.error("Missing " + live["key_env"])
        manifest = dict(
            mode=args.mode,
            seed=args.seed,
            config=cfg,
            backend="mock" if args.mock else "real",
            reader=None if args.mock else live,
            endpoint=base_url,
            prompt_hash=digest(PROMPTS),
            encoder=backend.identity,
            embedding_model=original["embedding"]["model"] if not args.mock else None,
            reranker=rerank_hash,
            tokenizer=tok.identity,
            source_sha256=file_hash(args.data),
            selected=[s["question_id"] for i, s in selected],
            code=digest(
                [
                    code_hash(),
                    {f.name: file_hash(f) for f in Path(__file__).parent.glob("*.py")},
                ]
            ),
            concurrency=1,
            retries=0,
        )
        mp = out / "run_manifest.json"
        if mp.exists() and (not args.resume or read(mp) != manifest):
            p.error("Existing run requires --resume and identical manifest")
        if not mp.exists() and any(p.name != ".writer.lock" for p in out.iterdir()):
            p.error("Output must start empty")
        save(mp, manifest)
        hyps = []
        reports = []
        all_usage = []
        invocation = []
        for i, sample in selected:
            root = out / f"scope_{i:05}"
            root.mkdir(exist_ok=True)
            turns, mapping = project(sample)
            qdate = sample.get("question_date", "")
            qt = stamp(qdate)
            if qt and any(t["date"] and stamp(t["date"]) > qt for t in turns):
                raise ValueError("Future session in " + root.name)
            encoder = CountingEncoder(backend)
            if args.execute:
                encoder.limit = read(
                    modeldir / "all-MiniLM-L6-v2/sentence_bert_config.json"
                )["max_seq_length"]
            call = mock
            if args.execute:
                call = chat_call(root, live, original, tok)

            ledgerpath = root / "ledger.json"
            before = set(read(ledgerpath)) if ledgerpath.exists() else set()
            cached = (root / "result.json").exists()
            result = (
                read(root / "result.json")
                if cached
                else sample_run(
                    root,
                    turns,
                    sample["question"],
                    qdate,
                    args.mode,
                    cfg,
                    args.seed,
                    tok,
                    encoder,
                    call,
                    reranker,
                )
            )
            ledger = read(ledgerpath)
            invocation.extend(v for k, v in ledger.items() if k not in before)
            all_usage.extend(dict(v, scope=root.name, key=k) for k, v in ledger.items())
            gold = set(sample.get("answer_session_ids", []))
            abstain = "_abs" in sample["question_id"]

            def recall(sessions):
                return (
                    len(gold.intersection(mapping[k] for k in sessions)) / len(gold)
                    if gold and not abstain
                    else None
                )

            report = dict(
                result["stats"],
                scope=root.name,
                question_id=sample["question_id"],
                question_type=sample["question_type"],
                answer_status=result["status"],
                missing_date_messages=sum(not t["date"] for t in turns),
                session_recall=recall(result["retrieved_sessions"]),
                packed_recall=recall(result["packed_sessions"]),
            )
            save(
                out / "evaluation_only" / (root.name + ".json"),
                dict(mapping=mapping, report=report),
            )
            reports.append(report)
            hyps.append(
                dict(question_id=sample["question_id"], hypothesis=result["answer"])
            )
            write_lines(out / "hypotheses.jsonl", hyps)
            write_lines(out / "construction_stats.jsonl", reports)
            write_lines(
                out / "llm_usage.jsonl",
                ({k: v for k, v in row.items() if k != "content"} for row in all_usage),
            )
            save(
                out / "metrics.json",
                dict(
                    usage(all_usage),
                    selected=len(selected),
                    processed=len(reports),
                    answer_failures=sum(
                        r["answer_status"] != "returned" for r in reports
                    ),
                    current_invocation=usage(invocation),
                    judge_status="not_run",
                    backend=manifest["backend"],
                ),
            )
            print(root.name, result["status"], flush=True)


if __name__ == "__main__":
    main()
