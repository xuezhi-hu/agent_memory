"""Run SparseMem with shared histories and the project's LoCoMo scoring protocol."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from . import BASE
from .construct import DEFAULT, MODES, PROMPTS
from .history import project_locomo
from .quality import audit
from .run import (
    CountingEncoder, answer_question, build_memory, chat_call, mock,
    pack_question, usage, write_lines,
)
from dpc_mem.client import ChatClient, Embedder, FixtureEmbedder, HTTPTransport, endpoint
from dpc_mem.data import load_history, load_questions, prepare
from dpc_mem.evaluation import RUBRICS, evaluate
from dpc_mem.io import RunLock, code_hash, digest, file_hash, key, read, save
from dpc_mem.metrics import Recorder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import FixtureTokenizer, HFTokenizer


def distribution(values):
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95))) if values else None


def run(args, original):
    data = args.data.resolve()
    if read(data / "manifest.json")["dataset"] != "locomo":
        raise ValueError("Expected a prepared LoCoMo dataset")
    questions = load_questions(data, [args.history])
    if not questions:
        raise ValueError("No selected LoCoMo questions")
    models = BASE / "models"
    embedding = dict(original["embedding"], model_path=str(models / "all-MiniLM-L6-v2"))
    tok = FixtureTokenizer() if args.mock else HFTokenizer(
        str(models / "Qwen3-30B-A3B-Instruct-2507-tokenizer")
    )
    out = args.out.resolve()
    with RunLock(out):
        backend = FixtureEmbedder() if args.mock else Embedder(embedding, None, Recorder(out), out)
        encoder = CountingEncoder(backend)
        encoder.limit = read(models / "all-MiniLM-L6-v2/sentence_bert_config.json")["max_seq_length"]
        live = original["reader"]
        url = None if args.mock else endpoint(os.environ[live["base_url_env"]])
        if not args.mock and not os.environ.get(live["key_env"]):
            raise ValueError("Missing Reader credential")
        manifest = dict(dataset="locomo", data_path=str(data), history_ids=[args.history],
                        data_sha256=file_hash(data / "manifest.json"), mode=args.mode,
                        backend="mock" if args.mock else "real", reader=live, endpoint=url,
                        config={"judge": original["judge"], "judge_max_tokens": original["judge_max_tokens"]},
                        memory_config=dict(DEFAULT, budget=args.budget, write_max=args.write_max), seed=42,
                        workers=args.workers, reader_retry_limit=512, question_ids=[q["question_id"] for q in questions],
                        tokenizer=tok.identity, encoder=backend.identity, prompts=digest(PROMPTS),
                        judge_rubric_sha256=file_hash(RUBRICS / "locomo.txt"),
                        code=digest([code_hash(), {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}]))
        mp = out / "run_manifest.json"
        if mp.exists() and read(mp) != manifest:
            raise ValueError("Run configuration changed. Use a new output directory.")
        save(mp, manifest)
        cfg = manifest["memory_config"]
        turns = project_locomo(load_history(data, args.history))
        build_root = out / "memory"
        build_root.mkdir(exist_ok=True)
        done = build_root / "stats.json"
        if done.exists():
            memory = (read(build_root / "sources.json"), read(build_root / "memories.json"),
                      Index.load(build_root / "index.json", encoder.identity), read(done))
        else:
            call = mock if args.mock else chat_call(build_root, live, original, tok)
            memory = build_memory(build_root, turns, args.mode, cfg, 42, tok, encoder, call)
            save(done, memory[3])
        save(build_root / "quality_report.json", audit(memory[1], memory[0], memory[3]))
        print(f"Built {args.history}: {memory[3]['block_count']} blocks, {len(questions)} questions", flush=True)
        answer_root = out / "answers" / "sparsemem" / args.mode
        trace_root = out / "queries"
        wall_start = time.perf_counter()
        targets = [answer_root / (key(q["question_id"]) + ".json") for q in questions]
        # A prior transport outage can leave a checkpoint with status=failed.
        # It must be retried after the Reader comes back; only successful
        # answers are terminal checkpoints.
        reports = {
            target: read(target)
            for target in targets
            if target.exists() and read(target).get("status") == "ok"
        }

        def answer(root, packed, qid, target):
            call = mock if args.mock else chat_call(root, live, original, tok)
            result = answer_question(
                root, memory, packed, cfg, tok, call, retry_limit=manifest["reader_retry_limit"]
            )
            result["status"] = "ok" if result["status"] == "returned" else result["status"]
            result["question_id"] = qid
            save(target, result)
            return target, result

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            work = []
            for q, target in zip(questions, targets):
                if target in reports:
                    continue
                qid = q["question_id"]
                root = trace_root / key(qid)
                root.mkdir(parents=True, exist_ok=True)
                packed = pack_question(root, memory, q["question"], q["question_date"], cfg, tok, encoder, None)
                work.append(pool.submit(answer, root, packed, qid, target))
            for future in as_completed(work):
                target, result = future.result()
                reports[target] = result
                if len(reports) % 10 == 0 or len(reports) == len(questions):
                    print(f"Answered {len(reports)}/{len(questions)}", flush=True)
        wall_file = out / "answer_wall.json"
        wall = read(wall_file) if wall_file.exists() else 0
        if work:
            wall += time.perf_counter() - wall_start
            save(wall_file, wall)
        rows = [reports[path] for path in targets]
        ledger_paths = [build_root / "ledger.json"] + [trace_root / key(q["question_id"]) / "ledger.json" for q in questions]
        ledger = [dict(x, scope=str(p.parent.relative_to(out))) for p in ledger_paths if p.exists() for x in read(p).values()]
        write_lines(out / "llm_usage.jsonl", ({k: v for k, v in x.items() if k != "content"} for x in ledger))
        write_lines(out / "hypotheses.jsonl", (dict(question_id=r["question_id"], hypothesis=r["answer"]) for r in rows))
        stats = dict(usage(ledger), construction=memory[3], processed=len(rows),
                     selected=len(questions), answer_failures=sum(r["status"] != "ok" for r in rows),
                     answer_batch_wall_seconds=wall)
        for metric in ("evidence_tokens", "recall_seconds", "query_wall_seconds", "query_generation_seconds"):
            name = metric + "_distribution" if metric == "query_generation_seconds" else metric
            stats[name] = distribution([r["stats"][metric] for r in rows])
        save(out / "metrics.json", stats)
        print(json.dumps({k: stats[k] for k in ("processed", "answer_failures", "total_generation_tokens")}), flush=True)


def score(out, original, requested_model=None):
    with RunLock(out):
        manifest = read(out / "run_manifest.json")
        metrics = read(out / "metrics.json")
        if manifest["backend"] != "real" or metrics["processed"] != metrics["selected"]:
            raise ValueError("Expected completed real predictions")
        if manifest["data_sha256"] != file_hash(Path(manifest["data_path"]) / "manifest.json"):
            raise ValueError("Prepared dataset changed")
        if manifest["judge_rubric_sha256"] != file_hash(RUBRICS / "locomo.txt"):
            raise ValueError("Judge rubric changed")
        config = dict(manifest["config"]["judge"])
        model = requested_model or config["model"]
        if model == "gpt-4o-mini":
            config.update(
                base_url_env="DPC_JUDGE_BASE_URL",
                key_env="DPC_JUDGE_API_KEY",
                model=model,
                # Lazu accepts the public alias but returns the provider's
                # dated model identifier in the response envelope.
                response_models=[model, "gpt-4o-mini-2024-07-18"],
                request_extra={},
            )
        elif config != original["judge"] or model != "qwen2.5-32b-instruct":
            raise ValueError("Unsupported judge configuration")
        recorder = Recorder(out / "judge")
        transport = HTTPTransport(out / "judge", recorder, True, original["timeout_seconds"],
                                  dict(max_retries_per_request=0, max_retries_per_run=0, backoff_seconds=[]))
        tok = HFTokenizer(str(BASE / "models/Qwen3-30B-A3B-Instruct-2507-tokenizer"))
        judge = ChatClient(config, transport, tok, reader=False)
        result = evaluate(out, manifest, "sparsemem", manifest["mode"], judge, recorder)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--history", action="append", required=True)
    generate = commands.add_parser("run")
    generate.add_argument("--data", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--history", required=True)
    generate.add_argument("--mode", choices=MODES, default="sparse")
    generate.add_argument("--budget", type=int, default=DEFAULT["budget"])
    generate.add_argument("--write-max", type=int, default=768)
    generate.add_argument("--workers", type=int, default=4)
    backend = generate.add_mutually_exclusive_group(required=True)
    backend.add_argument("--mock", action="store_true")
    backend.add_argument("--execute", action="store_true")
    scoring = commands.add_parser("score")
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--model", choices=("qwen2.5-32b-instruct", "gpt-4o-mini"), default=None)
    scoring.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    original = read(BASE / "configs/compact_v3.json")
    if args.command == "prepare":
        print(prepare(args.source, "locomo", args.data, args.history))
    elif args.command == "run":
        if args.workers < 1 or args.budget < 0 or args.write_max < 1:
            parser.error("Invalid workers or construction budget")
        run(args, original)
    else:
        score(args.out.resolve(), original, args.model)


if __name__ == "__main__":
    main()
