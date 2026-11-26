"""Append-only accounting: actual requests, logical cache reads, timed spans."""
from __future__ import annotations
import csv
import math
import resource
import statistics
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from .io import append, read, save


def percentile(values, p):
    values = sorted(float(x) for x in values if isinstance(x, (float, int)) and math.isfinite(x))
    if not values:
        return None
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos-lo)


def distribution(values):
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95)}


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "events.jsonl"
        if self.path.exists():
            import json
            raw=self.path.read_bytes();lines=raw.splitlines(keepends=True)
            if lines:
                try:json.loads(lines[-1])
                except (json.JSONDecodeError,UnicodeDecodeError):
                    archive=self.root/("events_interrupted_tail_"+uuid.uuid4().hex+".bin")
                    archive.write_bytes(lines[-1]);self.path.write_bytes(b"".join(lines[:-1]))
                else:
                    if not raw.endswith(b"\n"):
                        with self.path.open("ab") as f:f.write(b"\n")

    def emit(self, event, **values):
        append(self.path, {"event": event, "time": time.time(), **values})

    @contextmanager
    def span(self, stage, **context):
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.emit("span", span_id=uuid.uuid4().hex, stage=stage, status=status,
                      seconds=time.perf_counter()-start,
                      process_peak_rss_mib=rss/(1024**2 if sys.platform == "darwin" else 1024),
                      **context)


def events(root):
    import json
    path = Path(root) / "events.jsonl"
    if not path.exists():return []
    lines=path.read_text().splitlines();result=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1:break  # Interrupted final append; calls/*.json remains authoritative.
            raise ValueError("Event ledger corruption before final line") from None
    return result


def usage_summary(rows, pricing=None):
    out = {"requests": len(rows), "failed_or_pending": 0, "known_input_tokens": 0,
           "known_output_tokens": 0, "unknown_input_requests": 0, "unknown_output_requests": 0,
           "priced_known_cost": 0.0, "unpriced_requests": 0}
    for r in rows:
        out["failed_or_pending"] += r.get("status") not in {"returned", "ok"}
        usage = r.get("usage") or {}
        inp = usage.get("prompt_tokens", usage.get("input_tokens"))
        op = usage.get("completion_tokens", usage.get("output_tokens"))
        for name, v in [("input",inp),("output",op)]:
            if isinstance(v,int) and not isinstance(v,bool) and v >= 0:
                out["known_"+name+"_tokens"] += v
            else:
                out["unknown_"+name+"_requests"] += 1
        rates = (pricing or {}).get(r.get("model"), {})
        if (isinstance(inp,int) and isinstance(op,int)
                and rates.get("input_per_million") is not None and rates.get("output_per_million") is not None):
            out["priced_known_cost"] += (inp*rates["input_per_million"]+op*rates["output_per_million"])/1e6
        else:
            out["unpriced_requests"] += 1
    out["complete_priced_cost"] = out["priced_known_cost"] if not out["unpriced_requests"] else None
    return out


def report(root):
    root = Path(root)
    rows = events(root)
    requests = {}
    for r in rows:
        if r["event"] == "request":
            requests[r["call_id"]] = r  # pending + final counts once
    # Recover request accounting even if the process died between atomic call
    # persistence and the final append to the event ledger.
    imported = {}
    for path in root.glob("calls/*.json"):
        record=read(path)
        if record.get("imported_from"):
            imported[record["call_id"]]=record
            continue
        for attempt in record.get("attempts",[record]):
            requests[attempt["call_id"]]=attempt
    used_imports={r["call_id"]:imported[r["call_id"]] for r in rows
                  if r["event"]=="cache_hit" and r.get("call_id") in imported}
    config = read(root/"manifest.json")["config"] if (root/"manifest.json").exists() else {}
    by_stage = defaultdict(list)
    for r in requests.values():
        by_stage[r["stage"]].append(r)
    span_groups = defaultdict(list)
    for r in rows:
        if r["event"] == "span":
            tag = "/".join(str(r.get(k,"all")) for k in ["stage","group","mode"])
            span_groups[tag].append(r)
    latency = {k:{**distribution([r["seconds"] for r in v if r["status"]=="ok"]),
                  "failed_spans":sum(r["status"]!="ok" for r in v),
                  "sum_all_span_seconds":sum(r["seconds"] for r in v)} for k,v in span_groups.items()}
    predictions = defaultdict(list)
    for p in root.glob("answers/*/*/*.json"):
        r=read(p)
        predictions[(r["group"],r["mode"])].append(r)
    tables=[]
    for (group,mode), pp in sorted(predictions.items()):
        good=[p for p in pp if p["status"]=="ok"]
        tables.append({"group":group,"mode":mode,"recorded_questions":len(pp),"completed":len(good),
            "evidence_tokens_mean":statistics.fmean(p["evidence_tokens"] for p in good) if good else None,
            "retrieval_seconds":distribution([p["retrieval_seconds"] for p in good]),
            "answer_seconds":distribution([p["answer_seconds"] for p in good]),
            "query_seconds":distribution([p["query_seconds"] for p in good]),
            "search_and_pack_seconds":distribution([p.get("search_and_pack_seconds",p["retrieval_seconds"]) for p in good]),
            "actual_group_wall_seconds":distribution([p.get("actual_group_wall_seconds",p["query_seconds"]) for p in good]),
            "cached_answers":sum(p.get("answer_call",{}).get("cache_hit") is True for p in good),
            "uncached_answer_seconds":distribution([p["answer_seconds"] for p in good if p.get("answer_call",{}).get("cache_hit") is False]),
            "query_embedding_cache_hits":sum(p.get("query_embedding_cache_hits") or 0 for p in good)})
    # Attribute shared construction to each independent method, while keeping actual
    # experiment expenditure separate. No cache read is a second paid request.
    standalone={}
    built_groups={p.parent.name for p in root.glob("memories/*/*.json")}
    for group, stages in {"A":{"extract"},"B":{"extract","consolidate"},
                          "C":{"extract","consolidate","guard"},
                          "D":{"extract","consolidate","selfcheck"},
                          "E":{"extract","consolidate","guard_minimal"},
                          "T":{"tfm_write"}}.items():
        standalone[group]=(usage_summary([r for r in list(requests.values())+list(used_imports.values())
                                        if r["stage"] in stages],config.get("pricing"))
                           if group in built_groups else None)
    storage=defaultdict(int)
    for p in root.rglob("*"):
        if p.is_file():
            part=p.relative_to(root).parts[0]
            storage[part]+=p.stat().st_size
    usage_by_group=defaultdict(list)
    for r in requests.values():
        usage_by_group['/'.join(str(r.get(k,'shared')) for k in ['stage','group','mode'])].append(r)
    # Logical deployment usage reattributes identical cached requests to the method
    # that consumes them. This is an estimate, not a second actual charge.
    logical={}
    for r in rows:
        if r["event"]=="cache_hit" or (r["event"]=="request" and r.get("status")=="returned"):
            original=requests.get(r["call_id"],imported.get(r["call_id"],{}))
            item={**original,**{k:v for k,v in r.items() if k in {"stage","group","mode","question_id","window_id","history_id"}}}
            ident=tuple(item.get(k) for k in ["stage","group","mode","question_id","window_id","history_id","request_key"])
            logical[ident]=item
    logical_groups=defaultdict(list)
    for r in logical.values():logical_groups['/'.join(str(r.get(k,'shared')) for k in ['stage','group','mode'])].append(r)
    memory_stats=[]
    for p in root.glob("memories/*/*.json"):
        v=read(p)
        memory_stats.append({k:v.get(k) for k in ["history_id","group","card_count","memory_text_tokens","base_fact_tokens","raw_text_tokens","representation_ratio","representation_counts","model_tokens"]})
    result={"schema_version":1,"token_counts":"provider usage; missing fields remain unknown",
        "actual_model_usage":{k:usage_summary(v,config.get("pricing")) for k,v in sorted(by_stage.items())},
        "actual_usage_by_stage_group_mode":{k:usage_summary(v,config.get("pricing")) for k,v in sorted(usage_by_group.items())},
        "reused_from_prior_runs":{"requests":len(used_imports),
            "source_runs":sorted({r["imported_from"] for r in used_imports.values()}),
            "historical_usage":usage_summary(list(used_imports.values()),config.get("pricing"))},
        "standalone_construction_usage":standalone,
        "logical_usage_including_cache":{k:usage_summary(v,config.get("pricing")) for k,v in sorted(logical_groups.items())},
        "automatic_retries":sum(r["event"]=="automatic_retry" for r in rows),
        "logical_cache_reads":sum(r["event"]=="cache_hit" for r in rows),
        "latency_seconds":latency,"question_efficiency":tables,
        "memory_statistics":memory_stats,
        "fixture_calls":sum(r["event"]=="fixture_call" for r in rows),
        "construction_fallbacks":[r for r in rows if r["event"]=="construction_fallback"],
        "partial_consolidation_fallbacks":[r for r in rows if r["event"]=="partial_consolidation_fallback"],
        "guard_decision_fallbacks":[r for r in rows if r["event"]=="guard_decision_fallback"],
        "guard_json_recoveries":[r for r in rows if r["event"]=="guard_json_recovered"],
        "storage_bytes_by_artifact":dict(storage),
        "process_peak_rss_mib":max((r.get("process_peak_rss_mib",0) for r in rows),default=None),
        "notes":["Judge cost is separate from deployment cost.",
                 "Nested stage sums are not end-to-end wall time.",
                 "RSS is process lifetime high-water mark, not per-stage GPU memory.",
                 "Missing prices are not zero cost; record hardware and pricing for monetary comparisons.",
                 "Index cache hits and fixture timings are not cold-start model benchmarks.",
                 "Cached answers are excluded from uncached_answer_seconds; logical_usage_including_cache is attribution, not billing.",
                 "Imported model replies are recorded separately and are excluded from actual current-run usage."]}
    result["notes"].append("Per-question retrieval/query latency attributes the same shared query embedding duration to every compared group; actual group wall time and search-only time are also provided.")
    save(root/"reports/efficiency.json",result)
    root.joinpath("reports").mkdir(exist_ok=True)
    with (root/"reports/efficiency.csv").open("w",newline="",encoding="utf-8-sig") as f:
        names=["group","mode","recorded_questions","completed","evidence_tokens_mean","retrieval_p50_s","retrieval_p95_s","query_p50_s","query_p95_s"]
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in tables:
            w.writerow({**{k:r[k] for k in names[:5]},"retrieval_p50_s":r["retrieval_seconds"]["p50"],
                "retrieval_p95_s":r["retrieval_seconds"]["p95"],"query_p50_s":r["query_seconds"]["p50"],"query_p95_s":r["query_seconds"]["p95"]})
    lines=["# DPC-Mem 效率报告", "", "实际调用、共享构建归因、评分和索引分别统计。未知用量及未配置价格不记为零。", "",
           "|组别|检索|已完成/已记录|平均证据 token|检索 P50 / P95 秒|单题 P50 / P95 秒|", "|---|---|---|---|---|---|"]
    for r in tables:
        lines.append(f'|{r["group"]}|{r["mode"]}|{r["completed"]}/{r["recorded_questions"]}|{r["evidence_tokens_mean"]}|{r["retrieval_seconds"]["p50"]} / {r["retrieval_seconds"]["p95"]}|{r["query_seconds"]["p50"]} / {r["query_seconds"]["p95"]}|')
    lines += ["", "完整分项见 efficiency.json。报告中已记录题数不代替冻结题目总数；完整性见对应评分报告。"]
    (root/"reports/efficiency.md").write_text("\n".join(lines)+"\n")
    return result
