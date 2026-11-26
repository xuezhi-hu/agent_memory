"""Instrument the upstream judge's SDK boundary; keep its exact prompts and labels."""

import argparse, json, runpy, sys, time
from pathlib import Path
from unittest.mock import patch
from . import BASE
from dpc_mem.io import RunLock, append, read, save, file_hash
from .run import usage

PIN = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
SCRIPT = Path(__file__).resolve().parents[1] / "vendor/LongMemEval/evaluate_qa.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=("gpt-4o-mini", "gpt-4o", "llama-3.1-70b-instruct"),
    )
    p.add_argument("--execute", required=True, action="store_true")
    args = p.parse_args()
    manifest = read(args.run / "run_manifest.json")
    metrics = read(args.run / "metrics.json")
    if (
        manifest["backend"] != "real"
        or manifest["source_sha256"] != file_hash(args.data)
        or metrics["processed"] != metrics["selected"]
    ):
        p.error("Judge requires completed real predictions and matching dataset")
    from openai.resources.chat.completions import Completions

    original = Completions.create
    dest = args.run / "judge" / args.model
    log = dest / "usage.jsonl"

    def traced(self, *a, **kw):
        start = time.perf_counter()
        row = {"stage": "judge", "model": kw["model"]}
        try:
            response = original(self, *a, **kw)
            row["usage"] = response.usage.model_dump() if response.usage else None
            return response
        except Exception as exc:
            row["error"] = type(exc).__name__
            raise
        finally:
            row["seconds"] = time.perf_counter() - start
            append(log, row)

    root = args.run
    hyp = root / "hypotheses.jsonl"
    rows = [json.loads(s) for s in hyp.read_text().splitlines()]
    refs = {s["question_id"]: s for s in read(args.data)}
    stats = []
    with RunLock(dest):
        for i, row in enumerate(rows):
            item = dest / f"{i:05}"
            item.mkdir(exist_ok=True)
            saved = item / "result.json"
            signature = file_hash(SCRIPT) + str(row)
            if saved.exists():
                result = read(saved)
                if result["signature"] != signature:
                    raise ValueError("Judge checkpoint mismatch")
            else:
                h = item / "hyp.jsonl"
                h.write_text(json.dumps(row) + "\n")
                ref = item / "ref.json"
                sample = refs[row["question_id"]]
                save(
                    ref,
                    [
                        {
                            k: sample[k]
                            for k in (
                                "question_id",
                                "question_type",
                                "question",
                                "answer",
                            )
                        }
                    ],
                )
                with patch.object(Completions, "create", traced), patch.object(
                    sys, "argv", [str(SCRIPT), args.model, str(h), str(ref)]
                ):
                    runpy.run_path(str(SCRIPT), run_name="__main__")
                got = json.loads(
                    Path(str(h) + ".eval-results-" + args.model).read_text()
                )
                result = dict(
                    signature=signature,
                    label=got["autoeval_label"]["label"],
                    model=got["autoeval_label"]["model"],
                )
                save(saved, result)
            stats.append((refs[row["question_id"]]["question_type"], result["label"]))
        logs = (
            [json.loads(s) for s in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        u = [r.get("usage") or {} for r in logs]
        save(
            dest / "metrics.json",
            dict(
                commit=PIN,
                script_sha256=file_hash(SCRIPT),
                accuracy=sum(v for _, v in stats) / len(stats),
                by_type={
                    k: sum(v for t, v in stats if t == k)
                    / sum(t == k for t, v in stats)
                    for k, _ in stats
                },
                judge_calls=len(u),
                known_input=sum(x.get("prompt_tokens", 0) for x in u),
                known_output=sum(x.get("completion_tokens", 0) for x in u),
                unknown_usage=sum(not x for x in u),
                seconds=sum(r["seconds"] for r in logs),
            ),
        )


if __name__ == "__main__":
    main()
