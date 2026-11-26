"""Serial, durable write reservations. Persist the reservation before dispatch."""

import json, random, time
from pathlib import Path
from datetime import date
from string import Template
from dpc_mem.io import read, save, json_object, digest
from .dependency import render

PROMPTS = {
    p.stem: p.read_text().strip()
    for p in (Path(__file__).parent / "prompts").glob("*.txt")
}
MODES = (
    "raw",
    "raw_round",
    "raw_session",
    "raw_block",
    "linked_only",
    "full_extract",
    "random_all",
    "random_budget",
    "random_candidate",
    "initial_dependency",
    "sparse_no_link",
    "sparse",
)
DEFAULT = dict(
    block_tokens=512,
    split_threshold=0.50,
    lookback=4,
    max_links=2,
    link_tokens=256,
    weights=dict.fromkeys("ECTU", 0.25),
    threshold=0.25,
    budget=16000,
    write_max=384,
    margin=64,
    answer_max=256,
    evidence_tokens=4096,
    channel_k=40,
    fused_k=40,
    expand_k=10,
    final_k=12,
    rrf_k=60,
    aggregate_mode=True,
    key_sentence_index=False,
    key_sentences_per_record=2,
)


def messages(prompt, participant_dialogue=False, **fields):
    system = PROMPTS[prompt + "_system"]
    if participant_dialogue:
        system = system.replace("Assistant suggestions are not confirmed user actions.",
                                "Suggestions from one participant are not confirmed actions by the other.")
        system = system.replace("assistant advice when stated", "advice from either participant when stated")
        if prompt == "refine":
            system += (" Focus on resolving the flagged dependencies with one or two concise facts."
                       " Copy each full source ID including its @ span. Use compact JSON.")
    if prompt == "answer" and fields.get("aggregate_query"):
        system += (
            " For questions asking for a total, count, average, comparison, or all/different events, "
            "first enumerate every distinct supported event in the supplied evidence, then deduplicate "
            "repeated mentions and compute the requested result. Do not infer that no other event exists "
            "merely because the retrieved evidence is incomplete. "
            "Return only the answer and the shortest calculation needed to justify it. "
            "Do not restate the question or reproduce long evidence. For list/count questions, "
            "use at most 10 short items and stay below 120 words whenever possible. "
            "Do not output a long uncertainty discussion; if a value is missing, state that in one short sentence."
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": Template(PROMPTS[prompt + "_user"]).substitute(fields),
        },
    ]


def request(r, archive):
    return messages(
        "refine",
        participant_dialogue=archive[r["primary"][0]].get("participant_dialogue", False),
        block_id=r["id"],
        dependency_flags=",".join(r["flags"]) or "none",
        target_dialogue="\n".join(render(archive, k) for k in r["primary"]),
        linked_dialogue="\n".join(render(archive, k) for k in r["links"]) or "(none)",
    )


def strings(x):
    return isinstance(x, list) and all(isinstance(v, str) and v.strip() for v in x)


def validate(text, r):
    obj = json_object(text)
    facts = obj.get("facts")
    unresolved = obj.get("unresolved")
    if not isinstance(facts, list) or not strings(unresolved):
        raise ValueError("schema")
    for fact in facts:
        if (
            not isinstance(fact, dict)
            or not isinstance(fact.get("text"), str)
            or not fact["text"].strip()
        ):
            raise ValueError("fact text")
        refs = fact.get("source_ids")
        required = {
            "entities",
            "precision",
            "time_expression",
            "event_start",
            "event_end",
            "assertion",
        }
        if (
            not required <= fact.keys()
            or not strings(refs)
            or not set(refs) <= set(r["primary"] + r["links"])
            or not set(refs) & set(r["primary"])
        ):
            raise ValueError("schema/source")
        if (
            not strings(fact["entities"])
            or fact["assertion"]
            not in ("reported", "planned", "negated", "uncertain", "hypothetical")
            or fact["precision"]
            not in ("day", "week", "month", "year", "range", "unknown")
        ):
            raise ValueError("enums")
        if fact["time_expression"] is not None and not isinstance(
            fact["time_expression"], str
        ):
            raise ValueError("time expression")
        bounds = [
            date.fromisoformat(v) if v is not None else None
            for v in (fact["event_start"], fact["event_end"])
        ]
        if all(bounds) and bounds[1] < bounds[0]:
            raise ValueError("inverted interval")
    return facts, unresolved


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = read(path) if self.path.exists() else {}

    def spent(self, stage):
        return sum(v["charged"] for v in self.rows.values() if v["stage"] == stage)

    def generate(self, key, stage, msg, limit, margin, budget, tok, call, refs=()):
        signature = digest([stage, msg, limit, margin, refs])
        reserve = tok.messages(msg) + limit + margin
        if key in self.rows:
            row = self.rows[key]
            if row["signature"] != signature:
                raise ValueError("Resume request changed")
            return row
        if budget is not None and self.spent(stage) + reserve > budget:
            return None
        row = dict(
            signature=signature,
            stage=stage,
            status="interrupted_unknown",
            reserved=reserve,
            charged=reserve,
            usage=None,
            refs=list(refs),
            seconds=None,
        )
        self.rows[key] = row
        save(self.path, self.rows)
        start = time.perf_counter()
        try:
            content, usage = call(msg, limit, stage)
            row.update(content=content, status="returned", usage=usage)
        except Exception as exc:
            row.update(
                status="failed",
                error=type(exc).__name__,
                finish_reason=getattr(exc, "finish_reason", None),
                usage=getattr(exc, "usage", None),
            )
        usage = row["usage"] or {}
        a = usage.get("prompt_tokens", usage.get("input_tokens"))
        b = usage.get("completion_tokens", usage.get("output_tokens"))
        if all(type(v) is int and v >= 0 for v in (a, b)):
            row.update(input=a, output=b, charged=a + b)
        row["seconds"] = time.perf_counter() - start
        save(self.path, self.rows)
        return row


def construct(records, archive, cfg, mode, seed, tok, ledger, call):
    candidates = []
    for r in records:
        eligible = (
            r["support"]
            and (r["R"] >= cfg["threshold"] or r["critical"])
            and r["R"] > 0
        )
        r["refinement_status"] = (
            "insufficient_evidence"
            if r["flags"] and not r["support"]
            else "not_required"
        )
        r["generation_tokens"] = 0
        initial = (
            r["support"]
            and r["D"] > 0
            and (r["D"] >= cfg["threshold"] or 1 in r["initial"].values())
        )
        if mode in ("full_extract", "random_all", "random_budget") or (
            initial
            if mode == "initial_dependency"
            else eligible and mode in ("random_candidate", "sparse", "sparse_no_link")
        ):
            msg = request(r, archive)
            cost = tok.messages(msg) + cfg["write_max"] + cfg["margin"]
            r["estimated_cost"] = cost
            candidates.append(
                ((r["D"] if mode == "initial_dependency" else r["R"]) / cost, r, msg)
            )
    if mode.startswith("random"):
        random.Random(seed).shuffle(candidates)
    elif mode != "full_extract":
        candidates.sort(key=lambda t: -t[0])
    for priority, r, msg in candidates:
        r["priority"] = priority
        # Replays must obey the original admission sequence (including skipped expensive blocks).
        row = ledger.generate(
            r["id"],
            "construct",
            msg,
            cfg["write_max"],
            cfg["margin"],
            cfg["budget"],
            tok,
            call,
            r["primary"] + r["links"],
        )
        if row is None:
            r["refinement_status"] = "deferred"
            r["status"] = "budget_skipped"
            continue
        r["generation_tokens"] = row["charged"] if "input" in row else None
        r["refinement_status"] = "failed"
        if row["status"] != "returned":
            r["status"] = row["status"]
            continue
        try:
            facts, unresolved = validate(row["content"], r)
        except (ValueError, TypeError, KeyError) as exc:
            r["status"] = (
                "invalid_json"
                if isinstance(exc, json.JSONDecodeError)
                else "invalid_schema"
            )
            continue
        r["facts"] = facts
        r["model_unresolved"] = unresolved
        if facts:
            r.update(level="L2", status="refined", refinement_status="refined")
        else:
            r["status"] = "empty_facts"
            r["refinement_status"] = "unresolved"
    return len(candidates)
