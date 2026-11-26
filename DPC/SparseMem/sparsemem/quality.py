"""Deterministic audits for SparseMem intermediate memory products."""

from collections import Counter


def audit(records, archive, construction_stats=None):
    levels = Counter(r.get("level") for r in records)
    l1 = [r for r in records if r.get("level") == "L1"]
    l2 = [r for r in records if r.get("level") == "L2"]
    valid_links = invalid_links = cross_session_links = unsupported_links = 0
    no_link_non_temporal = 0
    for r in l1:
        primary_sessions = {archive[k]["session"] for k in r.get("primary", [])}
        link_sessions = {archive[k]["session"] for k in r.get("links", [])}
        if link_sessions - primary_sessions:
            cross_session_links += 1
        evidence = {e for t in r.get("trace", []) for e in t.get("evidence", [])}
        for link in r.get("links", []):
            if link in archive and link not in r.get("primary", []):
                valid_links += 1
                if link not in evidence:
                    unsupported_links += 1
            else:
                invalid_links += 1
        if not r.get("links") and not r.get("temporal"):
            no_link_non_temporal += 1

    accepted_l2 = invalid_l2_sources = 0
    for r in l2:
        if r.get("status") == "refined" and r.get("facts"):
            accepted_l2 += 1
        allowed = set(r.get("primary", []) + r.get("links", []))
        primary = set(r.get("primary", []))
        for fact in r.get("facts", []):
            refs = set(fact.get("source_ids", [])) if isinstance(fact, dict) else set()
            if not refs or not refs <= allowed or not refs.intersection(primary):
                invalid_l2_sources += 1

    fallback_statuses = {
        "failed", "invalid_json", "invalid_schema", "empty_facts",
        "interrupted_unknown", "budget_skipped",
    }
    fallbacks = sum(1 for r in records if r.get("status") in fallback_statuses)
    n = max(len(records), 1)
    return {
        "record_count": len(records),
        "level_counts": dict(levels),
        "level_ratios": {k: levels[k] / n for k in ("L0", "L1", "L2")},
        "l1_count": len(l1),
        "l1_valid_link_count": valid_links,
        "l1_invalid_link_count": invalid_links,
        "l1_unsupported_link_count": unsupported_links,
        "l1_no_link_non_temporal_count": no_link_non_temporal,
        "cross_session_link_record_count": cross_session_links,
        "l2_count": len(l2),
        "l2_accepted_count": accepted_l2,
        "l2_invalid_source_count": invalid_l2_sources,
        "fallback_count": fallbacks,
        "fallback_rate": fallbacks / n,
        "status_counts": dict(Counter(r.get("status") for r in records)),
        "construction_stats": construction_stats or {},
    }
