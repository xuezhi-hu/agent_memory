"""Optional, source-backed sentence keys for SparseMem development.

No model calls or benchmark labels are used. The original memory records and
retrieval logic are not modified. Text is an exact slice of the primary source;
an anchor is a lexical hint, not a claim that two mentions denote one entity.
"""
import math
import re
from collections import Counter, defaultdict

from sparsemem.recall import AGGREGATE_STOP, keys
from dpc_mem.retrieval import terms


def build_entries(records, archive, *, max_per_record=2, allowed_roles=None):
    """Add diverse verbatim keys to the baseline entries, without mutating inputs.

    Ranking uses document frequency within this history only.  Each key carries
    exact offsets into its original source and belongs to the original memory.
    Limits apply per memory; two mentions across sources are never deduplicated.
    """
    if type(max_per_record) is not int or max_per_record < 0:
        raise ValueError("max_per_record must be a nonnegative integer")
    roles = None if allowed_roles is None else frozenset(allowed_roles)
    candidates, document_frequency = {}, Counter()
    for record in records:
        rows, seen = [], set()
        for source_id in record["primary"]:
            source = archive[source_id]
            if roles is not None and source.get("role") not in roles:
                continue
            text = source["text"]
            for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|(?=\n|$))", text):
                start, end = match.span()
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                sentence = text[start:end]
                anchors = {w for w in terms(sentence) if len(w) > 3 and w not in AGGREGATE_STOP}
                identity = (source_id, " ".join(sentence.lower().split()))
                if not (25 <= len(sentence) <= 400 and len(anchors) >= 3) or identity in seen:
                    continue
                seen.add(identity)
                rows.append(dict(source_id=source_id, start=start, end=end,
                                 text=sentence, anchors=anchors))
        candidates[record["id"]] = rows
        document_frequency.update(set().union(*(r["anchors"] for r in rows)) if rows else set())

    total = max(len(records), 1)
    result = keys(records, archive)
    for record in records:
        by_source = defaultdict(list)
        for row in candidates[record["id"]]:
            rarity = sum(math.log((total + 1) / (document_frequency[w] + 1))
                         for w in row["anchors"])
            row["score"] = rarity / math.sqrt(len(row["anchors"]))
            by_source[row["source_id"]].append(row)
        for rows in by_source.values():
            rows.sort(key=lambda r: (-r["score"], r["start"]))
        # Select one sentence from each source before taking a second from any
        # source, while ranking each round by the same history-only score.
        selected = []
        while by_source and len(selected) < max_per_record:
            heads = sorted((rows[0] for rows in by_source.values()),
                           key=lambda r: (-r["score"], archive[r["source_id"]]["order"], r["start"]))
            for row in heads:
                if len(selected) >= max_per_record:
                    break
                selected.append(row)
                rows = by_source[row["source_id"]]
                rows.pop(0)
                if not rows:
                    del by_source[row["source_id"]]
        for n, row in enumerate(selected):
            src = archive[row["source_id"]]
            result.append(dict(id=f"{record['id']}.source_anchor.{n}", memory=record["id"],
                               kind="key_sentence", variant="source_anchor_v1", text=row["text"],
                               source_refs=[row["source_id"]], source_start=row["start"],
                               source_end=row["end"], source_date=src.get("date"),
                               source_session=src.get("session"), anchors=sorted(row["anchors"])))
    return result
