from dpc_mem.retrieval import Index, rrf, terms
from .dependency import render, names, TIME, temporal
from .utterance_index import KIND as UTTERANCE_KIND

import re
import math
import numpy as np


# LongMemEval-S contains a distinct family of questions that cannot be
# answered reliably from the first few nearest memories: totals, counts,
# averages, comparisons, and questions asking for all/different events. The
# detector is deterministic and adds no LLM call.
AGGREGATE_RE = re.compile(
    r"\b(?:how many|how much|how long|how often|what is the total|what was the total|"
    r"total amount|in total|combined|altogether|overall|average|mean|"
    r"all of|across all|different|each|number of|most|least|compare|"
    r"more than|less than|how many times)\b",
    re.I,
)

AGGREGATE_STOP = set("""
    "a an the i me my we our you your did do does have has had am is are was were "
    "be been being how many much long often what when where who which of for to "
    "in on at from by with and or as than this that these those last next past "
    "month months week weeks year years day days time times total amount number "
    "different all each every most least overall average mean combined altogether "
    "across recently currently ever ago before after since into during need pick "
    "up return store spend spent get got go went visit visited own"
""".split())


def is_aggregate_question(question):
    """Return whether a question needs cross-record enumeration/aggregation."""
    return bool(AGGREGATE_RE.search(question or ""))


def _norm_word(word):
    word = word.lower().strip("_'\"")
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _content_terms(text):
    result = set()
    for raw in re.findall(r"[a-z0-9_]+", text.lower()):
        word = _norm_word(raw)
        if word not in AGGREGATE_STOP and len(word) > 2:
            result.add(word)
    return result


def _raw_candidates(archive, question, limit):
    """Rank source fragments for aggregate questions without another model call."""
    qterms = _content_terms(question)
    if not qterms or not limit:
        return []
    rows = []
    df = {}
    source_terms = {}
    for key, source in archive.items():
        terms = _content_terms(source["text"])
        source_terms[key] = terms
        for term in terms:
            df[term] = df.get(term, 0) + 1
    total = max(len(source_terms), 1)
    for key, source in archive.items():
        overlap = qterms.intersection(source_terms[key])
        if not overlap:
            continue
        score = sum(math.log((total + 1) / (df[t] + 1)) for t in overlap)
        score += 0.35 * len(overlap)
        if source.get("role") == "user":
            score += 0.45
        if re.search(r"\d|\$|%", source["text"]):
            score += 0.15
        rows.append((score, source.get("order", 0), key))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [key for _, _, key in rows[:limit]]


def _raw_snippets(archive, question, limit, diversify=True):
    """Extract compact source snippets that contain query-specific content words."""
    qterms = _content_terms(question)
    if not qterms or not limit:
        return []
    source_terms = {key: _content_terms(source["text"]) for key, source in archive.items()}
    df = {}
    for values in source_terms.values():
        for term in values:
            df[term] = df.get(term, 0) + 1
    total = max(len(source_terms), 1)
    rows = []
    for key, source in archive.items():
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", source["text"]) if p.strip()]
        if not parts:
            parts = [source["text"].strip()]
        local = []
        for i, part in enumerate(parts):
            overlap = qterms.intersection(_content_terms(part))
            if not overlap:
                continue
            # Include a short neighboring fragment when the hit is a pronoun or
            # a continuation, while keeping long source messages compact.
            snippet = part
            if len(part.split()) < 9 and i + 1 < len(parts):
                snippet = part + " " + parts[i + 1]
            score = sum(math.log((total + 1) / (df[t] + 1)) for t in overlap)
            score += 0.35 * len(overlap)
            if source.get("role") == "user":
                score += 0.45
            if re.search(r"\d|\$|%", snippet):
                score += 0.15
            local.append((score, snippet))
        if not local:
            continue
        local.sort(key=lambda row: -row[0])
        # At most two snippets per source fragment avoids flooding the Reader
        # with repeated sentences from one long message.
        for rank, (score, snippet) in enumerate(local[:2]):
            rows.append((score - rank * 0.05, source.get("order", 0), key, snippet))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    # User turns are the primary factual channel in LongMemEval-S. Keep a
    # small assistant fallback because a few questions are answered only in an
    # assistant turn. A session cap prevents one long conversation from
    # consuming the whole evidence budget.
    user_rows = [row for row in rows if archive[row[2]].get("role") == "user"]
    other_rows = [row for row in rows if archive[row[2]].get("role") != "user"]
    ordered = user_rows[: int(limit * 0.8)] + other_rows[: max(1, int(limit * 0.2))]
    ordered.sort(key=lambda row: (-row[0], row[1], row[2]))
    if not diversify:
        return ordered[:limit]
    result = []
    per_session = {}
    sessions_seen = set()
    # First reserve one best hit per session. This is essential for questions
    # whose answer is distributed across many independently dated sessions.
    for row in ordered:
        session = archive[row[2]].get("session")
        if session in sessions_seen:
            continue
        result.append(row)
        sessions_seen.add(session)
        per_session[session] = 1
        if len(result) >= limit:
            return result
    for row in ordered:
        session = archive[row[2]].get("session")
        if per_session.get(session, 0) >= 8:
            continue
        result.append(row)
        per_session[session] = per_session.get(session, 0) + 1
        if len(result) >= limit:
            break
    return result


def _semantic_source_candidates(index, query_vector, archive, limit):
    """Project memory-entry similarity back to source fragments."""
    if not index.entries or not limit:
        return []
    q = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm == 0:
        return []
    scores = index.vectors @ (q / norm)
    source_scores = {}
    for i, entry in enumerate(index.entries):
        score = float(scores[i])
        for source_id in entry.get("source_refs", []):
            source_scores[source_id] = max(source_scores.get(source_id, -1.0), score)
    rows = []
    for source_id, score in source_scores.items():
        source = archive.get(source_id)
        if source is None:
            continue
        # User turns are the factual anchor, while assistant turns remain
        # eligible when a user turn has no matching semantic entry.
        score += 0.025 if source.get("role") == "user" else 0.0
        rows.append((score, source.get("order", 0), source_id))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return rows[:limit]


def _message_key(source_id):
    """Return the stable session/message part of a source id.

    A long user or assistant turn can be split into several indexed spans.  A
    query hit on one span should make its sibling spans eligible, but this is
    done only at query time and does not change the frozen memory archive.
    """
    return source_id.split("@", 1)[0]


def _turn_key(source_id):
    """Return (session, message number) for a prepared source id."""
    head = _message_key(source_id)
    match = re.match(r"^(S\d+):M(\d+)$", head)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _sentence_parts(text):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _evidence_snippet(source, qterms, max_words=90):
    """Make a compact evidence view of a source span.

    Semantic candidates often have different wording from the question.  For
    those candidates, keeping numeric/date sentences as well as lexical hits
    preserves the values needed by totals and comparisons without passing a
    whole multi-thousand-token assistant response to the Reader.
    """
    text = source.get("text", "").strip()
    parts = _sentence_parts(text)
    if not parts:
        return ""
    selected = []
    for part in parts:
        pterms = _content_terms(part)
        has_query = bool(qterms.intersection(pterms))
        has_value = bool(
            re.search(
                r"(?:\$\s?\d|\b\d+(?:\.\d+)?\s?(?:hours?|days?|weeks?|months?|years?|items?|times?|people|fish|plants?|projects?|pieces?|appointments?)\b|\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b)",
                part,
                re.I,
            )
        )
        if has_query or has_value:
            selected.append(part)
    if not selected:
        selected = parts[:1]
    # Preserve source order and remove repeated identical sentences.
    seen = set()
    selected = [p for p in selected if not (p in seen or seen.add(p))]
    words = []
    for part in selected:
        remaining = max_words - len(words)
        if remaining <= 0:
            break
        words.extend(part.split()[:remaining])
    return " ".join(words)


def _anchor_terms(archive, question, raw_candidates, raw_snippets, semantic_sources):
    """Add a small set of terms from the strongest query-side anchors.

    Cross-session questions often mention a category (for example, "items")
    while the answer turn names the concrete object (for example, "boots").
    Borrowing terms from the top user/source hits lets adjacent answer turns
    retain that concrete object without requiring an extra query rewrite call.
    """
    terms_out = set(_content_terms(question))
    keys = []
    for key in raw_candidates[:8]:
        keys.append(key)
    for _, _, key, _ in raw_snippets[:8]:
        keys.append(key)
    for _, _, key in semantic_sources[:8]:
        keys.append(key)
    seen = set()
    for key in keys:
        if key in seen or key not in archive:
            continue
        seen.add(key)
        values = [w for w in _content_terms(archive[key].get("text", "")) if len(w) >= 4]
        # The first few content terms are usually the concrete entities in a
        # short user turn; cap expansion to prevent topic drift.
        terms_out.update(values[:12])
    return terms_out


def _aggregate_coverage_candidates(archive, records, by, order, semantic_sources,
                                   raw_candidates, raw_snippets, question,
                                   query_terms=None, limit=220):
    """Merge query-time evidence channels with message-span expansion.

    The score is intentionally deterministic.  It rewards semantic retrieval,
    lexical overlap, ranked memory records, user turns, and value-bearing text.
    No new LLM call or construction step is introduced.
    """
    qterms = query_terms or _content_terms(question)
    scores = {}
    reasons = {}

    def add(key, score, reason):
        if key not in archive:
            return
        scores[key] = max(scores.get(key, -1e9), float(score))
        reasons.setdefault(key, set()).add(reason)

    for rank, mid in enumerate(order):
        r = by.get(mid)
        if not r:
            continue
        base = 4.0 - min(rank, 40) * 0.035
        for key in r.get("primary", []) + r.get("links", []):
            add(key, base, "memory")
        for fact in r.get("facts", []):
            for key in fact.get("source_ids", []):
                add(key, base + 0.25, "fact")

    for rank, (score, _, key) in enumerate(semantic_sources):
        add(key, 3.8 + float(score) + max(0, 1.0 - rank * 0.006), "semantic")
    for rank, key in enumerate(raw_candidates):
        add(key, 3.2 + max(0, 0.8 - rank * 0.004), "lexical_source")
    for rank, (score, _, key, _) in enumerate(raw_snippets):
        add(key, 3.0 + float(score) + max(0, 0.8 - rank * 0.003), "lexical_snippet")

    # Expand hit spans to siblings in the same original message.  This recovers
    # facts that were split from a matched span while keeping the expansion
    # bounded and score-lower than the direct hit.
    by_message = {}
    by_turn = {}
    by_session = {}
    for key, source in archive.items():
        by_message.setdefault(_message_key(key), []).append(key)
        turn = _turn_key(key)
        if turn is not None:
            by_turn.setdefault(turn, []).append(key)
        by_session.setdefault(source.get("session"), []).append(key)
    direct = list(scores)
    for key in direct:
        message = _message_key(key)
        parent = scores[key]
        for sibling in by_message.get(message, []):
            if sibling == key:
                continue
            sibling_score = parent - 0.18
            if archive[sibling].get("role") == "user":
                sibling_score += 0.10
            add(sibling, sibling_score, "message_sibling")

        # In the benchmark, many answers are in the assistant turn immediately
        # following a user request.  Expand one neighboring turn in either
        # direction so a hit on the request can recover its answer without
        # adding a new model call or changing construction.
        turn = _turn_key(key)
        if turn is not None:
            session, number = turn
            for neighbor in (number - 1, number + 1):
                for sibling in by_turn.get((session, neighbor), []):
                    sibling_score = parent - 0.24
                    if archive[sibling].get("role") == "user":
                        sibling_score += 0.06
                    add(sibling, sibling_score, "adjacent_turn")

        # A cross-session answer may mention several entities in one session,
        # each in a different user turn.  Once a session has a strong hit, make
        # its other user turns eligible at a lower score; the compact packer
        # will keep only value-bearing/query-relevant sentences.
        session = archive[key].get("session")
        session_rows = by_session.get(session, [])
        for sibling in session_rows:
            if archive[sibling].get("role") != "user":
                continue
            sibling_terms = _content_terms(archive[sibling].get("text", ""))
            overlap = len(qterms.intersection(sibling_terms))
            has_value = bool(re.search(r"\d|\$|%", archive[sibling].get("text", "")))
            sibling_score = parent - 0.34 + overlap * 0.12 + (0.08 if has_value else 0.0)
            add(sibling, sibling_score, "session_expansion")

    # Keep user turns slightly ahead of assistant suggestions when the evidence
    # is otherwise tied.  Assistant turns remain eligible for answer-only facts.
    rows = []
    for key, score in scores.items():
        source = archive[key]
        overlap = len(qterms.intersection(_content_terms(source.get("text", ""))))
        value = bool(re.search(r"\d|\$|%", source.get("text", "")))
        role_bonus = 0.22 if source.get("role") == "user" else 0.0
        rows.append((score + overlap * 0.08 + role_bonus + (0.05 if value else 0.0), source.get("order", 0), key))
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    return rows[:limit], reasons


def keys(records, a, key_sentences=0):
    entries = []
    for r in records:

        def add(kind, text, refs):
            entries.append(
                dict(
                    id=f"{r['id']}.{len(entries)}",
                    memory=r["id"],
                    kind=kind,
                    text=text,
                    source_refs=refs,
                )
            )

        primary = r["primary"]
        raw = "\n".join(render(a, k) for k in primary)
        add("raw", raw, primary)
        t = " ".join(
            (v["event_start"] + " " + v["event_end"])
            for v in r["temporal"]
            if v["event_start"]
        )
        if r["links"] or t:
            refs = sorted(primary + r["links"], key=lambda k: a[k]["order"])
            add("linked", t + "\n" + "\n".join(render(a, k) for k in refs), refs)
        for f in r["facts"]:
            add("refined", f["text"], f["source_ids"])
    if key_sentences:
        entries.extend(key_sentence_entries(records, a, max_per_record=key_sentences))
    return entries


def key_sentence_entries(records, archive, max_per_record=2):
    """Build deterministic, source-verbatim extra keys, without an LLM.

    Each entry points back to its owning record and exact source offsets. The
    rarity score is computed only from the frozen history, never the question.
    """
    candidates = {}
    df = {}
    for r in records:
        rows = []
        seen = set()
        for source_id in r["primary"]:
            text = archive[source_id]["text"]
            for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|(?=\n|$))", text):
                start, end = match.span()
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                sentence = text[start:end]
                words = set(terms(sentence))
                content = {w for w in words if len(w) > 3 and w not in AGGREGATE_STOP}
                if not (25 <= len(sentence) <= 400 and len(content) >= 3):
                    continue
                normalized = " ".join(sentence.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                rows.append((source_id, start, end, sentence, content))
        candidates[r["id"]] = rows
        for word in set().union(*(row[4] for row in rows)) if rows else ():
            df[word] = df.get(word, 0) + 1
    total = max(len(records), 1)
    entries = []
    for r in records:
        rows = candidates[r["id"]]
        def score(row):
            source_id, start, _, sentence, words = row
            rarity = sum(math.log((total + 1) / (df[w] + 1)) for w in words)
            return (-rarity / math.sqrt(len(words)), archive[source_id]["order"], start)
        for number, (source_id, start, end, sentence, _) in enumerate(sorted(rows, key=score)[:max_per_record]):
            entries.append(dict(id=f"{r['id']}.key_sentence.{number}", memory=r["id"],
                                kind="key_sentence", text=sentence, source_refs=[source_id],
                                source_start=start, source_end=end))
    return entries



def _utterance_hits(index, trace, order, cfg):
    """Keep high-ranked exact spans for the final records, grouped by owner."""
    if not cfg.get("utterance_index", False):
        return {}
    selected = set(order)
    scores = {}
    for channel in ("dense", "lexical"):
        for rank, key in enumerate(trace.get(channel, [])[:cfg["channel_k"]], 1):
            entry = index.by_id[key]
            if entry.get("kind") == UTTERANCE_KIND and entry["memory"] in selected:
                scores[key] = scores.get(key, 0) + 1 / (cfg["rrf_k"] + rank)
    grouped = {}
    for key in sorted(scores, key=lambda k: (-scores[k], k)):
        entry = index.by_id[key]
        grouped.setdefault(entry["memory"], []).append(entry)
    return grouped

def retrieve(index, encoder, records, a, question, qdate, cfg, tok, reranker=None):
    by = {r["id"]: r for r in records}
    aggregate = cfg.get("aggregate_mode", True) and (bool(cfg.get("aggregate_query")) or is_aggregate_question(question))
    channel_k = cfg["channel_k"]
    fused_k = cfg["fused_k"]
    expand_k = cfg["expand_k"]
    final_k = cfg["final_k"]
    if aggregate:
        channel_k = max(channel_k, cfg.get("aggregate_channel_k", channel_k))
        fused_k = max(fused_k, cfg.get("aggregate_fused_k", fused_k))
        expand_k = max(expand_k, cfg.get("aggregate_expand_k", expand_k))
        final_k = max(final_k, cfg.get("aggregate_final_k", final_k))
    n = len(index.entries)
    query_vector = encoder.embed([question], role="query")[0]
    _, trace = index.search(
        question,
        query_vector,
        dict(dense_k=n, lexical_k=n, bm25_k1=1.5, bm25_b=0.75, rrf_k=cfg["rrf_k"]),
    )
    channels = [
        [*dict.fromkeys(index.by_id[k]["memory"] for k in trace.get(channel, []))][
            : channel_k
        ]
        for channel in ("dense", "lexical")
    ]
    order, _ = rrf(channels, cfg["rrf_k"])
    order = order[:fused_k]
    initial = list(order)
    entities = names(question)
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question))
    lex = set(terms(question))
    for m in TIME.finditer(question):
        t = temporal(m[0], qdate)
        dates.update(v for v in (t["event_start"], t["event_end"]) if v)
    extra = []
    for r in records:
        if r["id"] in order:
            continue
        entity = entities.intersection(r["entities"])
        date_match = any(
            v["event_start"]
            and any(v["event_start"] <= d <= v["event_end"] for d in dates)
            for v in r["temporal"]
        )
        # Require an independent lexical cue for an ambiguous name match.
        overlap = (
            lex.intersection(terms(" ".join(a[k]["text"] for k in r["primary"])))
            - entity
        )
        if date_match or (entity and overlap):
            extra.append((int(date_match) + len(entity), r["id"]))
    expansion = [
        k for _, k in sorted(extra, key=lambda v: (-v[0], v[1]))[:expand_k]
    ]
    order.extend(expansion)
    if reranker is None and expansion:
        # Merge soft entity/time matches into the candidate order. Appending after top-40 alone would never reach top-12.
        order, _ = rrf([initial, expansion], cfg["rrf_k"])
    if reranker is not None and order:
        scores = reranker.predict(
            [
                (question, "\n".join(render(a, k) for k in by[m]["primary"]))
                for m in order
            ]
        )
        order = [m for _, m in sorted(zip(scores, order), key=lambda v: -v[0])]
    order = order[:final_k]
    # Capture only keys that actually competed in a channel's top memory
    # candidates. Several keys may retrieve one record, which remains unique.
    key_hits = {}
    for channel in ("dense", "lexical"):
        seen_memories = set()
        for rank, entry_id in enumerate(trace.get(channel, []), 1):
            entry = index.by_id[entry_id]
            memory_id = entry["memory"]
            if memory_id not in seen_memories and len(seen_memories) >= channel_k:
                break
            seen_memories.add(memory_id)
            if entry.get("kind") == "key_sentence" and memory_id in order:
                row = dict(entry_id=entry_id, memory=memory_id, source_id=entry["source_refs"][0],
                           text=entry["text"], channel=channel, channel_rank=rank)
                if memory_id not in key_hits or rank < key_hits[memory_id]["channel_rank"]:
                    key_hits[memory_id] = row
    utterance_hits = _utterance_hits(index, trace, order, cfg)
    packed = {}
    notes = []
    utterance_notes = {}
    raw_candidates = _raw_candidates(
        a, question, cfg.get("aggregate_raw_k", 0)
    ) if aggregate else []
    semantic_sources = _semantic_source_candidates(
        index,
        query_vector,
        a,
        cfg.get("aggregate_semantic_k", 0),
    ) if aggregate else []
    raw_snippets = _raw_snippets(
        a,
        question,
        cfg.get("aggregate_snippet_k", 0),
        cfg.get("aggregate_diversify_sessions", True),
    ) if aggregate else []
    raw_snippet_keys = {row[2] for row in raw_snippets}
    evidence_terms = (
        _anchor_terms(a, question, raw_candidates, raw_snippets, semantic_sources)
        if aggregate else _content_terms(question)
    )
    coverage_rows, coverage_reasons = ([], {})
    if aggregate and cfg.get("aggregate_coverage", False):
        coverage_rows, coverage_reasons = _aggregate_coverage_candidates(
            a,
            records,
            by,
            order,
            semantic_sources,
            raw_candidates,
            raw_snippets,
            question,
            evidence_terms,
            cfg.get("aggregate_coverage_k", 220),
        )
    evidence_budget = (
        cfg.get("aggregate_evidence_tokens", cfg["evidence_tokens"])
        if aggregate else cfg["evidence_tokens"]
    )

    def evidence():
        return "\n".join(
            [*notes, *[note for k, note in utterance_notes.values() if k not in packed],
             *[packed[s] for s in sorted(packed, key=lambda k: a[k]["order"])]]
        )

    def put(k):
        if cfg.get("user_evidence_only") and a[k].get("role") != "user":
            return
        if k in packed:
            return
        packed[k] = render(a, k)
        if tok.count(evidence()) > evidence_budget:
            del packed[k]

    def put_note(note):
        notes.append(note)
        if tok.count(evidence()) > evidence_budget:
            notes.pop()
            return False
        return True

    def put_source_snippet(key, label="Evidence"):
        if key not in a:
            return
        source = a[key]
        snippet = _evidence_snippet(source, evidence_terms, max_words=96)
        if not snippet:
            return
        reason = ",".join(sorted(coverage_reasons.get(key, ())))
        suffix = f" | {reason}" if reason else ""
        put_note(
            f"[{label} {key} | {source['role']} | {source['date'] or 'unknown date'}{suffix}] {snippet}"
        )

    # Reserve source coverage across distinct hits before additional representations.
    # Aggregate questions get all primary fragments for the top records, because
    # one fragment per record systematically loses events needed for totals.
    key_in_context = []

    def put_utterance(entry):
        source_id = entry["source_refs"][0]
        source = a[source_id]
        if source_id in packed or (cfg.get("user_evidence_only") and source["role"] != "user"):
            return
        if source["text"][entry["source_start"]:entry["source_end"]] != entry["text"]:
            raise ValueError("Utterance text does not match its source offsets")
        if entry["text"] == source["text"]:
            put(source_id)
            return
        note = f"[Utterance {source_id} | {source['role']} | {source['date'] or 'unknown date'}] {entry['text']}"
        utterance_notes[entry["id"]] = (source_id, note)
        if tok.count(evidence()) > evidence_budget:
            del utterance_notes[entry["id"]]

    for memory_id in order:
        if memory_id in utterance_hits:
            put_utterance(utterance_hits[memory_id][0])
    if not aggregate and cfg.get("key_sentence_context", cfg.get("key_sentence_index", False)):
        for m in order:
            hit = key_hits.get(m)
            if hit is None:
                continue
            source = a[hit["source_id"]]
            if cfg.get("user_evidence_only") and source.get("role") != "user":
                continue
            entry = index.by_id[hit["entry_id"]]
            if source["text"][entry["source_start"]:entry["source_end"]] != hit["text"]:
                raise ValueError("Key sentence does not match its source offsets")
            note = f"[KeySentence {hit['source_id']} | {source['role']} | {source['date'] or 'unknown date'}] {hit['text']}"
            if put_note(note):
                key_in_context.append(hit)
    if aggregate:
        if cfg.get("aggregate_coverage", False):
            # Reserve part of the budget for the original v3 lexical channel.
            # It is often the cleanest evidence for ordinary aggregate
            # questions, while the coverage channel recovers paraphrases and
            # split answer turns afterward.
            raw_reserve = min(
                evidence_budget,
                max(0, int(cfg.get("aggregate_raw_reserve", 4200))),
            )
            for _, _, key, snippet in raw_snippets:
                if tok.count(evidence()) >= raw_reserve:
                    break
                source = a[key]
                put_note(
                    f"[RawPrimary {key} | {source['role']} | {source['date'] or 'unknown date'}] {snippet}"
                )

            # Compact extracted facts get a second reserved pass.  This avoids
            # the previous failure mode where raw snippets consumed the entire
            # budget before the memory representation could be read.
            fact_count = 0
            for m in order:
                r = by[m]
                for f in r.get("facts", []):
                    refs = " ".join(f.get("source_ids", []))
                    put_note(f"[MemoryFact {refs}] {f['text']}")
                    fact_count += 1
                    if fact_count >= cfg.get("aggregate_fact_k", 64):
                        break
                if fact_count >= cfg.get("aggregate_fact_k", 64):
                    break

            # Add compact views of semantic/memory/lexical candidates.  A hit
            # on one split span also brings sibling spans from that same turn.
            for _, _, key in coverage_rows:
                put_source_snippet(key, "Evidence")

            # Ensure a small lexical fallback remains available when a semantic
            # hit has no readable sentence-level match.
            for _, _, key, snippet in raw_snippets:
                if tok.count(evidence()) >= int(evidence_budget * 0.98):
                    break
                if key in a:
                    source = a[key]
                    put_note(
                        f"[RawFallback {key} | {source['role']} | {source['date'] or 'unknown date'}] {snippet}"
                    )
        else:
            # Sentence-level source snippets come first. This is the v3/v5
            # behavior kept for exact reproducibility of earlier pilots.
            for _, _, key, snippet in raw_snippets:
                source = a[key]
                put_note(
                    f"[RawSnippet {key} | {source['role']} | {source['date'] or 'unknown date'}] {snippet}"
                )
            # Add semantically related source fragments that use different words
            # from the question (e.g. "fish" described as specific species).
            semantic_seen_sessions = {}
            semantic_order = []
            seen_semantic_sessions = set()
            for row in semantic_sources:
                session = a[row[2]].get("session")
                if session in seen_semantic_sessions:
                    continue
                semantic_order.append(row)
                seen_semantic_sessions.add(session)
            semantic_order.extend(
                row for row in semantic_sources if row not in semantic_order
            )
            for _, _, key in semantic_order:
                session = a[key].get("session")
                if semantic_seen_sessions.get(session, 0) >= 4:
                    continue
                put(key)
                semantic_seen_sessions[session] = semantic_seen_sessions.get(session, 0) + 1
            for m in order:
                for k in by[m]["primary"]:
                    put(k)
                for k in by[m]["links"]:
                    put(k)
            # Recover facts whose memory record was omitted or only partially refined
            # during construction using a deterministic source scan.
            for k in raw_candidates:
                put(k)
    else:
        for m in order:
            r = by[m]
            primary = [k for k in r["primary"] if not cfg.get("user_evidence_only") or a[k].get("role") == "user"]
            if not primary:
                continue
            target = max(primary, key=lambda k: len(lex.intersection(terms(a[k]["text"]))))
            put(target)
    for memory_id in order:
        for entry in utterance_hits.get(memory_id, ())[1:]:
            put_utterance(entry)
    for m in order:
        r = by[m]
        for f in r["facts"]:
            # The old fact may have been generated from a mixed-role block,
            # even when its cited source is a user turn. Omit it in this
            # evidence-only experiment instead of implying a clean rebuild.
            if cfg.get("user_evidence_only"):
                continue
            covered = any(k in packed for k in f["source_ids"])
            if aggregate:
                covered = covered or any(k in raw_snippet_keys for k in f["source_ids"])
            if not covered:
                continue
            note = "[Memory " + " ".join(f["source_ids"]) + "] " + f["text"]
            put_note(note)
        if not aggregate:
            for k in r["primary"] + r["links"]:
                put(k)
    utterance_trace = {}
    if cfg.get("utterance_index", False):
        hits = [entry for memory_id in order for entry in utterance_hits.get(memory_id, ())]
        utterance_trace = dict(
            utterance_hits=hits,
            utterance_in_context=[entry for entry in hits
                                  if entry["source_refs"][0] in packed or entry["id"] in utterance_notes],
        )
    return dict(
        **utterance_trace,
        evidence=evidence(),
        tokens=tok.count(evidence()),
        packed=list(packed),
        ranked=order,
        channels=channels,
        initial=initial,
        expansion=expansion,
        aggregate=aggregate,
        aggregate_raw_candidates=raw_candidates,
        aggregate_semantic_sources=[
            {"source_id": key, "score": score}
            for score, _, key in semantic_sources
        ],
        aggregate_raw_snippets=[
            {"source_id": key, "text": snippet, "score": score}
            for score, _, key, snippet in raw_snippets
        ],
        aggregate_coverage_sources=[
            {"source_id": key, "score": score, "reasons": sorted(coverage_reasons.get(key, ())) }
            for score, _, key in coverage_rows
        ],
        aggregate_memory_k=final_k if aggregate else 0,
        key_sentence_hits=list(key_hits.values()),
        key_sentence_in_context=key_in_context,
        reranker_pairs=(len(initial) + len(expansion)) if reranker is not None else 0,
    )
