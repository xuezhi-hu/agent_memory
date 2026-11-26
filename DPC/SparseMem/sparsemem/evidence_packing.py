"""User-only query-aware packing over already retrieved memory records.

This module never searches the complete archive and never reads benchmark
labels.  It ranks sentences from the primary and linked sources of the
supplied candidate records and keeps source identifiers for later auditing.
"""
import math
import re
from collections import Counter

from .recall import _content_terms, _sentence_parts


def pack_local_evidence(question, ranked, records, archive, tokenizer, budget,
                        primary_k=20, candidate_k=40):
    by_id = {record['id']: record for record in records}
    unique_sources = {}
    for rank, memory_id in enumerate(ranked[:candidate_k]):
        for source_id in by_id[memory_id].get('primary', []) + by_id[memory_id].get('links', []):
            if source_id in archive and archive[source_id].get('role') == 'user':
                unique_sources.setdefault(source_id, (rank, memory_id))

    qterms = _content_terms(question)
    candidates = []
    for source_id, (rank, memory_id) in unique_sources.items():
        source = archive[source_id]
        parts = _sentence_parts(source.get('text', ''))
        for number, sentence in enumerate(parts):
            # Previous sentence is included when the hit uses an anaphor;
            # deduplication below removes duplicate spans.
            if number and re.match(r'^(?:it|they|them|this|that|these|those|he|she|his|her|its|their)\b', sentence, re.I):
                sentence = parts[number - 1] + ' ' + sentence
            st = _content_terms(sentence)
            candidates.append(dict(source_id=source_id, memory=memory_id,
                                   memory_rank=rank, number=number, text=sentence,
                                   terms=st, overlap=qterms & st))
    df = Counter(t for row in candidates for t in row['terms'])
    n = len(candidates)
    # Source-level overlap lets a numerical fact compete even when the topic
    # appears only in a neighboring sentence of the same user message.
    source_overlap = {}
    for row in candidates:
        source_overlap.setdefault(row['source_id'], set()).update(row['overlap'])
    for row in candidates:
        lexical = sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5)) for t in row['overlap'])
        source_match = bool(source_overlap[row['source_id']])
        value = bool(re.search(r'\d|\$|%|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b', row['text'], re.I))
        row['score'] = (lexical + 0.2 * len(row['overlap']) +
                        (0.7 if value and source_match else 0) +
                        0.8 / (1 + row['memory_rank'] / 10) +
                        (0.15 if row['memory_rank'] < primary_k else 0))
        # A gentle cost penalty favors compact, relevant sentences without
        # discarding longer statements that carry several facts.
        row['score'] /= math.sqrt(max(1, tokenizer.count(row['text'])) / 24 + 1)
    candidates.sort(key=lambda row: (-row['score'], row['memory_rank'],
                                      archive[row['source_id']].get('order', 0), row['number']))
    chosen, notes, seen = [], [], set()
    used_sources, used_sessions, covered_terms = set(), set(), set()
    while candidates:
        # Prefer additional sources/sessions and uncovered question terms when
        # relevance is comparable; distinct events are never deduplicated.
        def priority(row):
            source = archive[row['source_id']]
            novelty = (0.3 * (row['source_id'] not in used_sources) +
                       0.15 * (source.get('session') not in used_sessions) +
                       0.15 * len(row['overlap'] - covered_terms))
            return row['score'] + novelty
        best = max(range(len(candidates)), key=lambda i: priority(candidates[i]))
        row = candidates.pop(best)
        # Exact deduplication only; do not merge similar but distinct facts.
        signature = (row['source_id'], row['text'])
        if signature in seen:
            continue
        seen.add(signature)
        source = archive[row['source_id']]
        note = f"[Evidence {row['source_id']} | user | {source.get('date') or 'unknown date'}] {row['text']}"
        candidate = '\n'.join(notes + [note])
        if tokenizer.count(candidate) > budget:
            continue
        notes.append(note)
        chosen.append({key: row[key] for key in ('source_id', 'memory', 'memory_rank', 'number', 'text', 'score')})
        used_sources.add(row['source_id'])
        used_sessions.add(source.get('session'))
        covered_terms.update(row['overlap'])
    text = '\n'.join(notes)
    return dict(evidence=text, tokens=tokenizer.count(text),
                selected_sentences=chosen, candidate_source_count=len(unique_sources),
                candidate_sentence_count=n,
                candidate_memories=list(ranked[:candidate_k]), primary_k=primary_k,
                candidate_k=candidate_k, local_only=True)
