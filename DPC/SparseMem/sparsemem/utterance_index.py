"""Source-exact retrieval keys for frozen SparseMem records."""
import re

from .history import fragments


KIND = "utterance"
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)]*(?=\s|$)|[。！？]+[”’」』）]*|\n+")


def _spans(text, tokenizer, max_tokens):
    if tokenizer.count(text) <= max_tokens:
        yield 0, len(text)
        return
    ends = [match.end() for match in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    start = 0
    for end in ends:
        for left, right in fragments(text[start:end], tokenizer, max_tokens):
            yield start + left, start + right
        start = end


def utterance_entries(records, archive, tokenizer, max_tokens=96):
    """Index every nonempty primary span, with offsets relative to its source."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    entries = []
    for record in records:
        for source_id in record["primary"]:
            source = archive[source_id]
            text = source["text"]
            for start, end in _spans(text, tokenizer, max_tokens):
                span = text[start:end]
                if not span.strip():
                    continue
                start += len(span) - len(span.lstrip())
                end -= len(span) - len(span.rstrip())
                entries.append(dict(
                    id=f"{record['id']}.{KIND}.{source_id}.{start}-{end}",
                    memory=record["id"], kind=KIND, text=text[start:end],
                    source_refs=[source_id], source_start=start, source_end=end,
                    role=source["role"], date=source.get("date"), session=source["session"],
                ))
    return entries
