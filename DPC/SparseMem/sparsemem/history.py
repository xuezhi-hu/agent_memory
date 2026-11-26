"""Project only historical fields, then segment without dropping source spans."""

from datetime import datetime
import re
import numpy as np


def stamp(value):
    if not value:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d", "%I:%M %p on %d %B, %Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError("Unrecognized timestamp: " + value)


def project(sample):
    rows, dates, ids = (
        sample[k]
        for k in ("haystack_sessions", "haystack_dates", "haystack_session_ids")
    )
    if len({len(rows), len(dates), len(ids)}) != 1 or len(set(ids)) != len(ids):
        raise ValueError("Misaligned or duplicate sessions")
    # Unknown times follow dated sessions in original order; never used as cross-session anchors.
    order = sorted(range(len(rows)), key=lambda i: stamp(dates[i]) or datetime.max)
    turns, mapping = [], {}
    for n, i in enumerate(order, 1):
        sid = f"S{n:04d}"
        mapping[sid] = ids[i]
        for j, m in enumerate(rows[i], 1):
            if m["role"] not in ("user", "assistant", "system") or not isinstance(
                m["content"], str
            ):
                raise ValueError("Malformed historical message")
            turns.append(
                dict(
                    id=f"{sid}:M{j}",
                    session=sid,
                    role=m["role"],
                    date=dates[i] or "",
                    text=m["content"],
                )
            )
    return turns, mapping


def project_locomo(history):
    """Use the same internal shape, with participant names in the speaker slot."""
    sessions = {}
    turns = []
    for turn in history:
        original_session = turn["session_id"]
        if original_session not in sessions:
            sessions[original_session] = f"S{len(sessions) + 1:04d}"
        date = stamp(turn["observed_at"])
        turns.append(dict(
            id=turn["id"], session=sessions[original_session], role=turn["speaker"],
            date=date.strftime("%Y-%m-%d %H:%M") if date else "", text=turn["text"],
            participant_dialogue=True,
        ))
    return turns


def fragments(text, tok, cap):
    start = 0
    while start < len(text):
        if tok.count(text[start:]) <= cap:
            yield start, len(text)
            return
        lo, hi = start + 1, len(text)
        if tok.count(text[start:lo]) > cap:
            raise ValueError("Token cap below one character")
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tok.count(text[start:mid]) <= cap:
                lo = mid
            else:
                hi = mid - 1
        ends = [m.end() + start for m in re.finditer(r"[.!?]\s+|\n+", text[start:lo])]
        end = ends[-1] if ends and ends[-1] - start > (lo - start) // 2 else lo
        yield start, end
        start = end


def segment(turns, tok, encoder, cfg, mode):
    archive = {}
    rounds = []
    for t in turns:
        for a, b in list(fragments(t["text"], tok, cfg["block_tokens"])) or [(0, 0)]:
            sid = f"{t['id']}@{a}-{b}"
            archive[sid] = {
                **t,
                "text": t["text"][a:b],
                "start": a,
                "end": b,
                "order": len(archive),
            }
            if (
                rounds
                and a == 0
                and (
                    (t["role"] == "assistant" and archive[rounds[-1][-1]]["role"] == "user")
                    or (t.get("participant_dialogue") and len(rounds[-1]) == 1
                        and archive[rounds[-1][-1]]["role"] != t["role"])
                )
                and archive[rounds[-1][-1]]["session"] == t["session"]
                and tok.count("\n".join(archive[k]["text"] for k in rounds[-1] + [sid]))
                <= cfg["block_tokens"]
            ):
                rounds[-1].append(sid)
            else:
                rounds.append([sid])
    if not rounds:
        return archive, []
    texts = ["\n".join(archive[k]["text"] for k in row) for row in rounds]
    vectors = encoder.embed(texts) if mode not in ("raw_round", "raw_session") else None
    blocks = []
    for i, row in enumerate(rounds):
        join = bool(
            blocks and archive[blocks[-1][-1]]["session"] == archive[row[0]]["session"]
        )
        if join and mode != "raw_session":
            join = (
                mode != "raw_round"
                and tok.count("\n".join(archive[k]["text"] for k in blocks[-1] + row))
                <= cfg["block_tokens"]
            )
            if join:
                x, y = vectors[i - 1 : i + 1]
                cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                join = cosine >= cfg["split_threshold"] or bool(
                    re.match(
                        r"\s*(yes|no|because|actually|that one|it was)\b",
                        texts[i],
                        re.I,
                    )
                )
        if join:
            blocks[-1].extend(row)
        else:
            blocks.append(list(row))
    return archive, blocks
