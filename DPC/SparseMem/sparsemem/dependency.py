"""Observable cues and their supporting source IDs; no generative inference."""

import calendar
from collections import defaultdict
from datetime import timedelta
import re
from .history import stamp

REF = re.compile(
    r"\b(he|she|it|they|him|her|them|those|the former|the latter|this one|that one|the other one)\b",
    re.I,
)
SHORT = re.compile(
    r"^\s*(yes|no|sure|correct|exactly|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years? old|weeks?|days?))\b",
    re.I,
)
CONT = re.compile(r"^\s*(because|also|actually|instead)\b", re.I)
UPDATE = re.compile(
    r"\b(changed|switched|cancelled|canceled|corrected|no longer|rather than|rescheduled|instead|used to|(?:now |just )?moved to|increased (?:its |their )?offer)\b",
    re.I,
)
TIME = re.compile(
    r"\b(yesterday|today|tomorrow|(?:last|this|next) (?:week|month|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|recently|later|"
    + "|".join(calendar.month_name[1:])
    + r")\b",
    re.I,
)
STOP = set(
    "i we she he they it that this the my our a an yes no yesterday today tomorrow actually also because how what when where why which thanks sure later recently three which has wow hey good thanks yeah yep gonna since last by your you anything what kinda so now then there their his her its really appreciate sounds great cool family".split()
) | {m.lower() for m in calendar.month_name if m}

# Sentence-initial title-case words are common in conversational data and are
# not reliable entity anchors. Keep names that are repeated or explicitly
# presented as names, while filtering generic discourse words.
CAP_STOP = {
    "I", "The", "A", "An", "My", "Our", "We", "You", "Your", "This", "That",
    "It", "He", "She", "They", "Their", "His", "Her", "Its", "What", "How",
    "When", "Where", "Why", "Which", "Anything", "Since", "Last", "Now",
    "Then", "There", "Wow", "Hey", "Good", "Thanks", "Yeah", "Yep", "Gonna",
    "So", "Really", "Actually", "By", "Family", "Great", "Cool", "Sounds",
}


def names(text):
    return {
        x.lower()
        for x in re.findall(r"\b[A-Z][\w-]*\b", text)
        if x not in CAP_STOP and x.lower() not in STOP
    }


def temporal(expr, anchor):
    row = dict(expression=expr, event_start=None, event_end=None, precision="unknown")
    moment = stamp(anchor)
    if not moment:
        return row
    base = moment.date()
    e = expr.lower()
    first = last = None
    prec = "day"
    if e in ("yesterday", "today", "tomorrow"):
        first = last = base + timedelta(
            days={"yesterday": -1, "today": 0, "tomorrow": 1}[e]
        )
    elif e.endswith((" week", " month")):
        delta = {"last": -1, "this": 0, "next": 1}[e.split()[0]]
        if e.endswith("week"):
            first = base + timedelta(days=7 * delta - base.weekday())
            last = first + timedelta(days=6)
            prec = "week"
        else:
            n = base.year * 12 + base.month - 1 + delta
            first = base.replace(year=n // 12, month=n % 12 + 1, day=1)
            last = first.replace(day=calendar.monthrange(first.year, first.month)[1])
            prec = "month"
    elif e.startswith(("last ", "next ")) and e.split()[1] in [
        d.lower() for d in calendar.day_name
    ]:
        weekday = [d.lower() for d in calendar.day_name].index(e.split()[1])
        sign = -1 if e.startswith("last") else 1
        days = ((weekday - base.weekday()) * sign) % 7 or 7
        first = last = base + timedelta(days=sign * days)
    if first:
        row.update(event_start=str(first), event_end=str(last), precision=prec)
    return row


def render(archive, k):
    a = archive[k]
    return f"[{k} | {a['role']} | {a['date'] or 'unknown date'}] {a['text']}"


RESIDENCE = re.compile(
    r"\bI (?:currently |now |used to )?(?:live|lived|reside) in|\bI (?:have )?(?:now |just )?moved to",
    re.I,
)
MOVE = re.compile(r"\bI (?:have )?(?:now |just )?moved to", re.I)


def short_reply(text):
    words = re.findall(r"[\w'-]+", text)
    return 0 < len(words) <= 4 and not re.search(
        r"\b(?:I|we|they|he|she|is|was|are|went|quit)\b|\?", text, re.I
    )


def prepare(archive, blocks, tok, cfg, mode):
    records = []
    previous = []
    entity_index = defaultdict(list)
    residences = defaultdict(list)
    for ordinal, primary in enumerate(blocks):
        raw = mode in ("raw", "raw_block", "raw_round", "raw_session")
        disabled = raw or mode == "sparse_no_link"
        links = []
        traces = []
        times = []

        def attach(k):
            if k in primary or k in links:
                return True
            if phase_disabled or len(links) >= cfg["max_links"]:
                return False
            if (
                tok.count("\n".join(render(archive, s) for s in links + [k]))
                > cfg["link_tokens"]
            ):
                return False
            links.append(k)
            return True

        for phase in ("direct", "linked"):
            phase_disabled = disabled or phase == "direct"
            traces = []
            times = []
            for i, k in enumerate(primary):
                m = archive[k]
                text = m["text"]
                ents = names(text)
                near = [
                    s
                    for s in (previous[-cfg["lookback"] :] + primary[:i])[
                        -cfg["lookback"] :
                    ]
                    if archive[s]["session"] == m["session"]
                ]

                def cue(dim, d, r, why, support):
                    traces.append(
                        dict(
                            dim=dim,
                            initial=d,
                            resolution=r,
                            reason=why,
                            target=k,
                            evidence=support,
                        )
                    )

                if REF.search(text):
                    pool = [s for s in reversed(near) if names(archive[s]["text"])]
                    # Entity links stay within the current session. Cross-session
                    # state changes are handled explicitly by the U detector.
                    candidates = set().union(*(names(archive[s]["text"]) for s in pool))
                    aligned = [s for s in pool[:2] if attach(s)]
                    # Only an explicit naming construction earns full credit for a pronoun.
                    explicit = bool(
                        len(candidates) == 1
                        and aligned
                        and re.search(
                            r"\b(?:named|called)\s+[A-Z]", archive[aligned[0]]["text"]
                        )
                    )
                    cue(
                        "E",
                        1,
                        1 if explicit else 0.5 if aligned else 0,
                        "reference",
                        aligned,
                    )
                adjacent_question = bool(
                    near
                    and archive[near[-1]]["role"] != m["role"]
                    and "?" in archive[near[-1]]["text"]
                )
                nominal_reply = short_reply(text) and adjacent_question
                if SHORT.search(text) or CONT.search(text) or nominal_reply:
                    q = (
                        near[-1]
                        if near and archive[near[-1]]["role"] != m["role"]
                        else None
                    )
                    support = [q] if q and attach(q) else []
                    clear = bool(
                        support
                        and "?" in archive[q]["text"]
                        and (SHORT.search(text) or nominal_reply)
                    )
                    cue(
                        "C",
                        1 if SHORT.search(text) or nominal_reply else 0.5,
                        1 if clear else 0.5 if support else 0,
                        "reply/continuation",
                        support,
                    )
                for match in TIME.finditer(text):
                    if match[0].lower() in {
                        month.lower() for month in calendar.month_name if month
                    } and re.match(r"\s+\d{4}\b", text[match.end() :]):
                        continue  # The absolute month and year are explicit in the raw key.
                    t = temporal(match[0], m["date"])
                    if not phase_disabled:
                        times.append(dict(t, source_id=k))
                    anchored = t["event_start"] is not None
                    cue(
                        "T",
                        (
                            0.5
                            if match[0].lower()
                            in {v.lower() for v in calendar.month_name if v}
                            or match[0].lower() in ("recently", "later")
                            else 1
                        ),
                        1 if anchored and not phase_disabled else 0,
                        "relative/partial time",
                        [k] if anchored else [],
                    )
                if UPDATE.search(text) and not text.rstrip().endswith("?"):
                    if MOVE.search(text) and re.search(r"\bto\s+[A-Z][\w-]+", text):
                        history = residences[m["role"]] + [
                            s
                            for s in primary[:i]
                            if archive[s]["role"] == m["role"]
                            and RESIDENCE.search(archive[s]["text"])
                        ]
                        old = next(
                            (
                                s
                                for s in reversed(history)
                                if archive[s]["session"] == m["session"]
                                or (
                                    m["date"]
                                    and archive[s]["date"]
                                    and stamp(archive[s]["date"]) < stamp(m["date"])
                                )
                            ),
                            None,
                        )
                        if old:
                            covered = attach(old)
                            cue(
                                "U",
                                1,
                                1 if covered else 0,
                                "same speaker / residence change",
                                [old] if covered else [],
                            )
                        continue  # Explicit move with no old state is self-contained.

                    pool = list(reversed(near))
                    for ent in sorted(ents):
                        pool.extend(
                            s
                            for s in reversed(entity_index[ent])
                            if m["date"]
                            and archive[s]["date"]
                            and stamp(archive[s]["date"]) < stamp(m["date"])
                        )
                    related = [
                        s
                        for s in dict.fromkeys(pool)
                        if ents & names(archive[s]["text"]) and attach(s)
                    ][:2]
                    explicit = re.search(r"\bfrom\s+\S.+?\bto\s+\S", text, re.I)
                    cue(
                        "U",
                        1,
                        1 if explicit else 0.5 if related else 0,
                        "state change",
                        [k] if explicit else related,
                    )
            observed = {
                j: max(
                    (
                        t["initial"] * (1 - t["resolution"])
                        for t in traces
                        if t["dim"] == j
                    ),
                    default=0,
                )
                for j in "ECTU"
            }
            if phase == "direct":
                initial = dict(observed)
        dims = "ECTU"
        ds = {}
        rs = {}
        residual = 0
        for dim in dims:
            rows = [t for t in traces if t["dim"] == dim]
            ds[dim] = initial[dim]
            worst = max((t["initial"] * (1 - t["resolution"]) for t in rows), default=0)
            rs[dim] = 1 - worst / ds[dim] if ds[dim] else 0
            residual += cfg["weights"][dim] * worst
        flags = sorted({t["dim"] for t in traces if t["resolution"] < 1})
        # A specific earlier anchor or an explicit relation inside target is required.
        support = any(t["evidence"] for t in traces if t["dim"] in flags)
        # Allow refinement of an explicitly anchored update, even when old/new values remain incomplete.
        support = support or any(
            t["dim"] == "U"
            and names(archive[t["target"]]["text"])
            and len(archive[t["target"]]["text"].split()) >= 5
            for t in traces
        )
        if disabled and mode == "sparse_no_link":
            support = support or any(
                names(archive[s]["text"]) and UPDATE.search(archive[s]["text"])
                for s in primary
            )
        level = "L1" if not raw and (links or flags or times) else "L0"
        records.append(
            dict(
                id=f"m{ordinal:06}",
                primary=primary,
                links=sorted(links, key=lambda s: archive[s]["order"]),
                level=level,
                initial=ds,
                residual_dependency=observed,
                session_id=archive[primary[0]]["session"],
                resolution=rs,
                D=sum(cfg["weights"][j] * ds[j] for j in dims),
                R=residual,
                flags=flags,
                trace=traces,
                critical=any(
                    t["initial"] == 1 and t["resolution"] == 0 for t in traces
                ),
                support=support or (not flags and bool(traces)),
                temporal=[] if disabled else times,
                entities=sorted(
                    set().union(*(names(archive[s]["text"]) for s in primary))
                ),
                facts=[],
                status="direct" if level == "L0" else "linked",
            )
        )
        for s in primary:
            if RESIDENCE.search(archive[s]["text"]):
                residences[archive[s]["role"]].append(s)
            for ent in names(archive[s]["text"]):
                entity_index[ent].append(s)
        previous.extend(primary)
    return records
