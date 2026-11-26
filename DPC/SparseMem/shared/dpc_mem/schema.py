"""Source validation checks references, not semantic truth."""
from .io import digest


def obj(value):
    if not isinstance(value,dict):raise ValueError("Expected object")
    return value


def nonempty(value):
    if not isinstance(value,str) or not value.strip(): raise ValueError("Expected nonempty text")
    return value.strip()


def refs(value, allowed):
    if not isinstance(value,list) or not value or any(not isinstance(x,str) or x not in allowed for x in value):
        raise ValueError("Missing/unknown source or fact reference")
    return list(dict.fromkeys(value))


def validate_history(turns):
    required={"id","sequence","speaker","text","observed_at","session_id"}
    if not isinstance(turns,list) or not turns: raise ValueError("Empty history")
    if any(not isinstance(t,dict) or set(t)!=required for t in turns):raise ValueError("History fields invalid")
    if len({t["id"] for t in turns})!=len(turns): raise ValueError("Duplicate turn ID")
    for i,t in enumerate(turns):
        if set(t)!=required or t["sequence"]!=i: raise ValueError("History schema contains unsupported fields or sequence")
        for field in ["id","speaker","text","session_id"]: nonempty(t[field])
    return turns


def validate_facts(result, turns, target_ids, window_id, on_context_only=None):
    obj(result)
    if not isinstance(result.get("facts"),list): raise ValueError("facts array required")
    allowed={t["id"] for t in turns};out=[]
    for i,f in enumerate(result["facts"]):
        obj(f)
        rr=refs(f.get("source_refs"),allowed)
        if not set(rr)&set(target_ids):
            if on_context_only is None:
                raise ValueError("Context-only fact not allowed")
            on_context_only(rr)
            continue
        if f.get("modality") not in {"FACT","PLAN","POSSIBLE","UNKNOWN"}: raise ValueError("Invalid modality")
        conditions=f.get("conditions",[])
        if not isinstance(conditions,list) or any(not isinstance(c,str) for c in conditions): raise ValueError("Invalid conditions")
        for field in ["subject","event_time"]:
            if f.get(field) is not None and not isinstance(f[field],str): raise ValueError("Invalid optional fact field")
        out.append({"fact_id":window_id+":f"+str(i),"text":nonempty(f.get("text")),
                    "subject":f.get("subject"),"event_time":f.get("event_time"),
                    "modality":f["modality"],"conditions":conditions,"source_refs":rr})
    return out


def qualifiers(values, allowed):
    if not isinstance(values,list): raise ValueError("Qualifier list required")
    for v in values:obj(v)
    return [{"text":nonempty(v.get("text")),"source_refs":refs(v.get("source_refs"),allowed)} for v in values]


def validate_evidence(values, card, facts, turns, require_fact_source=True):
    """Check traceability structure; this does not establish semantic entailment."""
    if not isinstance(values,list) or not values:raise ValueError("Claim evidence is required")
    turn_map={t["id"]:t for t in turns};fact_map={f["fact_id"]:f for f in facts}
    text=[card["body"]]+[q["text"] for q in card["necessary_qualifiers"]]
    covered_facts=set();covered_sources=set();out=[]
    for item in values:
        obj(item);span=nonempty(item.get("span"))
        aligned=any(span in part for part in text)
        fr=refs(item.get("fact_refs"),fact_map)
        if not set(fr)<=set(card["fact_refs"]):raise ValueError("Evidence fact is not in card")
        sources=item.get("sources")
        if not isinstance(sources,list) or not sources:raise ValueError("Evidence sources are required")
        checked=[]
        for source in sources:
            obj(source);rid=source.get("source_ref");quote=nonempty(source.get("quote"))
            if rid not in turn_map or rid not in card["source_refs"]:raise ValueError("Unknown evidence source")
            if quote not in turn_map[rid]["text"]:raise ValueError("Evidence quote is not in its source turn")
            checked.append({"source_ref":rid,"quote":quote});covered_sources.add(rid)
        if require_fact_source:
            cited={source["source_ref"] for source in checked}
            if any(not cited.intersection(fact_map[fid]["source_refs"]) for fid in fr):
                raise ValueError("Claim evidence does not cite a source of each referenced fact")
        covered_facts.update(fr);out.append({"span":span,"span_exact_match":aligned,
                                            "fact_refs":fr,"sources":checked})
    if covered_facts!=set(card["fact_refs"]):raise ValueError("Claim evidence omits a fact")
    if covered_sources!=set(card["source_refs"]):raise ValueError("Claim evidence omits a source")
    return out


def validate_cards(result, facts, turns, window_id, prefix="b", require_evidence=False):
    ff={f["fact_id"]:f for f in facts};allowed={t["id"] for t in turns}
    rows=obj(result).get("cards")
    if not isinstance(rows,list) or not rows: raise ValueError("Nonempty cards required")
    out=[];covered=set()
    for i,c in enumerate(rows):
        obj(c)
        fr=refs(c.get("fact_refs"),ff);sr=refs(c.get("source_refs"),allowed)
        qq=qualifiers(c.get("necessary_qualifiers",[]),allowed)
        needed={x for f in fr for x in ff[f]["source_refs"]}|{x for q in qq for x in q["source_refs"]}
        if not needed<=set(sr): raise ValueError("Card source attribution omits fact/qualifier sources")
        card={"card_id":window_id+":"+prefix+str(i),"representation":"SUMMARY",
              "body":nonempty(c.get("body")),"necessary_qualifiers":qq,"fact_refs":fr,"source_refs":sr}
        if require_evidence:card["evidence"]=validate_evidence(c.get("evidence"),card,facts,turns)
        out.append(card)
        covered.update(fr)
    if covered!=set(ff): raise ValueError("Candidate cards omit input facts")
    return out


def fact_cards(facts, turns):
    result=[]
    for f in facts:
        details=[]
        if f.get("subject"):details.append("Subject: "+f["subject"])
        if f.get("event_time"):details.append("Event time: "+f["event_time"])
        details.append("Modality: "+f["modality"])
        body=f["text"]+"\n"+"; ".join(details)
        result.append({"card_id":f["fact_id"]+":fact","representation":"FACT_FALLBACK","body":body,
            "necessary_qualifiers":[{"text":c,"source_refs":f["source_refs"]} for c in f.get("conditions",[])],
            "fact_refs":[f["fact_id"]],"source_refs":f["source_refs"]})
    return result


def raw_cards(turns, source_ids=None, reason="fallback"):
    chosen=set(source_ids) if source_ids else {t["id"] for t in turns}
    return [{"card_id":"raw:"+digest([t["id"],t["text"]])[:20],"representation":"RAW_FALLBACK",
        "body":t["text"],"necessary_qualifiers":[],"fact_refs":[],"source_refs":[t["id"]],
        "fallback_reason":reason} for t in turns if t["id"] in chosen]


def render_card(card, turns):
    by_id={t["id"]:t for t in turns}
    source=[]
    for ref in card["source_refs"]:
        t=by_id[ref]
        source.append(f'{ref} | {t["speaker"]} | spoken: {t["observed_at"]}')
    parts=[card["body"]]
    if card["necessary_qualifiers"]:
        parts.append("Necessary conditions / distinctions:\n"+"\n".join(q["text"] for q in card["necessary_qualifiers"]))
    parts.append("Sources: "+"; ".join(source))
    return "\n".join(parts)
