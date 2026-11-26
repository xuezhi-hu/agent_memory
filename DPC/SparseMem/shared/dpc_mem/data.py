"""Adapted from this project's dashmem.datasets: isolate history, QA and gold."""
import json
import re
from pathlib import Path
from .io import digest,file_hash,read,save
from .schema import validate_history


def prepare(source,dataset,out,ids=None):
    source,out=Path(source),Path(out)
    if out.exists():raise ValueError("Preparation requires a new directory")
    rows=read(source);histories={};questions=[];gold=[];selected=set(ids or [])
    matched=set()
    for sample in rows:
        if dataset=="locomo":
            sid=str(sample["sample_id"])
            hid="locomo_"+sid
            if selected and hid not in selected and sid not in selected:continue
            matched.update(selected&{hid,sid})
            conversation=sample["conversation"];turns=[]
            keys=sorted((k for k in conversation if re.fullmatch(r"session_\d+",k)),key=lambda k:int(k.split("_")[1]))
            for session in keys:
                for t in conversation[session]:
                    turns.append({"id":str(t["dia_id"]),"sequence":len(turns),"speaker":t["speaker"],
                        "text":t["text"],"observed_at":str(conversation.get(session+"_date_time","")),"session_id":session})
            cats={1:"multi-hop",2:"temporal",3:"open-domain",4:"single-hop"}
            for i,q in enumerate(sample["qa"]):
                if q["category"]==5:continue
                qid=hid+":q"+str(i)
                questions.append({"question_id":qid,"history_id":hid,"question":q["question"],
                    "question_date":"","cutoff":len(turns)-1,"history_hash":digest(turns)})
                gold.append({"question_id":qid,"category":cats[q["category"]],"answer":q["answer"],
                    "abstention":False,"gold_evidence":q.get("evidence",[])})
        elif dataset=="longmemeval":
            qid=str(sample["question_id"])
            if selected and qid not in selected:continue
            matched.add(qid)
            sessions=sample["haystack_sessions"];dates=sample["haystack_dates"];sids=sample["haystack_session_ids"]
            if not len(sessions)==len(dates)==len(sids):raise ValueError("Misaligned LongMemEval sessions")
            turns=[]
            for n,(sid,date,session) in enumerate(zip(sids,dates,sessions)):
                for i,t in enumerate(session):
                    turns.append({"id":f"{n}:{sid}:{i}","sequence":len(turns),"speaker":t["role"],
                        "text":t["content"],"observed_at":str(date),"session_id":f"{n}:{sid}"})
            hid="lme_"+digest(turns)[:24]
            questions.append({"question_id":qid,"history_id":hid,"question":sample["question"],
                "question_date":sample.get("question_date",""),"cutoff":len(turns)-1,"history_hash":digest(turns)})
            gold.append({"question_id":qid,"category":sample["question_type"],"answer":sample["answer"],
                "abstention":qid.endswith("_abs"),"gold_evidence":sample.get("answer_session_ids",[])})
        else:raise ValueError("Unsupported dataset")
        validate_history(turns)
        if hid in histories and histories[hid]!=turns:raise ValueError("History ID collision")
        histories[hid]=turns
    if selected-matched:raise ValueError("Requested sample IDs were not found: "+str(sorted(selected-matched)))
    if not histories or not questions:raise ValueError("Empty selection")
    if len({q["question_id"] for q in questions})!=len(questions):raise ValueError("Duplicate QA IDs")
    for hid,turns in histories.items():save(out/"histories"/(hid+".json"),turns)
    save(out/"questions.json",questions);save(out/"evaluation_only/gold.json",gold)
    manifest={"dataset":dataset,"source_hash":file_hash(source),"history_hashes":{h:digest(t) for h,t in histories.items()},
        "history_count":len(histories),"question_count":len(questions),"questions_hash":file_hash(out/"questions.json"),"gold_hash":file_hash(out/"evaluation_only/gold.json"),
        "scope":"non_adversarial_1_2_3_4" if dataset=="locomo" else "selected_original_histories"}
    save(out/"manifest.json",manifest)
    return manifest


def load_history(data,hid):
    manifest=read(Path(data)/"manifest.json")
    if hid not in manifest["history_hashes"]:raise ValueError("Unlisted history")
    turns=validate_history(read(Path(data)/"histories"/(hid+".json")))
    if digest(turns)!=manifest["history_hashes"][hid]:raise ValueError("History content changed")
    return turns


def load_questions(data,hids):
    root=Path(data);manifest=read(root/"manifest.json")
    if file_hash(root/"questions.json")!=manifest["questions_hash"]:raise ValueError("Questions changed")
    questions=[q for q in read(root/"questions.json") if q["history_id"] in hids]
    if len({q["question_id"] for q in questions})!=len(questions):raise ValueError("Duplicate question")
    allowed={"question_id","history_id","question","question_date","cutoff","history_hash"}
    for q in questions:
        if set(q)!=allowed:raise ValueError("Unsupported QA fields; gold/labels cannot enter answering")
        if q["history_hash"]!=manifest["history_hashes"][q["history_id"]]:raise ValueError("QA/history mismatch")
    return questions
