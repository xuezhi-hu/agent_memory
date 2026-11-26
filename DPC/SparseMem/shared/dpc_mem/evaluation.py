"""TSM-aligned rubric selection adapted from dashmem.evaluation.

Only this evaluation module opens evaluation_only/gold.json.
"""
from collections import Counter,defaultdict
from pathlib import Path
import re
from .io import digest,read,save,key,file_hash,json_object
from .data import load_questions


RUBRICS=Path(__file__).parent/"rubrics"


def rubric_name(dataset,row):
    if dataset=="locomo":return "locomo"
    if row.get("abstention"):return "abstention"
    return {"temporal-reasoning":"temporal","knowledge-update":"update",
            "single-session-preference":"preference"}.get(row["category"],"standard")


def parse_label(dataset,value):
    if dataset=="locomo":
        try:label=json_object(value).get("label")
        except (ValueError,AttributeError,TypeError):
            # MaaS judge responses may prepend an explanation before the
            # final JSON envelope. Recover only an unambiguous explicit label.
            matches=re.findall(r'"label"\s*:\s*"(CORRECT|WRONG)"',str(value),flags=re.IGNORECASE)
            labels={m.upper() for m in matches}
            if len(labels)!=1:return None
            label=next(iter(labels))
        return {"CORRECT":True,"WRONG":False}.get(label.upper()) if isinstance(label,str) else None
    return {"yes":True,"no":False}.get(value.strip().lower())


def aggregate(rows,expected):
    if len({r["question_id"] for r in rows})!=len(rows):raise ValueError("Duplicate scores")
    if set(r["question_id"] for r in rows)!=set(expected):raise ValueError("Scores must include unresolved questions")
    def group(rr):
        n=len(rr);correct=sum(r.get("correct") is True for r in rr);unresolved=sum(r.get("correct") is None for r in rr)
        return {"n":n,"correct":correct,"unresolved":unresolved,
                "accuracy_percent":100*correct/n if n and not unresolved else None,
                "accuracy_lower_bound_percent":100*correct/n if n else None}
    cats=defaultdict(list)
    for r in rows:cats[r["category"]].append(r)
    return {"metric":"TSM_aligned_LLM_accuracy","overall":group(rows),"by_category":{k:group(v) for k,v in cats.items()}}


def paired(left,right):
    a={r["question_id"]:r["correct"] for r in left};b={r["question_id"]:r["correct"] for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):raise ValueError("Paired IDs mismatch")
    out=Counter()
    for q,x in a.items():
        y=b[q]
        out["unresolved" if x is None or y is None else {(False,True):"wrong_to_right",(True,False):"right_to_wrong",(True,True):"both_right",(False,False):"both_wrong"}[(x,y)]]+=1
    return dict(out)


def evaluate(root,manifest,group,mode,judge,recorder):
    root=Path(root);data=Path(manifest["data_path"])
    questions=load_questions(data,manifest["history_ids"])
    qids={q["question_id"] for q in questions}
    if file_hash(data/"evaluation_only/gold.json")!=read(data/"manifest.json")["gold_hash"]:raise ValueError("Gold data changed after preparation")
    gold_rows=read(data/"evaluation_only/gold.json")
    gold={g["question_id"]:g for g in gold_rows}
    if len(gold)!=len(gold_rows) or not qids<=set(gold):raise ValueError("Missing or duplicate gold")
    records=[];failure=None
    for q in questions:
        qid=q["question_id"];g=gold[qid]
        target=root/"scores"/group/mode/(key(qid)+".json")
        pp=root/"answers"/group/mode/(key(qid)+".json")
        row={"question_id":qid,"category":g["category"],"correct":None,"status":"prediction_missing"}
        if pp.exists():
            prediction=read(pp)
            if prediction["status"]=="ok":
                name=rubric_name(manifest["dataset"],g);template=(RUBRICS/(name+".txt")).read_text()
                signature=digest([q,g,prediction["answer"],template,manifest["config"]["judge"]])
                if target.exists():
                    row=read(target)
                    if row["signature"]!=signature:raise ValueError("Frozen evaluation inputs changed")
                elif failure is None:
                    prompt=template.format(question=q["question"],answer=g["answer"],response=prediction["answer"],
                                           gold_answer=g["answer"],generated_answer=prediction["answer"])
                    try:
                        with recorder.span("judge",group=group,mode=mode,question_id=qid):
                            output=judge.complete(None,prompt,"judge",manifest["config"]["judge_max_tokens"],group=group,mode=mode,question_id=qid)
                        correct=parse_label(manifest["dataset"],output)
                        row.update(correct=correct,status="judged" if correct is not None else "invalid_judge_output",
                                   raw_judge=output,rubric=name,rubric_hash=digest(template),signature=signature)
                        save(target,row)
                    except Exception as exc:
                        failure=exc;row.update(status="judge_failed",error_type=type(exc).__name__)
                else:row["status"]="not_attempted_after_failure"
            else:row["status"]="prediction_failed"
        records.append(row)
    save(root/"reports"/(group+"_"+mode+"_scores.json"),records)
    result=aggregate(records,qids)
    save(root/"reports"/(group+"_"+mode+"_accuracy.json"),result)
    if failure:raise failure
    return result


def audit_template(root,history_ids):
    root=Path(root);rows=[]
    for hid in history_ids:
        base=root/"shared"/key(hid)
        if not (base/"windows.json").exists():continue
        for w in read(base/"windows.json"):
            wid=w["window_id"]
            rows.append({"history_id":hid,"window_id":wid,"annotation_status":"unreviewed",
                "distinctions":[],"instructions":"Annotate all six dimensions independently of Guard labels; include ACCEPT windows.",
                "distinction_schema":{"id":"human-assigned","source_refs":[],"description":"",
                    "dimension":"SUBJECT_EVENT|TIME|MODALITY|NEGATION|CONDITION|SCOPE",
                    "base_preserved":None,"A_preserved":None,"B_preserved":None,"C_preserved":None,"D_preserved":None},
                "revision_reviews":[],
                "revision_schema":{"group":"C|D","card_id":"...","was_revised":True,"harmful":None,"reason":"","source_refs":[]},"reviewer":None,"adjudicator":None})
    target=root/"audit/human_template.json"
    if not target.exists():save(target,rows)
    return target
