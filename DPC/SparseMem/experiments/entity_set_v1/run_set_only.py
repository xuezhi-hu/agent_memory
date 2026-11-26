#!/usr/bin/env python3
"""Diagnostic: B set selection over the original strict-user V0 Top-20."""
import argparse,json,time
from pathlib import Path
from strategy import _record_scores,make_units,pack
from sparsemem import BASE
from sparsemem.construct import messages
from sparsemem.run import CountingEncoder,answer_question,chat_call
from dpc_mem.client import Embedder
from dpc_mem.io import RunLock,file_hash,read,save
from dpc_mem.metrics import Recorder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import HFTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/longmemeval_strict_user_top20_40_20261122';RUN=ROOT/'runs/longmemeval_entity_set_40_20261125';DATA=ROOT/'data/longmemeval_s_nonempty_20261121_dedup_ids.json';OUT=RUN/'C_set_only'

def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=40);args=p.parse_args()
 shared=read(SOURCE/'run_manifest.json')['shared_manifest'];selected=shared['selected'];cfg=shared['config'];original=read(BASE/'configs/compact_v3.json');original['reader']=shared['reader'];original['embedding']['model_path']=str(BASE/'models/all-MiniLM-L6-v2');data=read(DATA)
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=dict(source_manifest_sha256=file_hash(SOURCE/'run_manifest.json'),strategy_sha256=file_hash(HERE/'strategy.py'),runner_sha256=file_hash(HERE/'run_set_only.py'),data_sha256=file_hash(DATA),selected=selected,
  description='V0 selected 20 memory records, B set packing; no association extension, identical LLM and budget',reader=shared['reader'],config=cfg)
 with RunLock(OUT):
  mp=OUT/'run_manifest.json'
  if mp.exists() and read(mp)!=manifest:raise ValueError('Manifest mismatch')
  save(mp,manifest);tok=HFTokenizer(str(BASE/'models/Qwen3-30B-A3B-Instruct-2507-tokenizer'));backend=Embedder(original['embedding'],None,Recorder(RUN),RUN);encoder=CountingEncoder(backend);encoder.limit=read(BASE/'models/all-MiniLM-L6-v2/sentence_bert_config.json')['max_seq_length']
  for pos,x in enumerate(selected[:args.limit],1):
   i=x['index'];qid=x['question_id'];name=f'scope_{i:05d}';target=OUT/name;target.mkdir(parents=True,exist_ok=True)
   if (target/'result.json').exists() and read(target/'result.json').get('status')=='returned':continue
   memory_root=SOURCE/'memories'/name;archive=read(memory_root/'sources.json');records=read(memory_root/'memories.json');idx=Index.load(memory_root/'index.json',backend.identity);memory=(archive,records,idx,{})
   if any(x['role']!='user' for x in archive.values()):raise ValueError('non-user')
   sample=data[i];baseline=read(SOURCE/'strict_user_top20'/name/'retrieval.json');start=time.perf_counter();qvec=encoder.embed([sample['question']],role='query')[0];scores=_record_scores(idx,sample['question'],qvec);pool=baseline['initial'][:40];ranked=baseline['ranked'][:20];pool=list(dict.fromkeys(pool+ranked))
   ds=[scores[m][0] for m in pool];dlo,dhi=min(ds),max(ds);lexmax=max((scores[m][1] for m in pool),default=0) or 1
   relevance={m:.6*((scores[m][0]-dlo)/(dhi-dlo) if dhi>dlo else 0)+.4*scores[m][1]/lexmax for m in ranked}
   candidates=dict(ranked=ranked,query_relevance=relevance);units=make_units(sample['question'],records,archive,candidates,tok);trace=pack(sample['question'],units,archive,tok,4096,'B');trace['candidate_info']={'initial':baseline['initial'],'pool':ranked,'extras':[],'anchors':[],'reason':'baseline_candidates_set_only'}
   assert trace['tokens']<=4096 and len(trace['ranked'])<=20 and all(archive[s]['role']=='user' for s in trace['packed'])
   save(target/'retrieval.json',trace)
   msg=messages('answer',participant_dialogue=False,question=sample['question'],question_date=sample.get('question_date',''),evidence=trace['evidence'],aggregate_query=False)
   result=answer_question(target,memory,(trace,msg,time.perf_counter()-start),cfg,tok,chat_call(target,original['reader'],original,tok),retry_limit=512);save(target/'result.json',result)
   if result.get('status')!='returned':raise RuntimeError(f'answer failed {qid}')
   rows=[]
   for y in selected:
    f=OUT/f"scope_{y['index']:05d}"/'result.json'
    if f.exists() and read(f).get('status')=='returned':rows.append(dict(question_id=y['question_id'],hypothesis=read(f)['answer']))
   (OUT/'hypotheses.jsonl').write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rows));save(OUT/'progress.json',{'answered':len(rows),'requested':args.limit})
   print(f'C_set_only {pos}/{args.limit} {qid} done evidence={trace["tokens"]}',flush=True)
if __name__=='__main__':main()
