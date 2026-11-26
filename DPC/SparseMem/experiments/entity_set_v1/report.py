#!/usr/bin/env python3
"""Audit paired answers and source coverage; gold fields used only here."""
import json, statistics
from pathlib import Path
from sparsemem.history import project
from dpc_mem.io import read, save
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/longmemeval_strict_user_top20_40_20261122'
OUT=ROOT/'runs/longmemeval_entity_set_40_20261125'
DATA=ROOT/'data/longmemeval_s_nonempty_20261121_dedup_ids.json'

def main():
 data=read(DATA);selected=read(OUT/'run_manifest.json')['selected'];groups={'V0':SOURCE/'strict_user_top20','A':OUT/'A_entity','B':OUT/'B_entity_set','C_set_only':OUT/'C_set_only'}
 allrows={};summary={}
 for group,path in groups.items():
  rows=[]
  for x in selected:
   i=x['index'];qid=x['question_id'];name=f'scope_{i:05d}';d=path/name
   if not (d/'result.json').exists():continue
   result=read(d/'result.json');trace=read(d/'retrieval.json');archive=read(SOURCE/'memories'/name/'sources.json');records=read(SOURCE/'memories'/name/'memories.json');by={r['id']:r for r in records};sample=data[i]
   _,mapping=project(sample);inverse={v:k for k,v in mapping.items()}
   gold={f'{inverse[sid]}:M{j}' for sid,msgs in zip(sample['haystack_session_ids'],sample['haystack_sessions']) if sid in inverse for j,msg in enumerate(msgs,1) if msg.get('has_answer') is True}
   goldsessions={inverse[s] for s in sample.get('answer_session_ids',[]) if s in inverse}
   info=trace.get('candidate_info',{});initial=info.get('initial',trace.get('initial',[]));pool=info.get('pool',initial)
   def primary_turns(ids):return {archive[s]['id'] for m in ids if m in by for s in by[m]['primary']}
   sources=set(trace.get('packed',[]))|{h['source_id'] for h in trace.get('key_sentence_in_context',[])}
   contextturns={archive[s]['id'] for s in sources if s in archive};contextsessions={archive[s]['session'] for s in sources if s in archive}
   jp=path/'judge_qwen/items'/f'{qid}.json';judge=read(jp) if jp.exists() else {}
   ledger=read(d/'ledger.json') if (d/'ledger.json').exists() else {}
   rows.append(dict(question_id=qid,index=i,correct=judge.get('correct') if judge.get('status')=='ok' else None,
    evidence_tokens=trace['tokens'],reader_input=result['stats'].get('query_known_input_tokens'),reader_output=result['stats'].get('query_known_output_tokens'),query_wall=result['stats'].get('query_wall_seconds'),recall_seconds=result['stats'].get('recall_seconds'),
    marked=len(gold),initial_marked=len(gold&primary_turns(initial)),pool_marked=len(gold&primary_turns(pool)),
    final_marked=len(gold&primary_turns(trace['ranked'])),context_marked=len(gold&contextturns),
    gold_sessions=len(goldsessions),context_gold_sessions=len(goldsessions&contextsessions),
    retrieved_memories=len(set(trace['ranked'])),user_only=all(archive[s]['role']=='user' for s in sources),
    used_baseline=(d/'reuse.json').exists(),fallback_reason=info.get('reason'),anchors=info.get('anchors',[]),
    expanded_records=len(info.get('extras',[])),selected_expanded=len(set(trace['ranked']) & {z['memory'] for z in info.get('extras',[])}),
    reader_recovery=sum(k.startswith('reader_recovery_') for k in ledger),
    selected_units=len(trace.get('selected_units',[]))))
  allrows[group]={r['question_id']:r for r in rows};scored=[r for r in rows if r['correct'] is not None]
  def mean(k):
   vs=[r[k] for r in rows if r.get(k) is not None];return round(statistics.mean(vs),2) if vs else None
  summary[group]=dict(answered=len(rows),scored=len(scored),correct=sum(r['correct'] for r in scored),accuracy=round(sum(r['correct'] for r in scored)/len(scored),4) if scored else None,
    marked_total=sum(r['marked'] for r in rows),marked_initial=sum(r['initial_marked'] for r in rows),marked_pool=sum(r['pool_marked'] for r in rows),
    marked_final=sum(r['final_marked'] for r in rows),marked_context=sum(r['context_marked'] for r in rows),
    all_marked_in_context=sum(r['marked']>0 and r['marked']==r['context_marked'] for r in rows),
    evidence_tokens_mean=mean('evidence_tokens'),reader_input_mean=mean('reader_input'),reader_output_mean=mean('reader_output'),
    query_wall_mean=mean('query_wall'),recall_seconds_mean=mean('recall_seconds'),
    baseline_reused=sum(r['used_baseline'] for r in rows),expanded_records_selected_questions=sum(r['selected_expanded']>0 for r in rows),
    reader_recoveries=sum(r['reader_recovery'] for r in rows),all_user=all(r['user_only'] for r in rows))
 pairs={}
 for a,b in [('V0','A'),('V0','B'),('A','B'),('V0','C_set_only'),('C_set_only','B')]:
  ids=[x['question_id'] for x in selected if x['question_id'] in allrows[a] and x['question_id'] in allrows[b] and allrows[a][x['question_id']]['correct'] is not None and allrows[b][x['question_id']]['correct'] is not None]
  pairs[f'{a}->{b}']=dict(n=len(ids),wrong_to_right=[z for z in ids if not allrows[a][z]['correct'] and allrows[b][z]['correct']],right_to_wrong=[z for z in ids if allrows[a][z]['correct'] and not allrows[b][z]['correct']],both_right=sum(allrows[a][z]['correct'] and allrows[b][z]['correct'] for z in ids),both_wrong=sum(not allrows[a][z]['correct'] and not allrows[b][z]['correct'] for z in ids))
 caches=[read(p) for p in (OUT/'candidate_cache').glob('scope_*.json')]
 index_stats=dict(count=len(caches),anchor_count_mean=round(statistics.mean(c['posting_stats']['anchor_count'] for c in caches),2) if caches else None,
   index_seconds_mean=round(statistics.mean(c['posting_stats']['seconds'] for c in caches),3) if caches else None,
   added_construct_llm_tokens=0,query_embedding_tokens=sum(c['query_embedding_tokens'] for c in caches),
   candidate_reasons={k:sum(c['candidates']['reason']==k for c in caches) for k in sorted({c['candidates']['reason'] for c in caches})})
 result=dict(development_sample=True,scope='40 previously explored multi-session questions; user-only histories; qwen judge; labels used only in this report',
  groups=summary,paired=pairs,index=index_stats,per_question=allrows,
  caveats=['fragment coverage does not prove the answer span was included','A changes both expansion and packing; do not attribute all gains to entity index','candidate cache and 40-Q samples used for development; not held-out benchmark','Qwen judge may disagree with GPT-4o-mini','frozen V0 construction has known invalid L2 output and unknown usage','wall timings mix historical baseline and shared cached retrieval, and are not controlled end-to-end benchmarks'])
 save(OUT/'paired_report.json',result)
 print(json.dumps({'groups':summary,'paired':pairs,'index':index_stats},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
