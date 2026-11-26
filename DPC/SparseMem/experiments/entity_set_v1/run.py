#!/usr/bin/env python3
"""Resumable 40-question A/B retrieval experiment over frozen user-only V0."""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from strategy import build_postings, retrieve_candidates, make_units, pack
from sparsemem import BASE
from sparsemem.construct import messages
from sparsemem.run import CountingEncoder, answer_question, chat_call
from dpc_mem.client import Embedder
from dpc_mem.io import RunLock, file_hash, read, save
from dpc_mem.metrics import Recorder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import HFTokenizer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/longmemeval_strict_user_top20_40_20261122'
DATA=ROOT/'data/longmemeval_s_nonempty_20261121_dedup_ids.json'
OUT=ROOT/'runs/longmemeval_entity_set_40_20261125'
GROUPS=('A_entity','B_entity_set')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=40);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if not 1<=args.limit<=40: ap.error('--limit must be in 1..40')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    shared=read(SOURCE/'run_manifest.json')['shared_manifest'];selected=shared['selected'];cfg=shared['config'];cfg=dict(cfg)
    original=read(BASE/'configs/compact_v3.json');original['reader']=shared['reader'];original['embedding']['model_path']=str(BASE/'models/all-MiniLM-L6-v2')
    paths=[HERE/'strategy.py',HERE/'run.py',ROOT/'sparsemem/recall.py',ROOT/'sparsemem/construct.py',ROOT/'sparsemem/run.py',ROOT/'sparsemem/prompts/answer_system.txt',ROOT/'sparsemem/prompts/answer_user.txt']
    manifest=dict(source_manifest_sha256=file_hash(SOURCE/'run_manifest.json'),dataset_sha256=file_hash(DATA),
        code_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in paths},selected=selected,config=cfg,
        reader=original['reader'],embedding=original['embedding'],groups=list(GROUPS),
        policy='all-user frozen archives; same A/B candidates and unit extraction, independent vs marginal ranking',
        limits=dict(channel=40,seed=8,anchors=4,postings_per_anchor=12,extra=20,protected_records=8,
                    final_records=20,evidence_tokens=4096,protected_unit_groups=4,protected_unit_tokens=1024),
        baseline='strict_user_top20 historical result Qwen LightMem rubric; zero new generative construction tokens',
        fact_alias_policy='local explicit spelling aliases, index links are candidates only, never proven identity')
    with RunLock(out):
        mp=out/'run_manifest.json'
        if mp.exists() and read(mp)!=manifest:raise RuntimeError('Manifest changed: create a new output directory')
        save(mp,manifest);data=read(DATA)
        tok=HFTokenizer(str(BASE/'models/Qwen3-30B-A3B-Instruct-2507-tokenizer'))
        backend=Embedder(original['embedding'],None,Recorder(out),out);encoder=CountingEncoder(backend)
        encoder.limit=read(BASE/'models/all-MiniLM-L6-v2/sentence_bert_config.json')['max_seq_length']
        for position,item in enumerate(selected[:args.limit],1):
            i,qid=item['index'],item['question_id']; sample=data[i];name=f'scope_{i:05d}'
            if sample['question_id']!=qid:raise ValueError('Question mismatch')
            memroot=SOURCE/'memories'/name; archive=read(memroot/'sources.json');records=read(memroot/'memories.json')
            if any(x.get('role')!='user' for x in archive.values()):raise ValueError('Non-user archive')
            index=Index.load(memroot/'index.json',backend.identity)
            if any(archive[s]['role']!='user' for entry in index.entries for s in entry['source_refs']):raise ValueError('Non-user index')
            memory=(archive,records,index,read(memroot/'construction_stats.json'))
            base_root=SOURCE/'strict_user_top20'/name; base=read(base_root/'retrieval.json')
            cp=out/'candidate_cache'/f'{name}.json'
            if cp.exists():
                cache=read(cp)
                if cache['index_sha256']!=file_hash(memroot/'index.json') or cache['source_sha256']!=file_hash(memroot/'sources.json'):
                    raise ValueError('Frozen memory cache changed')
            else:
                begin=time.perf_counter();posting=build_postings(records,archive)
                before_tokens,before_seconds=encoder.tokens,encoder.seconds
                qvec=encoder.embed([sample['question']],role='query')[0]
                cand=retrieve_candidates(sample['question'],records,archive,index,qvec,base,posting)
                units=[] if cand['fallback'] else make_units(sample['question'],records,archive,cand,tok)
                cache=dict(index_sha256=file_hash(memroot/'index.json'),source_sha256=file_hash(memroot/'sources.json'),
                    question_sha256=__import__('hashlib').sha256(sample['question'].encode()).hexdigest(),
                    candidates=cand,units=units,posting_stats={k:v for k,v in posting.items() if k not in ('postings','record_anchors')},
                    query_embedding_tokens=encoder.tokens-before_tokens,query_embedding_seconds=encoder.seconds-before_seconds,
                    preparation_seconds=time.perf_counter()-begin)
                save(cp,cache)
            cand=cache['candidates']
            for group in GROUPS:
                root=out/group/name;root.mkdir(parents=True,exist_ok=True);target=root/'result.json'
                if target.exists() and read(target).get('status')=='returned':continue
                start=time.perf_counter()
                trace=base.copy() if cand['fallback'] else pack(sample['question'],cache['units'],archive,tok,cfg['evidence_tokens'],'A' if group=='A_entity' else 'B')
                if not cand['fallback']:
                    # Extra candidates must supply an actual selected span; otherwise V0 is preserved.
                    extras={x['memory'] for x in cand['extras']}
                    if not extras.intersection(trace['ranked']): trace=base.copy();trace['fallback_reason']='no_expanded_evidence_selected'
                trace['candidate_info']=cand;trace['candidate_pool_cached']=True
                trace['added_index_construction_seconds']=cache['posting_stats']['seconds'];trace['added_generation_tokens']=0
                assert trace['tokens']<=4096 and len(set(trace['ranked']))<=20
                assert all(archive[s]['role']=='user' for s in trace['packed'])
                if '| assistant |' in trace['evidence']:raise ValueError('Assistant evidence')
                save(root/'retrieval.json',trace)
                if args.prepare_only:continue
                if trace['evidence']==base['evidence']:
                    shutil.copyfile(base_root/'result.json',target)
                    save(root/'reuse.json',{'source':str(base_root),'reason':'identical Reader evidence; same prompt, model and settings'})
                    oldjudge=SOURCE/'strict_user_top20/judge_qwen/items'/f'{qid}.json'
                    if oldjudge.exists():
                        dst=out/group/'judge_qwen/items'/f'{qid}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(oldjudge,dst)
                else:
                    msg=messages('answer',participant_dialogue=False,question=sample['question'],question_date=sample.get('question_date',''),evidence=trace['evidence'],aggregate_query=False)
                    elapsed=cache['preparation_seconds']+(time.perf_counter()-start)
                    result=answer_question(root,memory,(trace,msg,elapsed),cfg,tok,chat_call(root,original['reader'],original,tok),retry_limit=512)
                    save(target,result)
                    if result.get('status')!='returned':raise RuntimeError(f'Answer failed {group} {qid}; see ledger')
                print(f'{position}/{args.limit} {group} {qid} completed; evidence={trace["tokens"]}; anchors={len(cand["anchors"])}; fallback={cand["fallback"]}',flush=True)
            for group in GROUPS:
                rows=[]
                for x in selected:
                    p=out/group/f"scope_{x['index']:05d}"/'result.json'
                    if p.exists() and read(p).get('status')=='returned': rows.append(dict(question_id=x['question_id'],hypothesis=read(p)['answer']))
                (out/group/'hypotheses.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
            save(out/'progress.json',{'phase':'prepare' if args.prepare_only else 'answer','position':position,'requested':args.limit,
                                      'groups':{g:sum((out/g/f"scope_{x['index']:05d}"/'result.json').exists() for x in selected) for g in GROUPS}})
            print('progress '+json.dumps(read(out/'progress.json')),flush=True)
    print('finished',flush=True)

if __name__=='__main__':main()
