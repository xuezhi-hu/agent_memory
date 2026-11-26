"""Frozen development experiment: one-hop source-backed anchors and set packing.

This module uses no benchmark annotations, gold answers or generative calls.
It consumes one question's frozen user-only archive and index at a time.
"""
from collections import Counter, defaultdict
import math
import re
import time
import numpy as np
from sparsemem.recall import _content_terms, _sentence_parts
from dpc_mem.retrieval import terms

# The alias table only covers explicit spelling variants, not identity inference.
ALIASES = {'bikes':'bicycle','bike':'bicycle','bicycles':'bicycle',
           'plants':'plant','cars':'car','flights':'flight','trips':'trip'}
GENERIC = set('user assistant someone something things thing item items stuff help advice tips want wanted need needed like make made take took get getting got using used use know think thinking tell told say said went going go give gave find found asked ask question answer right sure led time year month week day yesterday today tomorrow last next first second third recently long much many total number cost costs amount spent spend bought buy purchase purchased purchase purchases different several couple also just one two three four five six seven eight nine ten new old good great best much really another every each other some any back forth very well could would should can may might what when where which'.split())
PRONOUN = re.compile(r'^(?:it|its|they|them|their|this|that|these|those|he|she|his|her)\b', re.I)
VALUE = re.compile(r'\$\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b', re.I)


def anchors(text):
    out=set()
    for t in _content_terms(text):
        t=ALIASES.get(t,t)
        if len(t)>=3 and not t.isdigit() and t not in GENERIC:
            out.add(t)
    return out


def build_postings(records, archive):
    """Build a per-history source-backed inverted index; no LLM and no answers."""
    start=time.perf_counter()
    posts=defaultdict(set); by={}
    for rec in records:
        mid=rec['id']; a=set()
        for sid in rec.get('primary',[]):
            src=archive[sid]
            if src.get('role')!='user': raise ValueError('Mixed-role archive')
            a.update(anchors(src['text']))
        # L2 texts are accepted only because the existing frozen construction
        # validated their source refs; index terms do not establish identity.
        for fact in rec.get('facts',[]):
            if fact.get('source_ids') and all(x in archive for x in fact['source_ids']):
                a.update(anchors(fact.get('text','')))
        by[mid]=a
        for term in a: posts[term].add(mid)
    return dict(postings={k:sorted(v) for k,v in sorted(posts.items())},
                record_anchors={k:sorted(v) for k,v in by.items()},
                seconds=time.perf_counter()-start, record_count=len(records),
                anchor_count=len(posts), posting_count=sum(map(len,posts.values())),
                added_generation_tokens=0)


def _record_scores(index, question, qvec):
    """Score all records using same global BM25 and embedding space as V0."""
    q=np.asarray(qvec,dtype=np.float32); q=q/np.linalg.norm(q)
    dense=index.vectors @ q
    qterms=Counter(terms(question)); total=len(index.entries); lex=[]
    for i,tf in enumerate(index.tf):
        score=0.0
        for term in qterms:
            freq=tf.get(term,0)
            if freq:
                idf=math.log(1+(total-index.df[term]+.5)/(index.df[term]+.5))
                score += idf*freq*2.5/(freq+1.5*(.25+.75*index.lengths[i]/max(index.avg,1)))
        lex.append(score)
    result={}
    for i,e in enumerate(index.entries):
        mid=e['memory']; row=result.setdefault(mid,[-1.,0.]); row[0]=max(row[0],float(dense[i])); row[1]=max(row[1],lex[i])
    return result


def retrieve_candidates(question, records, archive, index, qvec, base, posting):
    started=time.perf_counter(); byid={r['id']:r for r in records}
    initial=base['initial'][:40]; baseline=base['ranked'][:20]
    scores=_record_scores(index,question,qvec)
    qkeys=anchors(question); posts=posting['postings']; N=max(1,len(records))
    # A shared lexical cue is required; arbitrary seed words are not taken to
    # imply the query's subject. Terms without at least two records cannot expand.
    seedterms=set().union(*(set(posting['record_anchors'].get(mid,[])) for mid in initial[:8])) if initial else set()
    active=[t for t in qkeys & seedterms if 2 <= len(posts.get(t,[])) <= max(12,int(N*.3))]
    active.sort(key=lambda t:(len(posts[t]),t)); active=active[:4]
    seed_sessions={byid[mid].get('session_id') for mid in initial[:8]}
    new=defaultdict(set)
    for term in active:
        eligible=[m for m in posts[term] if m not in baseline and byid[m].get('session_id') not in seed_sessions]
        eligible.sort(key=lambda m:(-scores.get(m,[-1,0])[0],-scores.get(m,[-1,0])[1],m))
        for m in eligible[:12]: new[m].add(term)
    additions=sorted(new,key=lambda m:(-scores.get(m,[-1,0])[0],-scores.get(m,[-1,0])[1],m))[:20]
    if not additions:
        return dict(fallback=True,initial=initial,baseline=baseline,anchors=active,extras=[],ranked=baseline,
                    duration=time.perf_counter()-started, reason='no_new_cross_session_candidates')
    pool=list(dict.fromkeys(initial+additions)); lexmax=max((scores[m][1] for m in pool),default=0) or 1
    densevals=[scores[m][0] for m in pool]; dlo,dhi=min(densevals),max(densevals)
    def rel(mid):
        d,l=scores[mid]; dn=(d-dlo)/(dhi-dlo) if dhi>dlo else 0
        return .60*dn + .40*l/lexmax
    # Rank on a common scale. The anchor bonus is weak and cannot establish truth.
    def rankscore(mid):
        bonus=max((math.log((N+1)/(len(posts[t])+1))/math.log(N+1) for t in new.get(mid,[])),default=0)
        return .85*rel(mid)+.15*bonus
    protected=[x for x in baseline[:8] if x in pool]
    # Reserve at most three slots for query-relevant linked candidates. Without
    # a small reserve the initial top-40 systematically crowds out the one-hop
    # search, even though all candidates are scored on the same scale.
    eligible=sorted((m for m in additions if rel(m)>=.15),key=lambda m:(-rankscore(m),m))[:3]
    remaining=sorted((m for m in pool if m not in protected and m not in eligible),key=lambda m:(-rankscore(m),m))
    chosen=(protected+remaining[:max(0,20-len(protected)-len(eligible))]+eligible)[:20]
    selected_additions=[m for m in chosen if m in additions]
    if not selected_additions:
        return dict(fallback=True,initial=initial,baseline=baseline,anchors=active,
                    extras=[{'memory':m,'anchors':sorted(new[m]),'query_relevance':rel(m),'score':rankscore(m)} for m in additions],ranked=baseline,
                    duration=time.perf_counter()-started,reason='no_new_candidate_selected')
    return dict(fallback=False, initial=initial,baseline=baseline,anchors=active,
        extras=[{'memory':m,'anchors':sorted(new[m]),'session':byid[m].get('session_id'),
                 'query_relevance':rel(m),'score':rankscore(m)} for m in additions],
        pool=pool,ranked=chosen,protected=protected,
        query_relevance={m:rel(m) for m in chosen},
        duration=time.perf_counter()-started, reason='one_hop_cross_session')


def make_units(question, records, archive, candidates, tokenizer):
    byid={r['id']:r for r in records}; qterms=anchors(question); units=[]; seen=set()
    for rank,mid in enumerate(candidates['ranked']):
        rec=byid[mid]; sources=rec.get('primary',[])+rec.get('links',[])
        for sid in sources:
            source=archive[sid]; text=source['text']; parts=_sentence_parts(text)
            # Short turns stay intact so their conditions and values aren't lost.
            spans=[text] if tokenizer.count(text)<=120 else parts
            local=[]
            for j,part in enumerate(spans):
                if j and PRONOUN.match(part): part=spans[j-1]+' '+part
                skey=(sid,part)
                if skey in seen: continue
                overlap=qterms & anchors(part)
                score=.65*candidates['query_relevance'][mid]+.25*(len(overlap)/max(1,len(qterms)))
                if overlap and VALUE.search(part): score+=.10
                local.append(dict(id=f'{sid}#{j}',source_id=sid,memory=mid,session=source['session'],
                    date=source.get('date'),order=source.get('order',0),text=part,rank=rank,
                    relevance=score,termset=sorted(anchors(part)),values=sorted(set(VALUE.findall(part)))))
            local.sort(key=lambda x:(-x['relevance'],x['id']))
            for u in local[:3]:
                key=(u['source_id'],u['text'])
                if key not in seen: units.append(u); seen.add(key)
    return units


def pack(question, units, archive, tokenizer, budget, mode, max_memories=20):
    """A independent relevance; B marginal relevance/novelty/redundancy."""
    chosen=[]; remaining=list(units); qterms=anchors(question); protected=set()
    # Both modes protect the same best evidence per top four memory records.
    for rank in range(4):
        options=[u for u in remaining if u['rank']==rank]
        if options: protected.add(max(options,key=lambda u:(u['relevance'],u['id']))['id'])
    max_protected=1024
    def line(u): return f"[Evidence {u['source_id']} | user | {u['date'] or 'unknown date'}] {u['text']}"
    def serialized(rows): return '\n'.join(line(x) for x in sorted(rows,key=lambda x:(x['order'],x['id'])))
    def feasible(u):
        return len({x['memory'] for x in chosen}|{u['memory']})<=max_memories and tokenizer.count(serialized(chosen+[u]))<=budget
    for u in sorted((u for u in remaining if u['id'] in protected),key=lambda u:(u['rank'],-u['relevance'],u['id'])):
        if tokenizer.count(serialized(chosen+[u]))<=min(max_protected,budget) and feasible(u):
            chosen.append(u);remaining.remove(u)
    decisions=[]
    while remaining:
        covered=set().union(*(set(x['termset']) for x in chosen)) if chosen else set()
        def priority(u):
            base=u['relevance']
            if mode=='A': return base
            ut=set(u['termset']); novel=len((ut-covered)-qterms)/max(1,len(ut-qterms))
            dup=0.0
            for x in chosen:
                xt=set(x['termset']); sim=len(ut&xt)/max(1,len(ut|xt))
                # Differing numbers/dates are an explicit warning against
                # treating semantically similar independent events as duplicates.
                if u['values'] and x['values'] and u['values']!=x['values']: sim*=.2
                dup=max(dup,sim)
            return .60*base+.25*novel-.15*dup
        remaining.sort(key=lambda u:(-priority(u),u['rank'],u['id']))
        u=remaining.pop(0); p=priority(u)
        if feasible(u): chosen.append(u); reason='selected'
        else: reason='budget_or_memory_limit'
        decisions.append(dict(id=u['id'],memory=u['memory'],priority=p,decision=reason))
    evidence=serialized(chosen)
    return dict(evidence=evidence,tokens=tokenizer.count(evidence),packed=list(dict.fromkeys(u['source_id'] for u in chosen)),
                selected_units=chosen,decisions=decisions,mode=mode,
                ranked=list(dict.fromkeys(u['memory'] for u in chosen)),reranker_pairs=0,aggregate=False)
