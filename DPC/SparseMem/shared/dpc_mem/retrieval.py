"""Hybrid retrieval inspired by SimpleMem's semantic/lexical views.

Stable-ID dedup adapts HybridRetriever._merge_and_deduplicate_entries (MIT).
Provenance and source licenses are in THIRD_PARTY_NOTICES.md and licenses/.
The local NumPy/BM25 backend, two-level RRF and indivisible packing are DPC adaptations.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .io import digest, read, save
from .schema import render_card, raw_cards


def terms(text):
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]",text.lower())


def deduplicate_entries(entries):
    seen=set();out=[]
    for entry in entries:
        if entry["id"] not in seen:
            seen.add(entry["id"]);out.append(entry)
    return out


def rrf(rankings,k=60):
    scores=defaultdict(float)
    for ranking in rankings:
        for rank,ident in enumerate(dict.fromkeys(ranking),1):scores[ident]+=1/(k+rank)
    return sorted(scores,key=lambda x:(-scores[x],x)),dict(scores)


def memory_entries(cards,turns):
    return [{"id":"mem:"+c["card_id"],"text":render_card(c,turns),"source_refs":c["source_refs"],
             "channel":"memory","representation":c["representation"],
             "qualifier_count":len(c["necessary_qualifiers"]),"body":c["body"],
             "necessary_qualifiers":c["necessary_qualifiers"]} for c in cards]


def raw_entries(turns,chunk_turns):
    result=[];start=0
    while start<len(turns):
        end=start+1
        while end<len(turns) and end-start<chunk_turns and turns[end]["session_id"]==turns[start]["session_id"]:end+=1
        block=turns[start:end]
        rows=raw_cards(block)
        result.append({"id":"raw:"+digest([t["id"] for t in block])[:24],
            "text":"\n".join(render_card(c,turns) for c in rows),"source_refs":[t["id"] for t in block],
            "channel":"raw","representation":"RAW","qualifier_count":0})
        start=end
    return result


class Index:
    def __init__(self,entries,vectors):
        self.entries=deduplicate_entries(entries)
        if len(self.entries)!=len(entries):raise ValueError("Duplicate index IDs")
        self.by_id={e["id"]:e for e in entries}
        self.vectors=np.asarray(vectors,dtype=np.float32)
        if entries:
            if self.vectors.ndim!=2 or len(self.vectors)!=len(entries) or not np.isfinite(self.vectors).all():raise ValueError("Invalid vector matrix")
            norm=np.linalg.norm(self.vectors,axis=1,keepdims=True)
            if np.any(norm==0):raise ValueError("Zero vector")
            self.vectors=self.vectors/norm
        self.tf=[Counter(terms(e["text"])) for e in entries]
        self.df=Counter(t for row in self.tf for t in row)
        self.lengths=[sum(x.values()) for x in self.tf]
        self.avg=sum(self.lengths)/len(entries) if entries else 1

    @classmethod
    def build(cls,path,entries,embedder,recorder,**context):
        path=Path(path);signature=digest([entries,embedder.identity])
        if path.exists():
            data=read(path)
            if data["signature"]!=signature:raise ValueError("Index signature mismatch")
            return cls.load(path,embedder.identity)
        with recorder.span("index_build",**context):
            vv=embedder.embed([e["text"] for e in entries],**context)
            result=cls(entries,vv)
            save(path,{"signature":signature,"entries":entries,"vectors":vv,"embedding_identity":embedder.identity,"vector_hash":digest(vv)})
        return result

    @classmethod
    def load(cls,path,embedding_identity):
        data=read(path)
        if data["embedding_identity"]!=embedding_identity:raise ValueError("Query/index embedding identity mismatch")
        if data["signature"]!=digest([data["entries"],embedding_identity]):raise ValueError("Index contents changed")
        if data.get("vector_hash")!=digest(data["vectors"]):raise ValueError("Index vector checksum mismatch")
        return cls(data["entries"],data["vectors"])

    def search(self,query,vector,config):
        if not self.entries:return [],{}
        q=np.asarray(vector,dtype=np.float32)
        if q.shape!=(self.vectors.shape[1],) or not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError("Invalid query vector")
        cos=self.vectors@(q/np.linalg.norm(q))
        dense=sorted(range(len(cos)),key=lambda i:(-float(cos[i]),self.entries[i]["id"]))[:config["dense_k"]]
        kw=Counter(terms(query));scores=[]
        k1,b=config["bm25_k1"],config["bm25_b"]
        for i,tf in enumerate(self.tf):
            score=0.0
            for term in kw:
                freq=tf.get(term,0)
                if not freq:continue
                idf=math.log(1+(len(self.tf)-self.df[term]+.5)/(self.df[term]+.5))
                score+=idf*freq*(k1+1)/(freq+k1*(1-b+b*self.lengths[i]/max(self.avg,1)))
            if score>0:scores.append((score,i))
        sparse=[i for _,i in sorted(scores,key=lambda p:(-p[0],self.entries[p[1]]["id"]))[:config["lexical_k"]]]
        order,fused=rrf([[self.entries[i]["id"] for i in dense],[self.entries[i]["id"] for i in sparse]],config["rrf_k"])
        return order,{"dense":[self.entries[i]["id"] for i in dense],"lexical":[self.entries[i]["id"] for i in sparse],"fused":fused}


def pack(order,entries,tokenizer,budget,top_k):
    selected=[];skipped=[];seen=set();rendered=""
    for ident in order:
        e=entries[ident]
        signature=digest([e["text"],sorted(e["source_refs"])])
        if signature in seen:
            skipped.append({"id":ident,"reason":"exact_duplicate"});continue
        if len(selected)>=top_k:
            skipped.append({"id":ident,"reason":"unit_limit"});continue
        unit=f'[Evidence {len(selected)+1}]\n{e["text"]}'
        next_text=rendered+("\n\n" if rendered else "")+unit
        if tokenizer.count(next_text)>budget:
            skipped.append({"id":ident,"reason":"token_budget","unit_tokens":tokenizer.count(unit)})
            continue
        selected.append(e);seen.add(signature);rendered=next_text
    return {"evidence":rendered,"selected":selected,"skipped":skipped,"evidence_tokens":tokenizer.count(rendered)}


def retrieve(query,vector,memory,raw,mode,config,tokenizer):
    if mode not in {"M","MR"}:raise ValueError("Unknown retrieval mode")
    mo,mt=memory.search(query,vector,config)
    maps=dict(memory.by_id);trace={"memory":mt}
    if mode=="MR":
        if raw is None:raise ValueError("Raw index required")
        ro,rt=raw.search(query,vector,config);maps.update(raw.by_id);trace["raw"]=rt
        order,scores=rrf([mo,ro],config["rrf_k"])
    else:order,scores=rrf([mo],config["rrf_k"])
    packed=pack(order,maps,tokenizer,config["evidence_tokens"],config["final_k"])
    return {**packed,"ranked_ids":order,"fusion_scores":scores,"channel_trace":trace}
