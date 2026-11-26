"""Compatible HTTP clients with durable request records and explicit network opt-in."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from .metrics import events
from .io import digest, read, save


class TransportFailure(RuntimeError):
    pass


class ProtocolViolation(RuntimeError):
    """A model, thinking or endpoint mismatch must stop, never fall back."""
    pass


class InvalidCompletion(ValueError):
    def __init__(self, message, finish_reason=None):
        super().__init__(message)
        self.finish_reason=finish_reason


def endpoint(value):
    parsed=urlparse(value)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.hostname or parsed.scheme not in {"http","https"}):
        raise ValueError("Use a plain endpoint without credentials or query parameters")
    if parsed.scheme=="http" and parsed.hostname not in {"localhost","127.0.0.1","::1"}:
        raise ValueError("Remote endpoint must use HTTPS; use localhost for SSH forwarding")
    return value.rstrip("/")


class HTTPTransport:
    """Bounded, frozen recovery. Each attempt has a durable ID and separate cost."""
    def __init__(self, root, recorder, execute=False, timeout=600, recovery=None):
        if not execute:
            raise ValueError("Real model calls require --execute")
        self.root, self.recorder, self.timeout=Path(root),recorder,timeout
        self.recovery=recovery or {"max_retries_per_request":2,"max_retries_per_run":12,"backoff_seconds":[2,5]}
        self._local=threading.local()

    @property
    def last_call(self):
        return getattr(self._local,"last_call",None)

    @last_call.setter
    def last_call(self,value):
        self._local.last_call=value

    def request(self, url, payload, key_env, stage, **context):
        request_key=digest({"url":url,"payload":payload})
        path=self.root/"calls"/(request_key+".json")
        old=read(path) if path.exists() else None
        if old and old["status"]=="returned":
            self.recorder.emit("cache_hit",stage=stage,request_key=request_key,call_id=old["call_id"],**context)
            self.last_call={**old,"cache_hit":True}
            return old["response"]
        token=os.environ.get(key_env,"") if key_env else ""
        if key_env and not token:
            raise ValueError("Missing environment variable "+key_env)
        attempts=[] if old is None else old.get("attempts",[old])
        # An interrupted pending dispatch may have reached the provider. It consumes
        # an attempt and remains unknown, rather than being counted as free.
        while True:
            if attempts:
                if attempts[-1].get("retryable") is False:
                    raise ProtocolViolation("Nonretryable HTTP failure; inspect saved status code")
                retry_events=[r for r in events(self.root) if r["event"]=="automatic_retry"]
                if (len(attempts)>self.recovery["max_retries_per_request"]
                        or len(retry_events)>=self.recovery["max_retries_per_run"]):
                    raise TransportFailure("Frozen retry limit reached; completed work retained; inspect calls and efficiency report")
                self.recorder.emit("automatic_retry",stage=stage,request_key=request_key,
                                   attempt=len(attempts)+1,previous_usage_unknown=attempts[-1].get("usage") is None,**context)
                delays=self.recovery["backoff_seconds"]
                time.sleep(delays[min(len(attempts)-1,len(delays)-1)])
            record={"call_id":uuid.uuid4().hex,"request_key":request_key,"stage":stage,
                    "model":payload["model"],"status":"pending","usage":None,"attempt":len(attempts)+1,**context}
            attempts.append(record)
            save(path,{**record,"request":payload,"attempts":attempts})
            self.recorder.emit("request",**record)
            start=time.perf_counter()
            try:
                # Lazu's OpenAI-compatible gateway blocks the default
                # urllib user agent with Cloudflare error 1010.  Use the
                # conventional OpenAI SDK user agent while keeping the
                # request body and authentication unchanged.
                headers={"Content-Type":"application/json", "User-Agent":"OpenAI/Python"}
                if token: headers["Authorization"]="Bearer "+token
                req=Request(url,json.dumps(payload,ensure_ascii=False).encode(),headers)
                with urlopen(req,timeout=self.timeout) as response:
                    data=json.load(response)
                if not isinstance(data,dict): raise ValueError("Expected object response")
                usage=data.get("usage")
                if usage is None and isinstance(data.get("prompt_eval_count"),int):
                    usage={"input_tokens":data["prompt_eval_count"],"output_tokens":0}
                record.update(status="returned",response=data,usage=usage)
            except Exception as exc:
                code=exc.code if isinstance(exc,HTTPError) else None
                retryable=(code in {408,425,429,500,502,503,504} if code is not None else
                           isinstance(exc,(TimeoutError,ConnectionError,URLError,OSError,json.JSONDecodeError)))
                record.update(status="failed",error_type=type(exc).__name__,http_status=code,retryable=retryable)
                if isinstance(exc,HTTPError):
                    try:
                        detail=exc.read(4096).decode("utf-8",errors="replace")
                        if token:detail=detail.replace(token,"[REDACTED]")
                        detail=re.sub(r"sk-[A-Za-z0-9_.-]+","[REDACTED]",detail)
                        record["provider_error_excerpt"]=detail
                    except Exception:pass
            finally:
                record["seconds"]=time.perf_counter()-start
                save(path,{**record,"request":payload,"attempts":attempts})
                self.recorder.emit("request",**{k:v for k,v in record.items() if k!="response"})
            if record["status"]=="returned":
                self.last_call={**record,"cache_hit":False}
                return record["response"]


class ChatClient:
    def __init__(self, config, transport, tokenizer, reader=True):
        self.config,self.transport,self.tokenizer,self.reader=config,transport,tokenizer,reader
        self.base=endpoint(os.environ.get(config["base_url_env"],""))

    def complete(self, system, data, stage, max_tokens, *, json_object_output=False, **context):
        messages=[] if system is None else [{"role":"system","content":system}]
        messages.append({"role":"user","content":data if isinstance(data,str) else json.dumps(data,ensure_ascii=False)})
        if self.reader and self.tokenizer.messages(messages)+max_tokens>self.config["context_tokens"]:
            raise ValueError("Reader context budget exceeded before dispatch; input was not truncated")
        payload={"model":self.config["model"],"messages":messages,"temperature":self.config["temperature"],"max_tokens":max_tokens}
        extra=self.config.get("request_extra",{})
        if set(extra)&set(payload): raise ValueError("request_extra cannot override frozen model/messages/temperature/limit")
        payload.update(extra)
        if json_object_output:
            payload["response_format"] = {"type": "json_object"}
        response=self.transport.request(self.base+"/chat/completions",payload,self.config.get("key_env"),stage,**context)
        aliases=self.config.get("response_models",[self.config["model"]])
        if response.get("model") not in aliases or response.get("is_fallback") is True:
            raise ProtocolViolation("Unexpected returned model or provider fallback; response saved")
        try:
            choice=response["choices"][0];message=choice["message"]
            reasoning=(response.get("usage") or {}).get("completion_tokens_details") or {}
            if message.get("reasoning_content") or reasoning.get("reasoning_tokens",0):
                raise ProtocolViolation("Thinking output detected; frozen protocol requires nonthinking responses")
            content=message.get("content")
            if choice.get("finish_reason")!="stop" or not isinstance(content,str) or not content.strip():
                raise InvalidCompletion("Incomplete or empty response; response and usage saved", choice.get('finish_reason'))
            self.last_completion={k:self.transport.last_call.get(k) for k in ["call_id","cache_hit","seconds","usage"]}
            return content
        except (KeyError,IndexError,TypeError):
            raise InvalidCompletion("Malformed response; saved for inspection") from None


class Embedder:
    def __init__(self, config, transport, recorder, root):
        self.config,self.transport,self.recorder=config,transport,recorder
        self.model = None
        if config['backend'] == 'sentence_transformers':
            from .topic_segmentation import model_files_identity
            self.base='local-model:'+model_files_identity(config['model_path'],'Embedding')
        else:
            self.base=endpoint(os.environ.get(config["base_url_env"],""))
        self.root=Path(root)/"embedding_cache"
        self.identity=digest({"config":config,"base":self.base})

    def embed(self, texts, role="document", **context):
        prefix=self.config.get("query_prefix","") if role=="query" else self.config.get("document_prefix","")
        rendered=[prefix+t for t in texts]
        self.last_cache_hits=0
        values=[None]*len(texts); missing={}
        for i,t in enumerate(rendered):
            k=digest([self.identity,t]);p=self.root/(k+".json")
            if p.exists():
                self.last_cache_hits+=1
                cached=read(p)
                if cached.get("identity")!=self.identity:raise ValueError("Embedding cache identity mismatch")
                values[i]=cached["vector"]
                self._validate(values[i])
                self.recorder.emit("embedding_cache_hit",role=role,**context)
            else:
                missing.setdefault(k,{"text":t,"indices":[]})["indices"].append(i)
        entries=list(missing.items())
        for start in range(0,len(entries),self.config["batch_size"]):
            batch=entries[start:start+self.config["batch_size"]]
            if self.config['backend']=='sentence_transformers':
                if self.model is None:
                    from sentence_transformers import SentenceTransformer
                    self.model=SentenceTransformer(os.path.expandvars(self.config['model_path']),device='cpu',local_files_only=True)
                with self.recorder.span('embedding_local',role=role,**context):
                    vectors=self.model.encode([v['text'] for _,v in batch],batch_size=self.config['batch_size'],convert_to_numpy=True,show_progress_bar=False).tolist()
                for (k,v),vector in zip(batch,vectors):
                    self._validate(vector)
                    save(self.root/(k+'.json'),{'vector':vector,'identity':self.identity})
                    for i in v['indices']:values[i]=vector
                continue
            payload={"model":self.config["model"],"input":[v["text"] for _,v in batch]}
            if self.config["backend"]=="ollama":
                payload["truncate"]=False;url=self.base+"/api/embed"
            else:
                payload["dimensions"]=self.config["dimensions"];url=self.base+"/embeddings"
            response=self.transport.request(url,payload,self.config.get("key_env"),"embedding",role=role,**context)
            if response.get("model")!=self.config["model"]: raise ProtocolViolation("Embedding model mismatch")
            if self.config["backend"]=="ollama": vectors=response.get("embeddings",[])
            else:
                data=sorted(response.get("data",[]),key=lambda x:x["index"])
                if [x["index"] for x in data]!=list(range(len(batch))): raise ValueError("Embedding response order invalid")
                vectors=[x["embedding"] for x in data]
            if len(vectors)!=len(batch): raise ValueError("Embedding response count mismatch")
            for (k,v),vector in zip(batch,vectors):
                self._validate(vector)
                save(self.root/(k+".json"),{"vector":vector,"identity":self.identity})
                for i in v["indices"]: values[i]=vector
        return values

    def _validate(self,vector):
        if (not isinstance(vector,list) or len(vector)!=self.config["dimensions"]
                or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in vector)
                or not any(vector)):
            raise ValueError("Embedding dimension/nonfinite/zero vector failure")


class FixtureEmbedder:
    """Deterministic hash vectors for protocol tests. Not a semantic model."""
    identity="fixture-hash-vector-not-a-model"

    def embed(self,texts,role="document",**context):
        rows=[]
        for t in texts:
            v=[0.0]*64
            for word in re.findall(r"[\u4e00-\u9fff]|\w+",t.lower()):
                v[int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%64]+=1
            if not any(v):v[0]=1
            rows.append(v)
        return rows
