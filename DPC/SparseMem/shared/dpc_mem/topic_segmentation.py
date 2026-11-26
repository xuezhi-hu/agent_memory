"""LightMem-inspired attention boundaries for complete speaker turns.

Uses the configured independent embedding model for boundary adjustment;
LLMLingua-2 hidden vectors are a fallback only when no model is configured. Unlike LightMem's
original user/assistant pairing, each input item is one chronological turn.
"""
import os
from pathlib import Path

import numpy as np

from .io import digest, file_hash


def model_files_identity(path_value, label):
    path = Path(os.path.expandvars(path_value)).expanduser()
    if not path.is_dir():
        raise ValueError(label + ' model directory is missing')
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
             if p.is_file() and '.cache' not in p.relative_to(path).parts}
    if 'config.json' not in files or not ({'model.safetensors', 'pytorch_model.bin'} & set(files)):
        raise ValueError(label + ' model weights or config missing')
    return digest(files)


def model_identity(config):
    identities = {'topic': model_files_identity(config['model_path'], 'Topic')}
    if config.get('embedding_model_path'):
        identities['embedding'] = model_files_identity(config['embedding_model_path'], 'Embedding')
    return digest(identities)


class TopicSegmenter:
    def __init__(self, config):
        import torch
        from transformers import AutoModel, AutoTokenizer
        path = str(Path(os.path.expandvars(config['model_path'])).expanduser())
        self.identity = model_identity(config)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True,
                                               attn_implementation='eager',
                                               add_pooling_layer=False).eval()
        self.torch = torch
        self.max_tokens = min(config.get('buffer_tokens', 512), self.model.config.max_position_embeddings)
        self.layers = config.get('attention_layers', [8, 9, 10, 11])
        self.embedder = None
        if config.get('embedding_model_path'):
            from sentence_transformers import SentenceTransformer
            embedding_path = str(Path(os.path.expandvars(config['embedding_model_path'])).expanduser())
            self.embedder = SentenceTransformer(embedding_path, device='cpu', local_files_only=True)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def boundaries(self, encoded, texts):
        if len(encoded) < 3:
            return []
        ids = [self.tokenizer.cls_token_id]; spans = []
        for tokens in encoded:
            start = len(ids); ids.extend(tokens); spans.append((start, len(ids)))
        ids.append(self.tokenizer.sep_token_id)
        if len(ids) > self.max_tokens:
            raise ValueError('Topic input exceeds model capacity')
        x = self.torch.tensor([ids], dtype=self.torch.long)
        with self.torch.no_grad():
            out = self.model(x, attention_mask=self.torch.ones_like(x), output_attentions=True,
                             return_dict=True)
        attention = self.torch.stack([out.attentions[i] for i in self.layers]).mean(dim=(0, 2))[0].cpu().numpy()
        hidden = out.last_hidden_state[0].cpu().numpy()
        valid = np.ones(len(ids), dtype=bool)
        if len(ids) >= 6:
            valid[:3] = False; valid[-3:] = False
        matrix = np.zeros((len(encoded), len(encoded)))
        for i, (start, end) in enumerate(spans):
            ip = np.arange(start, end); ip = ip[valid[ip]]
            for j, (left, right) in enumerate(spans[:i]):
                jp = np.arange(left, right); jp = jp[valid[jp]]
                if ip.size and jp.size:
                    matrix[i, j] = attention[np.ix_(ip, jp)].sum(axis=1).mean()
            total = matrix[i].sum()
            if total > 0:
                matrix[i] /= total
        outer = [matrix[i, i-1] for i in range(1, len(encoded))]
        coarse = [k for k in range(1, len(outer)-1) if outer[k-1] < outer[k] > outer[k+1]]
        if not coarse:
            return []
        if self.embedder is not None:
            vectors = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            vectors = [hidden[a:b].mean(axis=0) if b > a else np.zeros(hidden.shape[-1]) for a,b in spans]
        similarities = []
        for left, right in zip(vectors, vectors[1:]):
            denom = np.linalg.norm(left) * np.linalg.norm(right)
            similarities.append(float(np.dot(left, right) / denom) if denom else 0.0)
        fine = []
        for threshold in np.arange(0.2, 0.501, 0.05):
            fine = [i+1 for i, value in enumerate(similarities) if value < threshold]
            if fine:
                break
        matched = [i for i in fine if any(abs(i-j) <= 3 for j in coarse)]
        return sorted(set(matched or fine))


def topic_windows(turns, config, tokenizer, segmenter):
    """Every turn appears once in targets; overlap is context only."""
    result = []; start = 0
    while start < len(turns):
        sid = turns[start]['session_id']; end = start; size = 0; encoded = []; segment_tokens = 2
        while end < len(turns) and turns[end]['session_id'] == sid and end-start < config['window_turns']:
            n = tokenizer.count(turns[end]['text'])
            ids = segmenter.encode(turns[end]['text'])
            if n > config['window_tokens'] or len(ids)+2 > segmenter.max_tokens:
                raise ValueError('A turn exceeds a topic or extraction limit; it was not truncated')
            if end > start and (size+n > config['window_tokens'] or segment_tokens+len(ids) > segmenter.max_tokens):
                break
            encoded.append(ids); size += n; segment_tokens += len(ids); end += 1
        cuts = segmenter.boundaries(encoded, [t['text'] for t in turns[start:end]])
        if any(type(c) is not int or c <= 0 or c >= end-start for c in cuts) or cuts != sorted(set(cuts)):
            raise ValueError('Topic segmenter returned invalid boundaries')
        for a, b in zip([0]+cuts, cuts+[end-start]):
            begin = start+a; stop = start+b
            context = [t for t in turns[max(0, begin-config['context_turns']):begin] if t['session_id'] == sid]
            result.append({'window_id':'w'+str(len(result)).zfill(5), 'target':turns[begin:stop], 'context':context})
        start = end
    if [t['id'] for w in result for t in w['target']] != [t['id'] for t in turns]:
        raise ValueError('Topic windows lost or duplicated turns')
    return result
