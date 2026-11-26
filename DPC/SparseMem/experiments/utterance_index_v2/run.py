"""Replay frozen SparseMem memory with optional utterance-level keys."""
import argparse
import time
from pathlib import Path

from sparsemem import BASE
from sparsemem.run import CountingEncoder, answer_question, chat_call, pack_question, write_lines
from sparsemem.utterance_index import utterance_entries
from dpc_mem.client import Embedder
from dpc_mem.io import RunLock, digest, file_hash, read, save
from dpc_mem.metrics import Recorder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import HFTokenizer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/longmemeval_strict_user_top20_40_20261122'
DATA = ROOT / 'data/longmemeval_s_nonempty_20261121_dedup_ids.json'
EVAL = ROOT / 'runs/longmemeval_full_qwen_20261121_dedup/evaluation_only'
GROUPS = ('baseline', 'utterance')


class KeyTokenizer(HFTokenizer):
    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))


def build_index(out, raw_index, records, archive, encoder, key_tokenizer):
    started = time.perf_counter()
    extra = utterance_entries(records, archive, key_tokenizer)
    # Replace the capped sentence keys on the experimental side. Reuse the
    # identical raw, linked and L2 entries with their already computed vectors.
    base_pairs = [(entry, vector) for entry, vector in zip(raw_index['entries'], raw_index['vectors'])
                  if entry['kind'] != 'key_sentence']
    base_entries = [entry for entry, _ in base_pairs]
    all_entries = base_entries + extra
    signature = digest([all_entries, encoder.identity])
    path = out / 'index.json'
    stat_path = out / 'index_stats.json'
    if path.exists() and stat_path.exists():
        stats = read(stat_path)
        if (stats['signature'] != signature or stats['baseline_vector_hash'] != raw_index['vector_hash']
                or file_hash(path) != stats['index_sha256']):
            raise ValueError('Experimental index changed')
        return Index.load(path, encoder.identity), stats
    tokens, seconds, hits = encoder.tokens, encoder.seconds, encoder.hits
    vectors = [vector for _, vector in base_pairs] + (encoder.embed([e['text'] for e in extra]) if extra else [])
    index = Index(all_entries, vectors)
    save(path, dict(entries=all_entries, vectors=vectors, signature=signature,
                    embedding_identity=encoder.identity, vector_hash=digest(vectors)))
    stats = dict(signature=signature, index_sha256=file_hash(path),
                 baseline_vector_hash=raw_index['vector_hash'], index_entries=len(all_entries),
                 utterance_keys=len(extra), index_size_bytes=path.stat().st_size,
                 index_wall_seconds=time.perf_counter() - started, added_generation_tokens=0,
                 embedding_tokens=encoder.tokens - tokens, embedding_seconds=encoder.seconds - seconds,
                 embedding_cache_hits=encoder.hits - hits)
    save(stat_path, stats)
    return index, stats


def evidence_audit(sample, scope, archive, trace):
    # Evaluation annotations are only read after retrieval and answer materialization.
    inverse = {real: neutral for neutral, real in read(EVAL / f'{scope}.json')['mapping'].items()}
    gold = {f'{inverse[session]}:M{number}': message['content']
            for session, messages in zip(sample['haystack_session_ids'], sample['haystack_sessions'])
            if session in inverse for number, message in enumerate(messages, 1)
            if message.get('has_answer') is True}
    ids = set(trace['packed']) | {row['source_id'] for row in trace.get('key_sentence_in_context', [])}
    ids.update(row['source_refs'][0] for row in trace.get('utterance_in_context', []))
    turns = {archive[ref]['id'] for ref in ids}
    return dict(marked_turns=len(gold), marked_with_fragment=len(gold.keys() & turns),
                marked_full_text=sum(bool(gold[turn]) and gold[turn] in trace['evidence']
                                     for turn in gold.keys() & turns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--answer', action='store_true', help='Also call the frozen Qwen Reader')
    args = parser.parse_args()
    out = args.out.resolve()
    source_manifest = SOURCE / 'run_manifest.json'
    frozen = read(source_manifest)['shared_manifest']
    if not (0 <= args.offset and 1 <= args.limit and args.offset + args.limit <= len(frozen['selected'])):
        parser.error('Requested slice is outside the frozen 40-question development set')
    if out == SOURCE or SOURCE in out.parents or out in SOURCE.parents:
        parser.error('Choose a separate run directory')
    selected = frozen['selected'][args.offset:args.offset + args.limit]
    if file_hash(DATA) != frozen['dataset_sha256']:
        raise ValueError('Dataset differs from baseline')
    data = read(DATA)
    config = frozen['config']
    original = dict(read(BASE / 'configs/compact_v3.json'), reader=frozen['reader'])
    embedding = frozen['embedding']
    if embedding['backend'] != 'sentence_transformers' or Path(embedding['model_path']).name != 'all-MiniLM-L6-v2':
        raise ValueError('Expected the frozen local MiniLM model')
    if embedding.get('document_prefix'):
        raise ValueError('This profile requires unprefixed utterance keys')
    manifest = dict(source_manifest_sha256=file_hash(source_manifest), dataset_sha256=frozen['dataset_sha256'],
                    selected=selected, config=config, modes=list(GROUPS), max_key_tokens=96,
                    code={p.name: file_hash(p) for p in (Path(__file__), ROOT / 'sparsemem/utterance_index.py',
                                                       ROOT / 'sparsemem/recall.py')})
    with RunLock(out):
        manifest_path = out / 'run_manifest.json'
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise ValueError('Configuration changed. Use a separate run directory.')
        save(manifest_path, manifest)
        tokenizer = HFTokenizer(BASE / 'models/Qwen3-30B-A3B-Instruct-2507-tokenizer')
        key_tokenizer = KeyTokenizer(embedding['model_path'])
        backend = Embedder(embedding, None, Recorder(out), out)
        encoder = CountingEncoder(backend)
        encoder.limit = read(Path(embedding['model_path']) / 'sentence_bert_config.json')['max_seq_length']
        diagnostics = []
        hypotheses = {group: [] for group in GROUPS}
        for chosen in selected:
            index_number = chosen['index']
            sample = data[index_number]
            scope = f'scope_{index_number:05d}'
            if sample['question_id'] != chosen['question_id']:
                raise ValueError(f'Question mismatch: {scope}')
            memory_dir = SOURCE / 'memories' / scope
            archive = read(memory_dir / 'sources.json')
            records = read(memory_dir / 'memories.json')
            if any(source['role'] != 'user' for source in archive.values()):
                raise ValueError(f'Unexpected input role in {scope}')
            baseline_path = memory_dir / 'index.json'
            baseline = Index.load(baseline_path, encoder.identity)
            variant, stats = build_index(out / 'indexes' / scope, read(baseline_path), records, archive,
                                         encoder, key_tokenizer)
            # Both recall timings read an already cached query vector.
            backend.embed([sample['question']], role='query')
            for group, index in zip(GROUPS, (baseline, variant)):
                dest = out / group / scope
                dest.mkdir(parents=True, exist_ok=True)
                cfg = dict(config, utterance_index=group == 'utterance')
                memory = (archive, records, index, {})
                packed = pack_question(dest, memory, sample['question'], sample.get('question_date', ''),
                                       cfg, tokenizer, encoder, None)
                trace, messages, elapsed = packed
                assert len(trace['ranked']) == len(set(trace['ranked'])) <= cfg['final_k']
                assert trace['tokens'] <= cfg['evidence_tokens']
                if group == 'baseline':
                    historical = SOURCE / 'strict_user_top20' / scope / 'retrieval.json'
                    if trace != read(historical):
                        raise ValueError(f'Baseline replay differs from the frozen trace: {scope}')
                row = dict(question_id=chosen['question_id'], group=group, evidence_tokens=trace['tokens'],
                           reader_prompt_tokens=tokenizer.messages(messages), retrieval_seconds=elapsed,
                           candidate_memories=len(trace['ranked']), index_entries=len(index.entries),
                           index_size_bytes=(stats['index_size_bytes'] if group == 'utterance'
                                             else baseline_path.stat().st_size))
                if args.answer:
                    result_path = dest / 'result.json'
                    if group == 'baseline':
                        source_result = SOURCE / 'strict_user_top20' / scope / 'result.json'
                        result = read(source_result)
                        save(dest / 'reuse.json', dict(source=str(source_result), reason='identical frozen replay'))
                    elif result_path.exists() and read(result_path)['status'] == 'returned':
                        result = read(result_path)
                    else:
                        ledger = dest / 'ledger.json'
                        if ledger.exists() and any(item['status'] in ('failed', 'interrupted_unknown')
                                                   for item in read(ledger).values()):
                            attempt = len(list(dest.glob('previous_ledger_*.json'))) + 1
                            ledger.rename(dest / f'previous_ledger_{attempt:02d}.json')
                        result = answer_question(dest, memory, packed, cfg, tokenizer,
                                                     chat_call(dest, frozen['reader'], original, tokenizer), retry_limit=512)
                    save(result_path, result)
                    row['reader_stats'] = result['stats']
                    if result['status'] == 'returned':
                        hypotheses[group].append(dict(question_id=chosen['question_id'], hypothesis=result['answer']))
                row.update(evidence_audit(sample, scope, archive, trace))
                diagnostics.append(row)
            save(out / 'diagnostics.json', diagnostics)
            for group in GROUPS:
                if args.answer:
                    write_lines(out / group / 'hypotheses.jsonl', hypotheses[group])
            print(f'{scope}: {len(baseline.entries)} → {len(variant.entries)} keys, '
                  f'marked source coverage {diagnostics[-2]["marked_with_fragment"]} → '
                  f'{diagnostics[-1]["marked_with_fragment"]} / {diagnostics[-1]["marked_turns"]}', flush=True)


if __name__ == '__main__':
    main()
