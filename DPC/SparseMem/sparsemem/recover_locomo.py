"""Retry truncated LoCoMo Reader responses using the saved request evidence."""

import argparse
from pathlib import Path
import time

from . import BASE
from .construct import Ledger
from .locomo import distribution
from .run import chat_call, usage, write_lines
from dpc_mem.io import RunLock, key, read, save
from dpc_mem.tokenization import HFTokenizer


def recover(root, limit):
    original = read(BASE / 'configs/compact_v3.json')
    tok = HFTokenizer(str(BASE / 'models/Qwen3-30B-A3B-Instruct-2507-tokenizer'))
    with RunLock(root):
        manifest = read(root / 'run_manifest.json')
        metrics = read(root / 'metrics.json')
        if manifest['dataset'] != 'locomo' or manifest['backend'] != 'real':
            raise ValueError('Recovery requires a real LoCoMo run')
        if manifest['reader'] != original['reader'] or limit <= manifest['memory_config']['answer_max']:
            raise ValueError('Recovery requires the original Reader and a larger output limit')
        answers = root / 'answers' / 'sparsemem' / manifest['mode']
        paths = [answers / (key(qid) + '.json') for qid in manifest['question_ids']]
        start = time.perf_counter()
        attempts = 0
        for path in paths:
            result = read(path)
            if result['status'] == 'ok':
                continue
            query = root / 'queries' / path.stem
            ledger = Ledger(query / 'ledger.json')
            requests = [read(p) for p in (query / 'calls').glob('*.json')]
            truncated = [r for r in requests if r['request']['max_tokens'] == manifest['memory_config']['answer_max']
                         and r.get('response', {}).get('choices', [{}])[0].get('finish_reason') == 'length']
            if len(truncated) != 1:
                raise ValueError(f"Cannot identify original truncated response for {result['question_id']}")
            call = chat_call(query, manifest['reader'], original, tok)
            retry_key = f'reader_recovery_{limit}'
            attempts += retry_key not in ledger.rows
            row = ledger.generate(retry_key, 'query', truncated[0]['request']['messages'],
                                  limit, manifest['memory_config']['margin'], None, tok, call)
            result['stats'].update(usage(list(ledger.rows.values())))
            if row['status'] == 'returned':
                result.update(status='ok', answer=row['content'], recovered_max_tokens=limit)
                result['stats']['query_wall_seconds'] += row['seconds']
            save(path, result)
            print(result['question_id'], result['status'], flush=True)
        if attempts:
            metrics['answer_recovery_wall_seconds'] = metrics.get('answer_recovery_wall_seconds', 0) + time.perf_counter() - start
        rows = list(map(read, paths))
        ledgers = [root / 'memory' / 'ledger.json'] + [root / 'queries' / p.stem / 'ledger.json' for p in paths]
        ledger = [dict(x, scope=str(p.parent.relative_to(root))) for p in ledgers if p.exists() for x in read(p).values()]
        metrics.update(usage(ledger), answer_failures=sum(r['status'] != 'ok' for r in rows))
        for metric in ('query_wall_seconds', 'query_generation_seconds'):
            metrics[metric + '_distribution'] = distribution([r['stats'][metric] for r in rows])
        metrics['answer_total_wall_seconds'] = metrics['answer_batch_wall_seconds'] + metrics.get('answer_recovery_wall_seconds', 0)
        write_lines(root / 'llm_usage.jsonl', ({k: v for k, v in x.items() if k != 'content'} for x in ledger))
        write_lines(root / 'hypotheses.jsonl', (dict(question_id=r['question_id'], hypothesis=r['answer']) for r in rows))
        save(root / 'metrics.json', metrics)
        if metrics['answer_failures']:
            raise RuntimeError('LoCoMo predictions still have failures')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--max-tokens', default=512, type=int)
    args = parser.parse_args()
    recover(args.out.resolve(), args.max_tokens)


if __name__ == '__main__':
    main()
