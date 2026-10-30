"""Small reproducible validation pilot: BM25 baseline and independent pi sessions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

from retrieval import Retrieval
from run import run_one, save
from score import load_question, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--max-turns', type=int, default=30)
    args = parser.parse_args()
    ids = sorted(int(i) for i in (args.data_root / 'qa/prime/split/val.index').read_text().split())
    if not 1 <= args.count <= len(ids) or min(args.workers, args.timeout, args.max_turns) < 1:
        parser.error('invalid count, workers, timeout or max-turns')
    selected = random.Random(args.seed).sample(ids, args.count)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'selection.json', {'split': 'val', 'seed': args.seed, 'query_ids': selected,
        'count': args.count, 'workers': args.workers, 'timeout_seconds': args.timeout, 'max_turns': args.max_turns,
        'purpose': 'engineering pilot, not a formal benchmark result',
        'created_at': datetime.now(timezone.utc).isoformat()})
    retrieval = Retrieval(args.index)
    candidates = set(retrieval.ids.tolist())
    baseline = []
    for qid in selected:
        query, gold = load_question(args.data_root, 'val', qid)
        hits = retrieval.search(query)
        prediction = json.dumps({'node_ids': [h['node_id'] for h in hits]})
        baseline.append({'query_id': str(qid), 'scores': hits, 'metrics': score(prediction, gold, candidates)})
    save(args.output / 'baseline.json', baseline)

    def one(qid):
        result = run_one(args.data_root, args.index, args.output / str(qid) / 'attempt_001',
                         'val', qid, args.timeout, args.max_turns)
        print(json.dumps({'query_id': qid, 'status': result['status'],
                          'hit_at_1': result.get('metrics', {}).get('hit_at_1'),
                          'wall_seconds': result['wall_seconds']}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, selected))
    metric_names = ['hit_at_1', 'hit_at_5', 'recall_at_20', 'mrr_at_100']
    summary = {'questions': len(selected), 'scored': sum(r['status'] == 'scored' for r in results),
               'strict_format_compliant': sum(r.get('metrics', {}).get('format_compliant', False) for r in results),
               'failed_query_ids': [r['query_id'] for r in results if r['status'] != 'scored'],
               'failed_questions_count_as_zero': True,
               'baseline': {m: sum(r['metrics'][m] for r in baseline) / len(selected) for m in metric_names},
               'pi': {m: sum(r.get('metrics', {}).get(m, 0) for r in results) / len(selected) for m in metric_names},
               'total_agent_wall_seconds': sum(r['wall_seconds'] for r in results),
               'total_tool_calls': sum(r.get('tool_calls', 0) for r in results),
               'usage_scope': 'reported assistant responses; see per-attempt state.json for failure/retry events'}
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['scored'] == len(selected) else 1


if __name__ == '__main__':
    sys.exit(main())
