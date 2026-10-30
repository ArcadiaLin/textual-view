"""Independent scoring of ranked node IDs, following DKGI's task.py conventions."""

import argparse
import ast
import csv
import json
from pathlib import Path


def normalize_prediction(text):
    """Extract only an unambiguous answer object from the final assistant text.

    Keep the original text separately and flag wrappers as format violations.
    Never search earlier assistant messages or infer/correct any predicted IDs.
    """
    try:
        value = json.loads(text)
        return json.dumps(value), True
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        objects = []
        for i, char in enumerate(text):
            if char != '{':
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and set(obj) == {'node_ids'}:
                objects.append(obj)
        if len(objects) != 1:
            raise ValueError('Final text must contain exactly one unambiguous node_ids object')
        return json.dumps(objects[0]), False


def load_question(data_root, split, query_id):
    root = Path(data_root) / 'qa/prime'
    filename = 'stark_qa_human_generated_eval.csv' if split == 'human_generated_eval' else 'stark_qa.csv'
    if split != 'human_generated_eval':
        split_ids = (root / 'split' / f'{split}.index').read_text().split()
        if str(query_id) not in split_ids:
            raise ValueError(f'query_id {query_id} is not in {split}')
    with (root / 'stark_qa' / filename).open(newline='') as source:
        for row in csv.DictReader(source):
            if row['id'] == str(query_id):
                return row['query'], ast.literal_eval(row['answer_ids'])
    raise ValueError(f'Unknown query_id: {query_id}')


def score(prediction, gold, candidates):
    parsed = json.loads(prediction)
    if not isinstance(parsed, dict) or set(parsed) != {'node_ids'}:
        raise ValueError('Expected exactly {"node_ids": [...]}')
    ranked = parsed['node_ids']
    if not isinstance(ranked, list) or len(ranked) > 100:
        raise ValueError('node_ids must be a list with at most 100 entries')
    gold = set(gold)
    if not gold or any(type(i) is not int or i < 0 for i in gold):
        raise ValueError('Invalid gold IDs')
    seen, hits, invalid = set(), [], []
    for rank, node_id in enumerate(ranked, 1):
        valid = type(node_id) is int and node_id in candidates and node_id not in seen
        if valid:
            seen.add(node_id)
        else:
            invalid.append(rank)
        hits.append(valid and node_id in gold)
    metrics = {'mrr_at_100': next((1 / i for i, hit in enumerate(hits, 1) if hit), 0.0),
               'returned_count': len(ranked), 'invalid_or_duplicate_ranks': invalid,
               'candidate_coverage': len(gold & candidates) / len(gold),
               'unreachable_gold_count': len(gold - candidates)}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        metrics[f'hit_at_{k}'] = float(any(hits[:k]))
        metrics[f'recall_at_{k}'] = sum(hits[:k]) / len(gold)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    task = json.loads((args.attempt / 'task.json').read_text())
    _, gold = load_question(args.data_root, task['split'], task['query_id'])
    import numpy as np
    candidates = set(np.load(Path(task['index']) / 'candidate_ids.npy', allow_pickle=False).tolist())
    prediction, format_compliant = normalize_prediction((args.attempt / 'final.txt').read_text())
    metrics = score(prediction, gold, candidates)
    metrics['format_compliant'] = format_compliant
    (args.attempt / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
