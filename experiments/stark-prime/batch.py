"""Resumable Prime evaluation with frozen inputs and independent pi sessions."""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from retrieval import Retrieval
from run import ROOT, process_start, run_one, save
from score import load_question, normalize_prediction, score

TERMINAL = {'scored', 'failed', 'parse_failed'}
METRICS = ['hit_at_1', 'hit_at_5', 'recall_at_20', 'mrr_at_100']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_ids(data_root, split):
    qa = data_root / 'qa/prime'
    if split == 'human_generated_eval':
        with (qa / 'stark_qa/stark_qa_human_generated_eval.csv').open(newline='') as source:
            return sorted(int(row['id']) for row in csv.DictReader(source))
    return sorted(int(i) for i in (qa / 'split' / f'{split}.index').read_text().split())


def protocol(args, selected):
    qa = args.data_root / 'qa/prime'
    paths = [qa / 'stark_qa' / ('stark_qa_human_generated_eval.csv' if args.split == 'human_generated_eval' else 'stark_qa.csv')]
    if args.split != 'human_generated_eval':
        paths.append(qa / 'split' / f'{args.split}.index')
    return {
        'version': 1, 'dataset': 'prime', 'split': args.split, 'query_ids': selected,
        'seed': args.seed, 'selection_mode': 'all' if args.all else 'sample',
        'timeout_seconds': args.timeout, 'max_turns': args.max_turns,
        'data_root': str(args.data_root), 'index': str(args.index),
        'data_sha256': {str(p.relative_to(qa)): digest(p) for p in paths},
        'index_manifest_sha256': digest(args.index / 'manifest.json'),
        'source_sha256': {p.name: digest(p) for p in sorted(ROOT.glob('*.py'))},
        'config_sha256': {name: digest(ROOT / 'config/pi' / name)
                          for name in ['models.json', 'settings.json', 'APPEND_SYSTEM.md']},
        'lock_sha256': digest(ROOT / 'uv.lock'), 'python': sys.version,
        'versions': {name: importlib.metadata.version(name) for name in ['bm25s', 'numpy', 'mcp']},
        'pi_version': subprocess.check_output(['pi', '--version'], text=True).strip(),
    }


@contextmanager
def run_lock(output, resume):
    if resume:
        if not output.is_dir():
            raise ValueError('--resume requires an existing output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another batch process already owns this output directory') from None
        yield


def freeze(args):
    path = args.output / 'protocol.json'
    prior = None
    if args.resume:
        if not path.exists():
            raise ValueError('No frozen protocol.json; historical pilot runs cannot be resumed in place')
        prior = json.loads(path.read_text())
    args.split = args.split or (prior['split'] if prior else 'val')
    args.seed = args.seed if args.seed is not None else (prior['seed'] if prior else 20261030)
    args.timeout = args.timeout or (prior['timeout_seconds'] if prior else 300)
    args.max_turns = args.max_turns or (prior['max_turns'] if prior else 30)
    if prior and args.count is None and not args.all:
        args.all = prior['selection_mode'] == 'all'
        args.count = len(prior['query_ids'])
    ids = split_ids(args.data_root, args.split)
    count = len(ids) if args.all else (args.count if args.count is not None else 10)
    if not 1 <= count <= len(ids):
        raise ValueError('count must be within the selected split')
    selected = ids if args.all else random.Random(args.seed).sample(ids, count)
    current = protocol(args, selected)
    if prior is not None:
        changed = [key for key in current if prior.get(key) != current[key]]
        if changed:
            raise ValueError('Frozen run configuration changed: ' + ', '.join(changed) + '; use a new output directory')
    else:
        save(path, current)
        save(args.output / 'selection.json', {
            'split': args.split, 'seed': args.seed, 'query_ids': selected, 'count': count,
            'workers': args.workers, 'timeout_seconds': args.timeout, 'max_turns': args.max_turns,
            'purpose': 'retrieval evaluation; interpretation depends on selected split and sample size',
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
    return selected


def stop_orphan(runtime):
    """Stop only the exact pi process owned by a crashed attempt, never a reused PID."""
    state_path = runtime / 'state.json'
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    pid, identity = state.get('pid'), state.get('process_start')
    if not pid or not identity or process_start(pid) != identity:
        return
    try:
        if os.getpgid(pid) != pid:
            raise ValueError('Unexpected process group for unfinished pi attempt')
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while process_start(pid) == identity and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_start(pid) == identity:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def recover(attempt, result, args, qid, candidates):
    """Recover a completed response without invoking the model a second time."""
    runtime = Path(result['runtime_directory']) if result.get('runtime_directory') else attempt
    if runtime != attempt and runtime.exists():
        stop_orphan(runtime)
        shutil.copytree(runtime, attempt, dirs_exist_ok=True)
    state_path = attempt / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('status') == 'completed' and (attempt / 'final.txt').exists():
        try:
            _, gold = load_question(args.data_root, args.split, qid)
            prediction, compliant = normalize_prediction((attempt / 'final.txt').read_text())
            metrics = score(prediction, gold, candidates)
            metrics['format_compliant'] = compliant
            (attempt / 'prediction.json').write_text(prediction + '\n')
            save(attempt / 'metrics.json', metrics)
            result.update(status='scored', metrics=metrics, recovered_without_model=True)
            result.pop('error', None)
        except (ValueError, json.JSONDecodeError) as error:
            result.update(status='parse_failed', error=f'{type(error).__name__}: {error}')
    else:
        result.update(status='interrupted')
        result.setdefault('error', 'Previous attempt did not publish a complete final answer')
    result.update(query_id=str(qid), usage_raw=state.get('usage_raw', result.get('usage_raw', {})),
                  tool_calls=state.get('tool_calls', result.get('tool_calls', 0)),
                  tool_errors=state.get('tool_errors', result.get('tool_errors', 0)),
                  wall_seconds=result.get('wall_seconds', state.get('wall_seconds', 0)))
    save(attempt / 'result.json', result)
    return result


def plan_question(args, qid, candidates):
    folder = args.output / str(qid)
    attempts = sorted((p for p in folder.glob('attempt_*') if p.is_dir() and p.name[8:].isdigit()),
                      key=lambda p: int(p.name[8:]))
    result = None
    if attempts:
        latest = attempts[-1]
        path = latest / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        if result.get('status') not in TERMINAL:
            result = recover(latest, result, args, qid, candidates)
        if result.get('status') == 'scored':
            # Never rerun an answer because its score is low or its list is empty.
            if not all((latest / name).is_file() for name in ['final.txt', 'prediction.json', 'metrics.json']):
                raise ValueError(f'Scored query {qid} is missing artifacts; refusing silent rerun')
            return result, None
        if result.get('status') in {'failed', 'parse_failed'} and not args.retry_failed:
            return result, None
    number = int(attempts[-1].name[8:]) + 1 if attempts else 1
    return result, folder / f'attempt_{number:03d}'


def ensure_baseline(args, selected, retrieval, candidates, stop):
    path = args.output / 'baseline.json'
    if path.exists():
        rows = json.loads(path.read_text())
        if [str(r['query_id']) for r in rows] != list(map(str, selected)):
            raise ValueError('Baseline question order differs from frozen selection')
        return rows
    folder = args.output / 'baseline'
    folder.mkdir(exist_ok=True)
    rows = []
    for qid in selected:
        if stop.is_set():
            raise InterruptedError('Interrupted while preparing the baseline; checkpoints retained')
        checkpoint = folder / f'{qid}.json'
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text())
            if row['query_id'] != str(qid):
                raise ValueError('Invalid baseline checkpoint')
        else:
            query, gold = load_question(args.data_root, args.split, qid)
            hits = retrieval.search(query)
            prediction = json.dumps({'node_ids': [h['node_id'] for h in hits]})
            row = {'query_id': str(qid), 'scores': hits, 'metrics': score(prediction, gold, candidates)}
            save(checkpoint, row)
        rows.append(row)
    save(path, rows)
    return rows


def summarize(output, selected, results, baseline, totals):
    rows = [results[qid] for qid in selected if qid in results]
    statuses = Counter(r['status'] for r in rows)
    summary = {
        'questions': len(selected), 'scored': statuses['scored'],
        'pending': len(selected) - sum(statuses[s] for s in TERMINAL),
        'failed_query_ids': [str(q) for q in selected if results.get(q, {}).get('status') in {'failed', 'parse_failed'}],
        'strict_format_compliant': sum(r.get('metrics', {}).get('format_compliant', False) for r in rows),
        'failed_questions_count_as_zero': True,
        'metric_scope': 'all selected questions; pending and failed predictions currently count as zero',
        'baseline': {m: sum(r['metrics'][m] for r in baseline) / len(selected) for m in METRICS},
        'pi': {m: sum(r.get('metrics', {}).get(m, 0) for r in rows if r['status'] == 'scored') / len(selected) for m in METRICS},
        'attempt_count': totals['attempt_count'],
        'total_agent_wall_seconds': totals['wall_seconds'],
        'total_tool_calls': totals['tool_calls'],
        'usage_scope': 'all recorded attempts, including interrupted and failed attempts',
    }
    save(output / 'summary.json', summary)
    return summary


def execute(args, selected, candidates, baseline, stop):
    results, jobs = {}, deque()
    totals = Counter()
    for qid in selected:
        result, attempt = plan_question(args, qid, candidates)
        if result:
            results[qid] = result
        if attempt:
            jobs.append((qid, attempt))
        for path in (args.output / str(qid)).glob('attempt_*/result.json'):
            history = json.loads(path.read_text())
            totals.update(attempt_count=1, wall_seconds=history.get('wall_seconds', 0),
                          tool_calls=history.get('tool_calls', 0))
    summarize(args.output, selected, results, baseline, totals)

    def one(qid, attempt):
        try:
            return run_one(args.data_root, args.index, attempt, args.split, qid,
                           args.timeout, args.max_turns, stop_event=stop)
        except Exception as error:
            attempt.mkdir(parents=True, exist_ok=True)
            result = {'query_id': str(qid), 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            save(attempt / 'result.json', result)
            return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        while jobs or active:
            while jobs and len(active) < args.workers and not stop.is_set():
                qid, attempt = jobs.popleft()
                active[pool.submit(one, qid, attempt)] = qid
            if not active:
                break
            done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                qid = active.pop(future)
                result = future.result()
                results[qid] = result
                totals.update(attempt_count=1, wall_seconds=result.get('wall_seconds', 0),
                              tool_calls=result.get('tool_calls', 0))
                summary = summarize(args.output, selected, results, baseline, totals)
                print(json.dumps({'query_id': qid, 'status': result['status'],
                                  'hit_at_1': result.get('metrics', {}).get('hit_at_1'),
                                  'scored': summary['scored'], 'pending': summary['pending']}), flush=True)
    return summarize(args.output, selected, results, baseline, totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'test-0.1', 'human_generated_eval'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--count', type=int)
    group.add_argument('--all', action='store_true', help='run every question of the selected split')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='explicitly retry failed/parse_failed questions; scored questions are always skipped')
    args = parser.parse_args()
    if args.workers < 1 or any(v is not None and v < 1 for v in [args.count, args.timeout, args.max_turns]):
        parser.error('count, workers, timeout and max-turns must be positive')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    for key in ['data_root', 'index', 'output']:
        setattr(args, key, getattr(args, key).resolve())
    stop = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in [signal.SIGINT, signal.SIGTERM]}
    try:
        with run_lock(args.output, args.resume):
            selected = freeze(args)
            invocations = args.output / 'invocations'
            invocations.mkdir(exist_ok=True)
            save(invocations / f'{uuid.uuid4().hex}.json', {
                'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                'resume': args.resume, 'retry_failed': args.retry_failed, 'workers': args.workers,
                'protocol_sha256': digest(args.output / 'protocol.json'),
            })
            retrieval = Retrieval(args.index)
            candidates = set(retrieval.ids.tolist())
            baseline = ensure_baseline(args, selected, retrieval, candidates, stop)
            summary = execute(args, selected, candidates, baseline, stop)
            print(json.dumps(summary, indent=2))
            return 130 if stop.is_set() else (0 if summary['scored'] == len(selected) else 1)
    except InterruptedError as error:
        print(str(error), file=sys.stderr)
        return 130
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
