"""Run a fresh pi RPC process for one Prime query; score only outside pi."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from score import load_question, normalize_prediction, score

ROOT = Path(__file__).resolve().parent


def save(path, value):
    """Publish complete JSON records atomically, including on interruption."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def process_start(pid):
    """Linux process identity, so an orphan check cannot target a reused PID."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def extract_final(last_assistant):
    """Use the final authoritative message; never concatenate streaming deltas."""
    if not last_assistant:
        raise ValueError('No final assistant message')
    reason = last_assistant.get('stopReason')
    blocks = last_assistant.get('content', [])
    if reason in ('error', 'aborted', 'toolUse', 'length') or any(b.get('type') == 'toolCall' for b in blocks):
        raise ValueError(f'Incomplete final answer: {reason}')
    text = '\n'.join(b.get('text', '') for b in blocks if b.get('type') == 'text').strip()
    if not text:
        raise ValueError('Empty final answer')
    return text


def prepare(attempt, query, split, query_id, index):
    for folder in ['agent', 'work', 'sessions']:
        (attempt / folder).mkdir(parents=True)
    for name in ['models.json', 'settings.json', 'APPEND_SYSTEM.md']:
        shutil.copyfile(ROOT / 'config/pi' / name, attempt / 'agent' / name)
    mcp = {'mcpServers': {'stark-prime': {
        'command': sys.executable,
        'args': [str(ROOT / 'server.py'), '--index', str(index)],
        'exposure': 'direct', 'timeout': 60,
        'description': 'BM25 search and document reading for STaRK-Prime biomedical candidate nodes',
    }}}
    save(attempt / 'agent/mcp.json', mcp)
    prompt = (
        'Find candidate node IDs answering the following STaRK-Prime retrieval query. '
        'Use the supplied BM25 search and document-reading tools as evidence. You may reformulate searches. '
        'The indexed documents contain node attributes only; no graph traversal tool is provided. '
        'Return at most 100 unique benchmark node_id integers, ranked by relevance. '
        'Use IDs observed in tool results, not external IDs in the text; do not invent or pad IDs. '
        'Return only JSON with exactly this structure: {"node_ids": [123, 456]}. '
        'Return an empty list if no candidates are supported.\n\n'
        f'Query: {query}\n'
    )
    (attempt / 'prompt.txt').write_text(prompt)
    save(attempt / 'task.json', {'dataset': 'prime', 'split': split, 'query_id': str(query_id),
                                 'query': query, 'index': str(index)})
    return prompt


async def run_rpc(attempt, prompt, timeout, max_turns, stop_event=None):
    command = ['pi', '--mode', 'rpc', '--session-dir', str(attempt / 'sessions'),
               '--tools', 'mcp__stark_prime__*', '--no-skills', '--no-prompt-templates', '--no-themes']
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(attempt / 'agent'),
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    save(attempt / 'command.json', command)
    counts, usage = Counter(), Counter()
    state = {'status': 'running', 'turns': 0, 'last_assistant': None, 'prompt_accepted': False,
             'usage_scope': 'Provider-reported assistant message usage; unreported requests excluded'}
    started = time.monotonic()
    with (attempt / 'stderr.txt').open('wb') as errors, (attempt / 'events.jsonl').open('wb') as events:
        process = await asyncio.create_subprocess_exec(*command, cwd=attempt / 'work', env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=errors,
            start_new_session=True, limit=16 * 1024 * 1024)
        state.update(pid=process.pid, process_start=process_start(process.pid))
        save(attempt / 'state.json', state)

        async def send(value):
            process.stdin.write(json.dumps(value).encode() + b'\n')
            await process.stdin.drain()

        async def consume():
            # get_state forces model/config initialization before submitting the task.
            await send({'id': 'state', 'type': 'get_state'})
            while True:
                line = await process.stdout.readline()
                if not line:
                    raise RuntimeError('Pi exited before agent_settled')
                events.write(line)
                events.flush()
                event = json.loads(line)
                kind = event.get('type')
                counts[kind] += 1
                if kind == 'response':
                    if not event.get('success'):
                        raise RuntimeError(str(event.get('error')))
                    if event.get('id') == 'state':
                        save(attempt / 'initial_state.json', event.get('data'))
                        await send({'id': 'task', 'type': 'prompt', 'message': prompt})
                    elif event.get('id') == 'task':
                        if event.get('data', {}).get('disposition') != 'started':
                            raise RuntimeError('Prompt did not start a fresh run')
                        state['prompt_accepted'] = True
                elif kind == 'message_end' and event.get('message', {}).get('role') == 'assistant':
                    state['last_assistant'] = event['message']
                    for key in ['input', 'output', 'cacheRead', 'cacheWrite']:
                        usage[key] += event['message'].get('usage', {}).get(key, 0)
                elif kind == 'turn_start':
                    state['turns'] += 1
                    if state['turns'] > max_turns:
                        state['budget_exceeded'] = True
                        await send({'id': 'budget-abort', 'type': 'abort'})
                elif kind == 'tool_execution_end':
                    state['tool_calls'] = state.get('tool_calls', 0) + 1
                    state['tool_errors'] = state.get('tool_errors', 0) + int(bool(event.get('isError')))
                elif kind == 'agent_settled':
                    if event.get('aborted') or state.get('budget_exceeded'):
                        raise RuntimeError('Agent aborted or turn budget exceeded')
                    if not state['prompt_accepted']:
                        raise RuntimeError('Settled without accepted prompt')
                    return extract_final(state['last_assistant'])

        async def interruptible_consume():
            task = asyncio.create_task(consume())
            try:
                while not task.done():
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError('Batch interrupted; preserve this attempt for resume')
                    await asyncio.wait({task}, timeout=0.2)
                return task.result()
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        try:
            final = await asyncio.wait_for(interruptible_consume(), timeout)
            (attempt / 'final.txt').write_text(final + '\n')
            state['status'] = 'completed'
        except BaseException as error:
            interrupted = isinstance(error, (InterruptedError, KeyboardInterrupt, asyncio.CancelledError))
            state.update(status='interrupted' if interrupted else 'failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            state.update(event_counts=dict(counts), usage_raw=dict(usage), wall_seconds=time.monotonic() - started)
            save(attempt / 'state.json', state)
            if process.stdin:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                pass
            # Kill any MCP/tool descendants as well as a stalled pi process.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
    return final, state


def run_one(data_root, index, output, split, query_id, timeout=300, max_turns=30, stop_event=None):
    import numpy as np
    output = Path(output).resolve()
    index = Path(index).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch_root = Path('/tmp/stark-prime')
    scratch_root.mkdir(exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix=f'query-{query_id}-', dir=scratch_root))
    # Gold stays in this parent process and is not written into agent/work/prompt.
    query, gold = load_question(data_root, split, query_id)
    prompt = prepare(runtime, query, split, query_id, index)
    result = {'dataset': 'prime', 'split': split, 'query_id': str(query_id), 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(), 'timeout_seconds': timeout,
              'max_turns': max_turns, 'pi_version': subprocess.check_output(['pi', '--version'], text=True).strip(),
              'runtime_directory': str(runtime),
              'python': sys.version, 'system_prompt': 'pi default plus APPEND_SYSTEM.md output contract',
              'tools': ['search', 'read_document'],
              'index_manifest_sha256': hashlib.sha256((index / 'manifest.json').read_bytes()).hexdigest(),
              'versions': {name: importlib.metadata.version(name) for name in ['bm25s', 'mcp', 'numpy']},
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
              'isolation': 'separate config and cwd; not an OS filesystem sandbox'}
    save(output / 'result.json', result)
    started = time.monotonic()
    try:
        final, state = asyncio.run(run_rpc(runtime, prompt, timeout, max_turns, stop_event))
        result.update(status='completed', tool_calls=state.get('tool_calls', 0),
                      tool_errors=state.get('tool_errors', 0), usage_raw=state['usage_raw'])
        candidates = set(np.load(index / 'candidate_ids.npy', allow_pickle=False).tolist())
        result['status'] = 'parse_failed'
        prediction, format_compliant = normalize_prediction(final)
        (output / 'prediction.json').write_text(prediction + '\n')
        metrics = score(prediction, gold, candidates)
        metrics['format_compliant'] = format_compliant
        save(output / 'metrics.json', metrics)
        result.update(status='scored', metrics=metrics)
    except Exception as error:
        status = 'interrupted' if isinstance(error, InterruptedError) else (
            'parse_failed' if result['status'] == 'parse_failed' else 'failed')
        result.update(status=status,
                      error=f'{type(error).__name__}: {error}')
        state_path = runtime / 'state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            result.update(tool_calls=state.get('tool_calls', 0), tool_errors=state.get('tool_errors', 0),
                          usage_raw=state.get('usage_raw', {}))
    result.update(wall_seconds=time.monotonic() - started, finished_at=datetime.now(timezone.utc).isoformat())
    shutil.copytree(runtime, output, dirs_exist_ok=True)
    save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'test-0.1', 'human_generated_eval'], default='val')
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--max-turns', type=int, default=30)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_turns <= 0:
        parser.error('timeout and max-turns must be positive')
    result = run_one(args.data_root, args.index, args.output, args.split, args.query_id, args.timeout, args.max_turns)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['status'] == 'scored' else 1)
