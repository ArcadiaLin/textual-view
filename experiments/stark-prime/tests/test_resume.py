"""Resume contracts without model calls: frozen inputs, skips, recovery, attempts."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import bm25s

import batch
from run import process_start, save
from score import score


class FakeRetrieval:
    def __init__(self, index):
        self.ids = np.array([1, 2])

    def search(self, query):
        return [{'node_id': 1, 'score': 1.0}, {'node_id': 2, 'score': 0.5}]


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.index, self.output = [self.root / n for n in ['data', 'index', 'output']]
        qa = self.data / 'qa/prime'
        (qa / 'split').mkdir(parents=True)
        (qa / 'stark_qa').mkdir()
        for split in ['val', 'test', 'test-0.1']:
            (qa / 'split' / f'{split}.index').write_text('1\n2\n')
        for name in ['stark_qa.csv', 'stark_qa_human_generated_eval.csv']:
            with (qa / 'stark_qa' / name).open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=['id', 'query', 'answer_ids'])
                writer.writeheader()
                writer.writerows([{'id': i, 'query': f'query {i}', 'answer_ids': f'[{i}]'} for i in [1, 2]])
        self.index.mkdir()
        save(self.index / 'manifest.json', {'fixture': True})
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *extra):
        argv = ['batch.py', '--data-root', str(self.data), '--index', str(self.index),
                '--output', str(self.output), *extra]
        with patch.object(sys, 'argv', argv), patch.object(batch, 'Retrieval', FakeRetrieval), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return batch.main()

    def fake_one(self, data, index, attempt, split, qid, timeout, turns, stop_event=None):
        self.calls.append((qid, attempt.name))
        attempt.mkdir(parents=True)
        prediction = json.dumps({'node_ids': [] if qid == 1 else [qid]})
        (attempt / 'final.txt').write_text(prediction)
        (attempt / 'prediction.json').write_text(prediction)
        (attempt / 'events.jsonl').write_text('original trace\n')
        metrics = score(prediction, [qid], {1, 2})
        metrics['format_compliant'] = True
        save(attempt / 'metrics.json', metrics)
        result = {'query_id': str(qid), 'status': 'scored', 'metrics': metrics,
                  'wall_seconds': 1.0, 'tool_calls': 2}
        save(attempt / 'result.json', result)
        return result

    def start(self):
        with patch.object(batch, 'run_one', self.fake_one):
            return self.cli('--split', 'test', '--all', '--workers', '1', '--timeout', '10', '--max-turns', '5')

    def test_resume_skips_scored_including_empty_zero_score(self):
        self.assertEqual(self.start(), 0)
        snapshots = {p: p.read_bytes() for p in self.output.glob('*/attempt_001/*') if p.is_file()}
        with patch.object(batch, 'run_one', side_effect=AssertionError('Must not call model')), \
                patch.object(FakeRetrieval, 'search', side_effect=AssertionError('Reuse baseline')):
            self.assertEqual(self.cli('--resume', '--workers', '2'), 0)
        self.assertEqual(self.calls, [(1, 'attempt_001'), (2, 'attempt_001')])
        for path, contents in snapshots.items():
            self.assertEqual(path.read_bytes(), contents)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['attempt_count'], 2)
        self.assertEqual(summary['pending'], 0)
        self.assertEqual(summary['pi']['hit_at_1'], 0.5)

    def test_interrupted_query_gets_new_attempt_and_keeps_old_trace(self):
        def interrupted(*args, **kwargs):
            qid, attempt = args[4], args[2]
            if qid == 1:
                return self.fake_one(*args, **kwargs)
            attempt.mkdir(parents=True)
            (attempt / 'events.jsonl').write_text('partial trace\n')
            result = {'query_id': str(qid), 'status': 'interrupted', 'wall_seconds': 0.5, 'tool_calls': 1}
            save(attempt / 'result.json', result)
            return result
        with patch.object(batch, 'run_one', interrupted):
            self.assertEqual(self.cli('--split', 'test', '--all', '--timeout', '10', '--max-turns', '5'), 1)
        self.calls.clear()
        with patch.object(batch, 'run_one', self.fake_one):
            self.assertEqual(self.cli('--resume'), 0)
        self.assertEqual(self.calls, [(2, 'attempt_002')])
        self.assertEqual((self.output / '2/attempt_001/events.jsonl').read_text(), 'partial trace\n')
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['attempt_count'], 3)
        self.assertEqual(summary['total_agent_wall_seconds'], 2.5)
        self.assertEqual(summary['total_tool_calls'], 5)

    def test_failed_queries_require_explicit_retry(self):
        self.start()
        failed = self.output / '2/attempt_001/result.json'
        save(failed, {'query_id': '2', 'status': 'failed', 'wall_seconds': 1, 'tool_calls': 2})
        original = failed.read_bytes()
        with patch.object(batch, 'run_one', side_effect=AssertionError('No automatic retry')):
            self.assertEqual(self.cli('--resume'), 1)
        self.assertEqual(failed.read_bytes(), original)
        self.calls.clear()
        with patch.object(batch, 'run_one', self.fake_one):
            self.assertEqual(self.cli('--resume', '--retry-failed'), 0)
        self.assertEqual(self.calls, [(2, 'attempt_002')])
        self.assertEqual(failed.read_bytes(), original)

    def test_completed_runtime_is_scored_without_model_call(self):
        self.start()
        attempt = self.output / '2/attempt_001'
        runtime = self.root / 'unfinished-runtime'
        runtime.mkdir()
        (runtime / 'final.txt').write_text('Explanation\n{"node_ids": [2]}')
        save(runtime / 'state.json', {'status': 'completed', 'tool_calls': 3, 'wall_seconds': 1.2})
        save(attempt / 'result.json', {'status': 'running', 'runtime_directory': str(runtime)})
        with patch.object(batch, 'run_one', side_effect=AssertionError('Recover without model')):
            self.assertEqual(self.cli('--resume'), 0)
        result = json.loads((attempt / 'result.json').read_text())
        self.assertTrue(result['recovered_without_model'])
        self.assertEqual(result['metrics']['hit_at_1'], 1)
        self.assertFalse(result['metrics']['format_compliant'])

    def test_changed_budget_or_data_is_rejected(self):
        self.start()
        before = (self.output / 'protocol.json').read_bytes()
        with patch.object(batch, 'run_one', side_effect=AssertionError('No calls on mismatched config')):
            for options in [('--timeout', '11'), ('--count', '1'), ('--split', 'val')]:
                with self.assertRaises(SystemExit) as error:
                    self.cli('--resume', *options)
                self.assertEqual(error.exception.code, 2)
            qa = self.data / 'qa/prime/stark_qa/stark_qa.csv'
            qa.write_text(qa.read_text() + '\n')
            with self.assertRaises(SystemExit):
                self.cli('--resume')
        self.assertEqual((self.output / 'protocol.json').read_bytes(), before)

    def test_exclusive_lock_and_legacy_run_refusal(self):
        with batch.run_lock(self.output, resume=False):
            with self.assertRaises(ValueError):
                with batch.run_lock(self.output, resume=True):
                    self.fail('lock should be exclusive')
        with self.assertRaises(SystemExit):
            self.cli('--resume')

    def test_missing_scored_artifacts_do_not_trigger_silent_rerun(self):
        self.start()
        (self.output / '1/attempt_001/final.txt').unlink()
        with patch.object(batch, 'run_one', side_effect=AssertionError('No silent rerun')):
            with self.assertRaises(SystemExit):
                self.cli('--resume')

    def test_human_split_uses_its_own_query_file(self):
        with patch.object(batch, 'run_one', self.fake_one):
            self.assertEqual(self.cli('--split', 'human_generated_eval', '--all'), 0)
        frozen = json.loads((self.output / 'protocol.json').read_text())
        self.assertEqual(list(frozen['data_sha256']), ['stark_qa/stark_qa_human_generated_eval.csv'])

    def test_sigterm_then_resume_through_real_subprocess_controller(self):
        # Real pi process boundary and signal handling, with an offline RPC
        # fixture in place of the model client. No model requests are made.
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(['query one', 'query two'], show_progress=False), show_progress=False)
        retriever.save(self.index / 'bm25', show_progress=False)
        np.save(self.index / 'candidate_ids.npy', np.array([1, 2]))
        sqlite3.connect(self.index / 'documents.sqlite').close()
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        fake = bin_dir / 'pi'
        fake.write_text('''#!/usr/bin/python3
import json,os,sys,time
from pathlib import Path
if '--version' in sys.argv:
 print('offline-rpc-fixture');sys.exit(0)
def emit(value):
 print(json.dumps(value),flush=True)
for line in sys.stdin:
 command=json.loads(line)
 if command['type']=='get_state':
  emit({'type':'response','id':'state','success':True,'data':{}})
 elif command['type']=='prompt':
  emit({'type':'response','id':'task','success':True,'data':{'disposition':'started'}})
  emit({'type':'turn_start'})
  marker=Path(os.environ['PRIME_TEST_MARKER'])
  if not marker.exists():
   marker.write_text('started')
   time.sleep(30)
  qid=2 if 'query 2' in command['message'] else 1
  emit({'type':'message_end','message':{'role':'assistant','stopReason':'stop','content':[{'type':'text','text':json.dumps({'node_ids':[qid]})}],'usage':{'input':1,'output':1}}})
  emit({'type':'agent_end','willRetry':False})
  emit({'type':'agent_settled','aborted':False})
''')
        fake.chmod(0o755)
        marker = self.root / 'started'
        env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], PRIME_TEST_MARKER=str(marker))
        command = [sys.executable, str(batch.ROOT / 'batch.py'), '--data-root', str(self.data),
                   '--index', str(self.index), '--output', str(self.output)]
        with (self.root / 'stdout').open('wb') as out, (self.root / 'stderr').open('wb') as err:
            process = subprocess.Popen(command + ['--split', 'test', '--all', '--workers', '1',
                                        '--timeout', '60', '--max-turns', '5'], env=env, stdout=out, stderr=err)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail((self.root / 'stderr').read_text())
                    time.sleep(0.05)
                self.assertTrue(marker.exists())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=15), 130)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        interrupted = json.loads((self.output / '1/attempt_001/result.json').read_text())
        self.assertEqual(interrupted['status'], 'interrupted')
        events = self.output / '1/attempt_001/events.jsonl'
        trace = events.read_bytes()
        resumed = subprocess.run(command + ['--resume'], env=env, capture_output=True, timeout=15)
        self.assertEqual(resumed.returncode, 0, resumed.stderr.decode())
        self.assertEqual(events.read_bytes(), trace)
        self.assertTrue((self.output / '1/attempt_002/final.txt').exists())
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['scored'], 2)
        self.assertEqual(summary['attempt_count'], 3)
        self.assertEqual(summary['pending'], 0)
        self.assertEqual(len(list((self.output / 'invocations').glob('*.json'))), 2)

    def test_orphan_cleanup_checks_process_identity(self):
        runtime = self.root / 'orphan'
        runtime.mkdir()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            identity = process_start(child.pid)
            save(runtime / 'state.json', {'pid': child.pid, 'process_start': 'wrong-identity'})
            batch.stop_orphan(runtime)
            self.assertIsNone(child.poll())
            save(runtime / 'state.json', {'pid': child.pid, 'process_start': identity})
            batch.stop_orphan(runtime)
            self.assertIsNotNone(child.wait(timeout=5))
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()


if __name__ == '__main__':
    unittest.main()
