"""Meaningful contracts: source parity, ranking validation, final-message extraction."""

import ast
import json
import os
from pathlib import Path
import pickle
import random
import sys
import unittest
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import clean_dict, render_document
from run import extract_final
from score import normalize_prediction, score

REFERENCES = Path(os.environ.get('STARK_REFERENCE_ROOT', '/home/lzx/projs/textual-view/references'))


def reference_function(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'json': json}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


class ChainTests(unittest.TestCase):
    def test_output_normalization_never_guesses_or_selects_between_answers(self):
        self.assertEqual(normalize_prediction('{"node_ids":[1]}'), ('{"node_ids": [1]}', True))
        self.assertEqual(normalize_prediction('Explanation.\n```json\n{"node_ids":[2]}\n```'),
                         ('{"node_ids": [2]}', False))
        for text in ['No answer', '{"node_ids":[1]} and {"node_ids":[2]}']:
            with self.assertRaises(ValueError):
                normalize_prediction(text)

    def test_prime_documents_match_official_source(self):
        # Compile the unmodified official method only, avoiding its unrelated
        # GPU/embedding/model imports; compare on actual data across all types.
        method = reference_function(REFERENCES / 'repos/stark/stark_qa/skb/prime.py', 'get_doc_info', 'PrimeSKB')
        cleaner = reference_function(REFERENCES / 'repos/stark/stark_qa/tools/process_text.py', 'clean_dict')
        with zipfile.ZipFile(REFERENCES / 'datasets/stark/skb/prime/processed.zip') as archive:
            with archive.open('processed/node_info.pkl') as source:
                raw = pickle.load(source)
        nodes = clean_dict(raw)
        self.assertEqual(nodes, cleaner(raw))
        class Reference:
            node_info = nodes
            def __getitem__(self, idx):
                return SimpleNamespace(**self.node_info[idx])
        ids = random.Random(20261030).sample(sorted(nodes), 1000)
        # Ensure at least one representative of each node type.
        by_type = {}
        for i, node in nodes.items():
            by_type.setdefault(node['type'], i)
        for i in ids + list(by_type.values()):
            self.assertEqual(render_document(nodes[i]), method(Reference(), i), i)

    def test_scoring_matches_dkgi_with_invalid_and_duplicate_ids(self):
        reference = reference_function(REFERENCES / 'repos/DKGI/experiments/stark/task.py', 'score')
        for ids in [[], [1, 2], [1, 1, 999, True, 2], [999, 2, 1], [0] * 100]:
            prediction = json.dumps({'node_ids': ids})
            self.assertEqual(score(prediction, [1, 2, 7], {0, 1, 2}), reference(prediction, [1, 2, 7], {0, 1, 2}))
        for prediction in ['```json\n{}\n```', '{}', '{"node_ids":[],"explanation":"x"}', '{"node_ids":"1"}']:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                score(prediction, [1], {1})

    def test_final_only_uses_text_and_rejects_unfinished_tool_calls(self):
        message = {'stopReason': 'stop', 'content': [{'type': 'thinking', 'thinking': 'intermediate'},
                                                    {'type': 'text', 'text': '{"node_ids":[1]}'}]}
        self.assertEqual(extract_final(message), '{"node_ids":[1]}')
        for reason in ['error', 'aborted', 'toolUse', 'length']:
            with self.assertRaises(ValueError):
                extract_final({**message, 'stopReason': reason})
        with self.assertRaises(ValueError):
            extract_final({'stopReason': 'stop', 'content': [{'type': 'toolCall', 'name': 'search'}]})


if __name__ == '__main__':
    unittest.main()
