"""Prime BM25 adapter; document rendering adapted from STaRK (MIT, see NOTICE)."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sqlite3
import subprocess
import tempfile
import time
import zipfile

import bm25s
import numpy as np


def clean_dict(dictionary):
    # Same recursive cleaning as stark_qa.tools.process_text.clean_dict.
    return {k: clean_dict(v) if isinstance(v, dict) else v
            for k, v in dictionary.items() if isinstance(v, dict) or str(v) not in ('', 'nan')}


def render_document(node):
    """Equivalent to PrimeSKB.get_doc_info(add_rel=False, compact=False)."""
    doc = f"- name: {node['name']}\n- type: {node['type']}\n- source: {node['source']}\n"
    explanations = {
        'name': 'gene name', 'type_of_gene': 'gene types', 'alias': 'other gene names',
        'other_names': 'extended other gene names', 'genomic_pos': 'genomic position',
        'generif': 'PubMed text', 'interpro': 'protein family and classification information',
        'summary': 'protein summary text',
    }
    features = []
    for key, value in node.get('details', {}).items():
        if str(value) in ('', 'nan') or key.startswith('_') or '_id' in key:
            continue
        if node['type'] == 'gene/protein' and key in explanations:
            if 'interpro' in key:
                if isinstance(value, dict):
                    value = [value]
                value = [v['desc'] for v in value]
            if 'generif' in key:
                value = '; '.join(v['text'] for v in value)
                value = ' '.join(value.split(' ')[:50000])
            if 'genomic_pos' in key and isinstance(value, list):
                value = value[0]
            features.append(f'  - {key} ({explanations[key]}): {value}\n')
        else:
            features.append(f'  - {key}: {value}\n')
    return doc + ('- details:\n' + ''.join(features) if features else '')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def build(archive, output, stark_repo):
    """Build an index from KB data only; QA files are never read here."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Index already exists: {output}')
    started = time.monotonic()
    with zipfile.ZipFile(archive) as source:
        with source.open('processed/node_info.pkl') as stream:
            nodes = clean_dict(pickle.load(stream))
    # All ten Prime node types are official candidate types.
    candidate_types = {
        'disease', 'gene/protein', 'molecular_function', 'drug', 'pathway', 'anatomy',
        'effect/phenotype', 'biological_process', 'cellular_component', 'exposure',
    }
    ids = sorted(i for i, n in nodes.items() if n['type'] in candidate_types)
    if len(ids) != len(nodes):
        raise ValueError('Unexpected Prime node types')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + '.building-', dir=output.parent) as tmp:
        temp = Path(tmp)
        connection = sqlite3.connect(temp / 'documents.sqlite')
        connection.execute('CREATE TABLE documents (node_id INTEGER PRIMARY KEY, name TEXT, node_type TEXT, document TEXT)')
        documents = []
        for node_id in ids:
            node = nodes[node_id]
            document = render_document(node)
            documents.append(document)
            connection.execute('INSERT INTO documents VALUES (?, ?, ?, ?)',
                               (node_id, node['name'], node['type'], document))
        connection.commit()
        connection.close()
        del nodes
        # Same default BM25 and tokenize settings as STaRK. Omit corpus so
        # retrieve returns positions, fixing the upstream text/position mismatch.
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(documents, show_progress=False), show_progress=False)
        retriever.save(temp / 'bm25')
        np.save(temp / 'candidate_ids.npy', np.asarray(ids, dtype=np.int64), allow_pickle=False)
        source_paths = ['stark_qa/skb/prime.py', 'stark_qa/models/bm25.py', 'stark_qa/tools/process_text.py']
        manifest = {
            'dataset': 'prime', 'archive_sha256': sha256(archive),
            'stark_commit': subprocess.check_output(['git', '-C', str(stark_repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {p: sha256(Path(stark_repo) / p) for p in source_paths},
            'adapter_sha256': sha256(__file__), 'candidate_count': len(ids),
            'document_options': {'add_rel': False, 'compact': False},
            'versions': {name: importlib.metadata.version(name) for name in ['bm25s', 'numpy']},
            'build_seconds': time.monotonic() - started,
            'index_corpus_sha256': hashlib.sha256(''.join(documents).encode()).hexdigest(),
            'bm25_parameters': json.loads((temp / 'bm25/params.index.json').read_text()),
            'compatibility_fix': 'BM25(corpus=None): return positions before mapping to original candidate IDs',
        }
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        Path(tmp).rename(output)
    return manifest


class Retrieval:
    def __init__(self, index):
        self.index = Path(index).resolve()
        self.ids = np.load(self.index / 'candidate_ids.npy', allow_pickle=False)
        self.retriever = bm25s.BM25.load(self.index / 'bm25', mmap=True, load_corpus=False, show_progress=False)
        self.connection = sqlite3.connect((self.index / 'documents.sqlite').as_uri() + '?mode=ro',
                                          uri=True, check_same_thread=False)

    def document(self, node_id):
        row = self.connection.execute('SELECT name, node_type, document FROM documents WHERE node_id=?', (node_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown candidate node_id: {node_id}')
        return {'node_id': node_id, 'name': row[0], 'node_type': row[1], 'document': row[2]}

    def search(self, query, k=100):
        if not query.strip() or type(k) is not int or not 1 <= k <= 100:
            raise ValueError('Use a nonempty query and k between 1 and 100')
        positions, scores = self.retriever.retrieve(bm25s.tokenize(query, show_progress=False),
                                                   k=min(k, len(self.ids)), show_progress=False)
        return [{'node_id': int(self.ids[pos]), 'score': float(score)}
                for pos, score in zip(positions[0], scores[0])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--stark-repo', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.index, args.stark_repo), indent=2))
