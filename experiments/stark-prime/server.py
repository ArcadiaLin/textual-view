"""Read-only Prime BM25 tools over MCP stdio."""

import argparse
import contextlib
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from retrieval import Retrieval


def create_server(index):
    # bm25s may print loading diagnostics; stdout belongs to the MCP protocol.
    with contextlib.redirect_stdout(sys.stderr):
        retrieval = Retrieval(index)
    server = FastMCP('stark-prime', instructions='Search the STaRK-Prime biomedical candidate documents with BM25 and read their text. No embeddings or graph traversal.')

    @server.tool(annotations={'readOnlyHint': True, 'destructiveHint': False}, structured_output=True)
    def search(query: str, k: int = 100) -> dict[str, Any]:
        """Search Prime candidate documents using STaRK's BM25 defaults. Returns the top k original benchmark node_id integers and scores in rank order, with names and types. k is 1..100. Queries may be reformulated across calls. The index contains node attributes, not graph relationships. Use read_document to inspect the text of promising candidates. Scores from different queries are not calibrated probabilities."""
        results = retrieval.search(query, k)
        for hit in results:
            node = retrieval.document(hit['node_id'])
            hit.update(name=node['name'], node_type=node['node_type'])
        return {'query': query, 'results': results}

    @server.tool(annotations={'readOnlyHint': True, 'destructiveHint': False}, structured_output=True)
    def read_document(node_id: int, offset: int = 0, length: int = 6000) -> dict[str, Any]:
        """Read the same Prime document used by BM25 for an original benchmark node_id. offset and length are character counts; length is 1..12000. Follow next_offset to read the remaining text. Text includes node attributes only, with no added graph relationships."""
        if offset < 0 or not 1 <= length <= 12000:
            raise ValueError('offset must be nonnegative; length must be 1..12000')
        node = retrieval.document(node_id)
        text = node.pop('document')
        end = min(offset + length, len(text))
        return {**node, 'text': text[offset:end], 'total_chars': len(text),
                'next_offset': end if end < len(text) else None}

    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    args = parser.parse_args()
    create_server(args.index).run(transport='stdio')
