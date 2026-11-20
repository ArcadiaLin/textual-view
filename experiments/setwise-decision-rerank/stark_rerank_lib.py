"""
STaRK-Prime Setwise & Pointwise Re-ranking utilities for Decision Models
(`AnyJev Tacit-9B`, `Kev-9B`, and `Google Gemini Decision 1`).

Uses the pre-retrieved BM25 shortlists in `runs/stark-prime/test-10pct/baseline.json`
and pre-rendered node documents in `data/stark-prime/bm25/documents.sqlite` (zero retrieval).
"""
from __future__ import annotations

import ast
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from rerank_lib import (
    EXP_DIR,
    PROJECT_ROOT,
    SCRATCH_DIR,
    AnyJevBackend,
    DiskCache,
    GeminiDecisionBackend,
    KevBackend,
    load_env,
    rerank_coarse_to_fine,
    rerank_oneshot_choice,
    rerank_pointwise,
    rerank_pointwise_fanout,
    rerank_setwise_heap,
    rerank_setwise_sliding_window,
    rerank_setwise_tournament,
)

STARK_BASELINE_JSON = PROJECT_ROOT / "runs" / "stark-prime" / "test-10pct" / "baseline.json"
STARK_QA_CSV = PROJECT_ROOT / "references" / "datasets" / "stark" / "qa" / "prime" / "stark_qa" / "stark_qa.csv"
STARK_SQLITE_DB = PROJECT_ROOT / "data" / "stark-prime" / "bm25" / "documents.sqlite"

STARK_NOUL_INSTRUCTIONS = (
    "You are evaluating a candidate entity from a biomedical knowledge base (STaRK-Prime) "
    "for a user query. Does this candidate entity satisfy all requirements, biomedical "
    "attributes, and relational conditions specified in the query?"
)
STARK_NOUL_TRUE_DESC = (
    "There is explicit and strong evidence in the candidate entity record supporting "
    "that this entity satisfies the requirements and type asked by the query."
)
STARK_NOUL_FALSE_DESC = (
    "The candidate entity does not satisfy the requirements in the query, or is of the "
    "wrong entity type or only partially/weakly related."
)
STARK_CHOICE_INSTRUCTIONS = (
    "Which candidate biomedical entity from the STaRK-Prime knowledge base best satisfies "
    "all requirements, attributes, and conditions specified in the query?"
)


def make_stark_anyjev(cache: DiskCache | None = None) -> AnyJevBackend:
    return AnyJevBackend(
        cache=cache,
        noul_instructions=STARK_NOUL_INSTRUCTIONS,
        noul_true_desc=STARK_NOUL_TRUE_DESC,
        noul_false_desc=STARK_NOUL_FALSE_DESC,
        choice_instructions=STARK_CHOICE_INSTRUCTIONS,
    )


def make_stark_kev(cache: DiskCache | None = None) -> KevBackend:
    return KevBackend(
        cache=cache,
        noul_instructions=STARK_NOUL_INSTRUCTIONS,
        noul_true_desc=STARK_NOUL_TRUE_DESC,
        noul_false_desc=STARK_NOUL_FALSE_DESC,
        choice_instructions=STARK_CHOICE_INSTRUCTIONS,
    )


def make_stark_gemini(cache: DiskCache | None = None) -> GeminiDecisionBackend:
    return GeminiDecisionBackend(
        cache=cache,
        noul_instructions=STARK_NOUL_INSTRUCTIONS,
        noul_true_desc=STARK_NOUL_TRUE_DESC,
        noul_false_desc=STARK_NOUL_FALSE_DESC,
        choice_instructions=STARK_CHOICE_INSTRUCTIONS,
    )


def load_stark_prime_top20(
    top_k: int = 20,
    max_doc_chars: int = 1200,
    only_covered: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], dict[str, list[str]], int]:
    """
    Load STaRK-Prime `test-0.1` pre-retrieved BM25 Top-K candidates and node documents.

    Returns:
        corpus: {str(node_id): truncated_document_text}
        queries: {str(query_id): query_text}
        golds: {str(query_id): set of str(gold_node_id)}
        candidates: {str(query_id): list of str(node_id) of length top_k}
        total_split_queries: total queries in test-0.1 (280)
    """
    raw_baseline = json.loads(STARK_BASELINE_JSON.read_text(encoding="utf-8"))
    total_split_queries = len(raw_baseline)

    all_queries: dict[str, str] = {}
    all_golds: dict[str, set[str]] = {}
    with STARK_QA_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row["id"])
            all_queries[qid] = row["query"]
            all_golds[qid] = {str(x) for x in ast.literal_eval(row["answer_ids"])}

    queries: dict[str, str] = {}
    golds: dict[str, set[str]] = {}
    candidates: dict[str, list[str]] = {}
    needed_node_ids: set[int] = set()

    for item in raw_baseline:
        qid = str(item["query_id"])
        cids = [str(x["node_id"]) for x in item["scores"][:top_k]]
        gold_set = all_golds[qid]
        if only_covered and not any(c in gold_set for c in cids):
            continue
        queries[qid] = all_queries[qid]
        golds[qid] = gold_set
        candidates[qid] = cids
        for c in cids:
            needed_node_ids.add(int(c))

    conn = sqlite3.connect(STARK_SQLITE_DB.as_uri() + "?mode=ro", uri=True)
    sorted_nids = sorted(needed_node_ids)
    corpus: dict[str, str] = {}
    # Fetch in chunks of 500 to stay below SQLite parameter limits
    for i in range(0, len(sorted_nids), 500):
        chunk = sorted_nids[i : i + 500]
        ph = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT node_id, document FROM documents WHERE node_id IN ({ph})", chunk
        ).fetchall()
        for nid, doc in rows:
            text = doc.strip()
            if max_doc_chars and len(text) > max_doc_chars:
                text = text[:max_doc_chars] + "..."
            corpus[str(nid)] = text
    conn.close()

    return corpus, queries, golds, candidates, total_split_queries


def evaluate_stark_rankings(
    rankings: dict[str, list[str]],
    golds: dict[str, set[str]],
    qids: list[str],
    total_split_queries: int = 280,
) -> dict[str, Any]:
    """
    Evaluate multi-label STaRK rankings following `experiments/stark-prime/score.py`.
    Computes both:
      - Oracle-covered subset metrics (over `len(qids)` queries, e.g. 115)
      - Full `test-0.1` 280-query end-to-end metrics (since uncovered Top-20 queries score 0 on Top-20 reranking).
    """
    n = len(qids)
    hit1_list, hit3_list, hit5_list, hit10_list = [], [], [], []
    rec5_list, rec10_list, rec20_list = [], [], []
    mrr_list, ndcg10_list, best_ranks = [], [], []
    per_query_best_rank: dict[str, int] = {}

    for q in qids:
        ranked = rankings[q]
        gold_set = golds[q]
        hits = [c in gold_set for c in ranked]
        first_hit_rank = next((i + 1 for i, h in enumerate(hits) if h), len(ranked) + 1)
        per_query_best_rank[q] = first_hit_rank
        best_ranks.append(first_hit_rank)

        hit1_list.append(1.0 if any(hits[:1]) else 0.0)
        hit3_list.append(1.0 if any(hits[:3]) else 0.0)
        hit5_list.append(1.0 if any(hits[:5]) else 0.0)
        hit10_list.append(1.0 if any(hits[:10]) else 0.0)

        n_gold = len(gold_set)
        rec5_list.append(sum(hits[:5]) / n_gold)
        rec10_list.append(sum(hits[:10]) / n_gold)
        rec20_list.append(sum(hits[:20]) / n_gold)

        mrr_list.append(1.0 / first_hit_rank if first_hit_rank <= len(ranked) else 0.0)

        # Multi-label NDCG@10
        dcg10 = sum((1.0 / math.log2(i + 2)) for i, h in enumerate(hits[:10]) if h)
        ideal_hits10 = min(10, len(gold_set & set(ranked)))
        idcg10 = sum((1.0 / math.log2(i + 2)) for i in range(ideal_hits10))
        ndcg10_list.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

    scale_280 = n / total_split_queries if total_split_queries > 0 else 1.0
    hit1 = sum(hit1_list) / n
    hit3 = sum(hit3_list) / n
    hit5 = sum(hit5_list) / n
    hit10 = sum(hit10_list) / n
    rec5 = sum(rec5_list) / n
    rec10 = sum(rec10_list) / n
    rec20 = sum(rec20_list) / n
    mrr = sum(mrr_list) / n
    ndcg10 = sum(ndcg10_list) / n

    return {
        "n_queries": n,
        "total_split_queries": total_split_queries,
        "hit1": hit1,
        "hit3": hit3,
        "hit5": hit5,
        "hit10": hit10,
        "recall5": rec5,
        "recall10": rec10,
        "recall20": rec20,
        "mrr": mrr,
        "ndcg10": ndcg10,
        "mean_best_rank": sum(best_ranks) / n,
        # Exact projections to all 280 test-0.1 queries for metrics bounded within Top-20
        "full280_hit1": hit1 * scale_280,
        "full280_hit5": hit5 * scale_280,
        "full280_hit10": hit10 * scale_280,
        "full280_recall20": rec20 * scale_280,
        "full280_mrr20": mrr * scale_280,
        "best_ranks": per_query_best_rank,
    }
