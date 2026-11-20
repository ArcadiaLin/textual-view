#!/usr/bin/env python3
"""
CLI runner to evaluate Pointwise vs. Setwise re-ranking methods on CLERC
and populate the deterministic disk cache (`notebooks/_scratch/decision_cache.json`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rerank_lib import (
    SCRATCH_DIR,
    AnyJevBackend,
    DiskCache,
    GeminiDecisionBackend,
    KevBackend,
    evaluate_rankings,
    load_clerc_slice,
    load_env,
    rerank_coarse_to_fine,
    rerank_oneshot_choice,
    rerank_pointwise,
    rerank_pointwise_fanout,
    rerank_setwise_heap,
    rerank_setwise_sliding_window,
    rerank_setwise_tournament,
)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Run Setwise Decision Model Re-ranking experiments")
    parser.add_argument("--backend", choices=["anyjev", "kev", "gemini", "all"], default="all")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of CLERC queries to evaluate (1..40)")
    args = parser.parse_args()

    corpus, queries, golds, candidates = load_clerc_slice()
    qids = list(queries)[: args.num_queries]
    cache = DiskCache(SCRATCH_DIR / "decision_cache.json")

    print(f"=== CLERC Setwise Decision Model Re-ranking ({len(qids)}/{len(queries)} queries, Top-30 BM25) ===")
    bm25_sub = {q: candidates[q] for q in qids}
    m_bm25 = evaluate_rankings(bm25_sub, golds, qids)
    print(
        f"[BM25 Fast Search] Top-1: {m_bm25['top1']:5.1%} | Top-5: {m_bm25['top5']:5.1%} | "
        f"Top-10: {m_bm25['top10']:5.1%} | NDCG@10: {m_bm25['ndcg10']:.3f} | MRR: {m_bm25['mrr']:.3f}"
    )

    summary_records = [
        {
            "backend": "BM25",
            "method": "BM25 Fast Search (Top-30)",
            **{k: m_bm25[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
            "calls_per_query": 0.0,
            "decisions_per_query": 0.0,
            "tokens_per_query": 0.0,
            "latency_s_per_query": 0.0,
        }
    ]

    if args.backend in ("anyjev", "all"):
        print("\n--- Local AnyJev (Tacit-9B @ 127.0.0.1:8100, Direct No-Proxy) ---")
        aj = AnyJevBackend(cache=cache)
        methods_aj = [
            ("Pointwise Noul (1 batch x 30)", lambda q: rerank_pointwise(aj, queries[q], candidates[q], corpus)),
            (
                "Setwise Tournament (6x5 -> 12, tagged, 2 calls)",
                lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "tagged"),
            ),
            (
                "Setwise Tournament (6x5 -> 12, inline, 2 calls)",
                lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "inline"),
            ),
            (
                "Setwise Sliding Window (w=6, s=4, tagged)",
                lambda q: rerank_setwise_sliding_window(aj, queries[q], candidates[q], corpus, 6, 4, 1, "tagged"),
            ),
            (
                "Setwise Heap Sort (c=4, top-10, tagged)",
                lambda q: rerank_setwise_heap(aj, queries[q], candidates[q], corpus, 3, 10, "tagged"),
            ),
            (
                "Coarse-to-Fine (Noul-30 -> Choice-10, tagged)",
                lambda q: rerank_coarse_to_fine(aj, queries[q], candidates[q], corpus, 10, "tagged"),
            ),
        ]
        for name, fn in methods_aj:
            aj.reset_stats()
            ranked = {q: fn(q) for q in qids}
            m = evaluate_rankings(ranked, golds, qids)
            nq = len(qids)
            rec = {
                "backend": "AnyJev (Tacit-9B)",
                "method": name,
                **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
                "calls_per_query": aj.calls / nq,
                "decisions_per_query": aj.decisions / nq,
                "tokens_per_query": 0.0,
                "latency_s_per_query": aj.total_latency_s / nq,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<47}] Top-1: {m['top1']:5.1%} | Top-5: {m['top5']:5.1%} | "
                f"Top-10: {m['top10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | MRR: {m['mrr']:.3f} | "
                f"Calls/q: {rec['calls_per_query']:4.1f} | Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    if args.backend in ("kev", "all"):
        print("\n--- Local Kev-9B (Qwen3.5-9B Pointer Head @ 127.0.0.1:8009, Direct No-Proxy) ---")
        kv = KevBackend(cache=cache)
        methods_kv = [
            (
                "Pointwise Noul (30 state-pairs, cookbook)",
                lambda q: rerank_pointwise(kv, queries[q], candidates[q], corpus),
            ),
            (
                "Pointwise Noul Fan-out (1 call x 30 Nouls)",
                lambda q: rerank_pointwise_fanout(kv, queries[q], candidates[q], corpus),
            ),
            (
                "One-shot Setwise Choice-30 (1 call, tagged)",
                lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="tagged"),
            ),
            (
                "Setwise Tournament (6x5 -> 12, tagged, 7 calls)",
                lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "tagged"),
            ),
            (
                "Setwise Tournament (6x5 -> 12, criteria, 7 calls)",
                lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "criteria"),
            ),
            (
                "Setwise Sliding Window (w=6, s=4, tagged)",
                lambda q: rerank_setwise_sliding_window(kv, queries[q], candidates[q], corpus, 6, 4, 1, "tagged"),
            ),
            (
                "Coarse-to-Fine (Pointwise Noul -> Choice-5, criteria)",
                lambda q: rerank_coarse_to_fine(kv, queries[q], candidates[q], corpus, 5, "criteria", use_fanout=False),
            ),
        ]
        for name, fn in methods_kv:
            kv.reset_stats()
            ranked = {q: fn(q) for q in qids}
            m = evaluate_rankings(ranked, golds, qids)
            nq = len(qids)
            rec = {
                "backend": "Kev-9B (Local)",
                "method": name,
                **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
                "calls_per_query": kv.calls / nq,
                "decisions_per_query": kv.decisions / nq,
                "tokens_per_query": kv.total_tokens / nq,
                "latency_s_per_query": kv.total_latency_s / nq,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<52}] Top-1: {m['top1']:5.1%} | Top-5: {m['top5']:5.1%} | "
                f"Top-10: {m['top10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | MRR: {m['mrr']:.3f} | "
                f"Calls/q: {rec['calls_per_query']:4.1f} | Tok/q: {rec['tokens_per_query']:6.0f} | "
                f"Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    if args.backend in ("gemini", "all"):
        print("\n--- Google Gemini Decision 1 (gemini-decision-1 via proxy 127.0.0.1:10809) ---")
        gm = GeminiDecisionBackend(cache=cache)
        methods_gm = [
            (
                "Pointwise Noul Fan-out (1 call x 30 Nouls)",
                lambda q: rerank_pointwise_fanout(gm, queries[q], candidates[q], corpus),
            ),
            (
                "One-shot Setwise Choice-30 (1 call, criteria)",
                lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="criteria"),
            ),
            (
                "One-shot Setwise Choice-30 (1 call, tagged)",
                lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="tagged"),
            ),
            (
                "Setwise Tournament (6x5 -> 12, criteria, 7 calls)",
                lambda q: rerank_setwise_tournament(gm, queries[q], candidates[q], corpus, 5, 2, "criteria"),
            ),
            (
                "Coarse-to-Fine (Fan-out Noul-30 -> Choice-10)",
                lambda q: rerank_coarse_to_fine(gm, queries[q], candidates[q], corpus, 10, "criteria", use_fanout=True),
            ),
        ]
        for name, fn in methods_gm:
            gm.reset_stats()
            ranked = {q: fn(q) for q in qids}
            m = evaluate_rankings(ranked, golds, qids)
            nq = len(qids)
            rec = {
                "backend": "Gemini Decision 1",
                "method": name,
                **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
                "calls_per_query": gm.calls / nq,
                "decisions_per_query": gm.decisions / nq,
                "tokens_per_query": gm.total_tokens / nq,
                "latency_s_per_query": gm.total_latency_s / nq,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<47}] Top-1: {m['top1']:5.1%} | Top-5: {m['top5']:5.1%} | "
                f"Top-10: {m['top10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | MRR: {m['mrr']:.3f} | "
                f"Calls/q: {rec['calls_per_query']:4.1f} | Tok/q: {rec['tokens_per_query']:6.0f} | "
                f"Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    out_path = SCRATCH_DIR / f"summary_q{len(qids)}.json"
    out_path.write_text(json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
