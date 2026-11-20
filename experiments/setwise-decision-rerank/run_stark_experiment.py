#!/usr/bin/env python3
"""
CLI runner to evaluate Pointwise vs. Setwise re-ranking methods on STaRK-Prime
(Top-20 pre-retrieved BM25 candidates from `runs/stark-prime/test-10pct/baseline.json`,
115 oracle-covered queries) across `AnyJev Tacit-9B`, `Kev-9B`, and `Gemini Decision 1`,
and populate `notebooks/_scratch/stark_decision_cache.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from rerank_lib import (
    SCRATCH_DIR,
    DiskCache,
    load_env,
    rerank_coarse_to_fine,
    rerank_oneshot_choice,
    rerank_pointwise,
    rerank_pointwise_fanout,
    rerank_setwise_tournament,
)
from stark_rerank_lib import (
    evaluate_stark_rankings,
    load_stark_prime_top20,
    make_stark_anyjev,
    make_stark_gemini,
    make_stark_kev,
)


def _run_method_queries(qids: list[str], fn, max_workers: int = 1) -> dict[str, list[str]]:
    if max_workers <= 1:
        return {q: fn(q) for q in qids}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(fn, qids))
    return dict(zip(qids, results))


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Run STaRK-Prime Top-20 Setwise Decision Model Re-ranking")
    parser.add_argument("--backend", choices=["anyjev", "kev", "gemini", "all"], default="all")
    parser.add_argument("--num-queries", type=int, default=115, help="Number of covered queries to evaluate (1..115)")
    args = parser.parse_args()

    corpus, queries, golds, candidates, total_split_q = load_stark_prime_top20(
        top_k=20, max_doc_chars=1200, only_covered=True
    )
    qids = list(queries)[: args.num_queries]
    cache = DiskCache(SCRATCH_DIR / "stark_decision_cache.json", flush_every=15)

    print(
        f"=== STaRK-Prime Top-20 Re-ranking ({len(qids)}/{len(queries)} covered queries; "
        f"full test-0.1 split = {total_split_q} queries) ==="
    )
    bm25_sub = {q: candidates[q] for q in qids}
    m_bm25 = evaluate_stark_rankings(bm25_sub, golds, qids, total_split_queries=total_split_q)
    print(
        f"[BM25 Top-20 Baseline] Hit@1: {m_bm25['hit1']:5.1%} (280q: {m_bm25['full280_hit1']:5.1%}) | "
        f"Hit@5: {m_bm25['hit5']:5.1%} (280q: {m_bm25['full280_hit5']:5.1%}) | "
        f"Hit@10: {m_bm25['hit10']:5.1%} | NDCG@10: {m_bm25['ndcg10']:.3f} | MRR: {m_bm25['mrr']:.3f}"
    )

    summary_records = [
        {
            "backend": "BM25",
            "method": "BM25 Fast Search (Top-20)",
            **{
                k: m_bm25[k]
                for k in (
                    "hit1",
                    "hit3",
                    "hit5",
                    "hit10",
                    "recall5",
                    "recall10",
                    "recall20",
                    "ndcg10",
                    "mrr",
                    "mean_best_rank",
                    "full280_hit1",
                    "full280_hit5",
                    "full280_hit10",
                    "full280_mrr20",
                )
            },
            "calls_per_query": 0.0,
            "decisions_per_query": 0.0,
            "tokens_per_query": 0.0,
            "latency_s_per_query": 0.0,
        }
    ]

    if args.backend in ("anyjev", "all"):
        print("\n--- 1. Local AnyJev (Tacit-9B @ 127.0.0.1:8100, Direct No-Proxy) ---")
        aj = make_stark_anyjev(cache=cache)
        methods_aj = [
            (
                "Pointwise Noul (1 batch x 20)",
                lambda q: rerank_pointwise(aj, queries[q], candidates[q], corpus),
            ),
            (
                "One-shot Setwise Choice-20 (1 call, tagged)",
                lambda q: rerank_oneshot_choice(aj, queries[q], candidates[q], corpus, mode="tagged"),
            ),
            (
                "Setwise Tournament (4x5 -> 8, tagged, 2 calls)",
                lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "tagged"),
            ),
            (
                "Coarse-to-Fine (Noul-20 -> Choice-10, tagged)",
                lambda q: rerank_coarse_to_fine(aj, queries[q], candidates[q], corpus, 10, "tagged"),
            ),
        ]
        for name, fn in methods_aj:
            aj.reset_stats()
            t0 = time.perf_counter()
            ranked = _run_method_queries(qids, fn, max_workers=2)
            wall_s = time.perf_counter() - t0
            cache.save()
            m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
            nq = len(qids)
            rec = {
                "backend": "AnyJev (Tacit-9B)",
                "method": name,
                **{
                    k: m[k]
                    for k in (
                        "hit1",
                        "hit3",
                        "hit5",
                        "hit10",
                        "recall5",
                        "recall10",
                        "recall20",
                        "ndcg10",
                        "mrr",
                        "mean_best_rank",
                        "full280_hit1",
                        "full280_hit5",
                        "full280_hit10",
                        "full280_mrr20",
                    )
                },
                "calls_per_query": aj.calls / nq,
                "decisions_per_query": aj.decisions / nq,
                "tokens_per_query": 0.0,
                "latency_s_per_query": aj.total_latency_s / nq,
                "wall_s_total": wall_s,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<48}] Hit@1: {m['hit1']:5.1%} (280q:{m['full280_hit1']:5.1%}) | "
                f"Hit@5: {m['hit5']:5.1%} | Hit@10: {m['hit10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | "
                f"MRR: {m['mrr']:.3f} | Calls/q: {rec['calls_per_query']:4.1f} | Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    if args.backend in ("kev", "all"):
        print("\n--- 2. Local Kev-9B (Qwen3.5-9B Pointer Head @ 127.0.0.1:8009, Direct No-Proxy) ---")
        kv = make_stark_kev(cache=cache)
        methods_kv = [
            (
                "Pointwise Noul (20 state-pairs, cookbook)",
                lambda q: rerank_pointwise(kv, queries[q], candidates[q], corpus),
            ),
            (
                "Pointwise Noul Fan-out (1 call x 20 Nouls)",
                lambda q: rerank_pointwise_fanout(kv, queries[q], candidates[q], corpus),
            ),
            (
                "One-shot Setwise Choice-20 (1 call, criteria)",
                lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="criteria"),
            ),
            (
                "One-shot Setwise Choice-20 (1 call, tagged)",
                lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="tagged"),
            ),
            (
                "Setwise Tournament (4x5 -> 8, criteria, 5 calls)",
                lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "criteria"),
            ),
            (
                "Coarse-to-Fine (Noul-20 -> Choice-5, criteria)",
                lambda q: rerank_coarse_to_fine(kv, queries[q], candidates[q], corpus, 5, "criteria", use_fanout=False),
            ),
        ]
        for name, fn in methods_kv:
            kv.reset_stats()
            t0 = time.perf_counter()
            ranked = _run_method_queries(qids, fn, max_workers=1)
            wall_s = time.perf_counter() - t0
            cache.save()
            m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
            nq = len(qids)
            rec = {
                "backend": "Kev-9B (Local)",
                "method": name,
                **{
                    k: m[k]
                    for k in (
                        "hit1",
                        "hit3",
                        "hit5",
                        "hit10",
                        "recall5",
                        "recall10",
                        "recall20",
                        "ndcg10",
                        "mrr",
                        "mean_best_rank",
                        "full280_hit1",
                        "full280_hit5",
                        "full280_hit10",
                        "full280_mrr20",
                    )
                },
                "calls_per_query": kv.calls / nq,
                "decisions_per_query": kv.decisions / nq,
                "tokens_per_query": kv.total_tokens / nq,
                "latency_s_per_query": kv.total_latency_s / nq,
                "wall_s_total": wall_s,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<48}] Hit@1: {m['hit1']:5.1%} (280q:{m['full280_hit1']:5.1%}) | "
                f"Hit@5: {m['hit5']:5.1%} | Hit@10: {m['hit10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | "
                f"MRR: {m['mrr']:.3f} | Calls/q: {rec['calls_per_query']:4.1f} | Tok/q: {rec['tokens_per_query']:6.0f} | "
                f"Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    if args.backend in ("gemini", "all"):
        print("\n--- 3. Google Gemini Decision 1 (gemini-decision-1 via proxy 127.0.0.1:10809) ---")
        gm = make_stark_gemini(cache=cache)
        methods_gm = [
            (
                "Pointwise Noul Fan-out (1 call x 20 Nouls)",
                lambda q: rerank_pointwise_fanout(gm, queries[q], candidates[q], corpus),
            ),
            (
                "One-shot Setwise Choice-20 (1 call, criteria)",
                lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="criteria"),
            ),
            (
                "One-shot Setwise Choice-20 (1 call, tagged)",
                lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="tagged"),
            ),
            (
                "Coarse-to-Fine (Fan-out Noul-20 -> Choice-10)",
                lambda q: rerank_coarse_to_fine(gm, queries[q], candidates[q], corpus, 10, "criteria", use_fanout=True),
            ),
        ]
        for name, fn in methods_gm:
            gm.reset_stats()
            t0 = time.perf_counter()
            ranked = _run_method_queries(qids, fn, max_workers=1)
            wall_s = time.perf_counter() - t0
            cache.save()
            cache.save()
            m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
            nq = len(qids)
            rec = {
                "backend": "Gemini Decision 1",
                "method": name,
                **{
                    k: m[k]
                    for k in (
                        "hit1",
                        "hit3",
                        "hit5",
                        "hit10",
                        "recall5",
                        "recall10",
                        "recall20",
                        "ndcg10",
                        "mrr",
                        "mean_best_rank",
                        "full280_hit1",
                        "full280_hit5",
                        "full280_hit10",
                        "full280_mrr20",
                    )
                },
                "calls_per_query": gm.calls / nq,
                "decisions_per_query": gm.decisions / nq,
                "tokens_per_query": gm.total_tokens / nq,
                "latency_s_per_query": gm.total_latency_s / nq,
                "wall_s_total": wall_s,
            }
            summary_records.append(rec)
            print(
                f"  [{name:<48}] Hit@1: {m['hit1']:5.1%} (280q:{m['full280_hit1']:5.1%}) | "
                f"Hit@5: {m['hit5']:5.1%} | Hit@10: {m['hit10']:5.1%} | NDCG@10: {m['ndcg10']:.3f} | "
                f"MRR: {m['mrr']:.3f} | Calls/q: {rec['calls_per_query']:4.1f} | Tok/q: {rec['tokens_per_query']:6.0f} | "
                f"Lat/q: {rec['latency_s_per_query']:.2f}s"
            )

    out_path = SCRATCH_DIR / f"stark_summary_q{len(qids)}.json"
    out_path.write_text(json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved STaRK-Prime summary to {out_path}")


if __name__ == "__main__":
    main()
