"""Evaluate SemStats estimates_single.json against SemCEB sf1000 ground-truth cardinalities.

Reports both:
1. UNSIGNED q-error clamped at 1.0 (SemStats / standard CE convention):
       qerr(est, true) = max(max(1.0, est), max(1.0, true)) / min(max(1.0, est), max(1.0, true))
2. SemCEB-native SIGNED q-error on rounded integer estimates
   (matches references/repos/SemCEB/src/semceb/benchmark/benchmark.py:_calculate_q_error):
       negative for underestimation, 1.0 for exact match, positive for overestimation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

QS = (0.50, 0.90, 0.95, 0.99)


def qerr(est: float, true: float) -> float:
    e = max(1.0, float(est))
    t = max(1.0, float(true))
    return max(e, t) / min(e, t)


def semceb_signed_qerr(est: float, true: float) -> tuple[int, float]:
    """Exact match to SemCEB benchmark.py:_calculate_q_error on rounded int estimates."""
    est_int = max(0, int(round(float(est))))
    true_int = max(0, int(round(float(true))))
    safe_est = max(1, est_int)
    safe_true = max(1, true_int)
    if est_int == true_int:
        return est_int, 1.0
    if est_int < true_int:
        return est_int, -(safe_true / safe_est)
    return est_int, safe_est / safe_true


def summarize(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    q = np.quantile(v, QS)
    return {
        "n": int(len(v)),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "geo": float(np.exp(np.log(v).mean())),
    }


def summarize_semceb_signed(rows: list[dict]) -> dict:
    sv = np.asarray([float(r["semceb_signed_qerr"]) for r in rows], dtype=float)
    av = np.abs(sv)
    q = np.quantile(av, QS)
    return {
        "n": int(len(sv)),
        "signed_mean": float(sv.mean()),
        "signed_min": float(sv.min()),
        "signed_max": float(sv.max()),
        "underestimate_count": int(sum(1 for r in rows if r["estimate_int"] < int(round(r["true"])))),
        "exact_count": int(sum(1 for r in rows if r["estimate_int"] == int(round(r["true"])))),
        "overestimate_count": int(sum(1 for r in rows if r["estimate_int"] > int(round(r["true"])))),
        "abs_p50": float(q[0]),
        "abs_p90": float(q[1]),
        "abs_p95": float(q[2]),
        "abs_p99": float(q[3]),
        "abs_max": float(av.max()),
        "abs_mean": float(av.mean()),
        "abs_geo": float(np.exp(np.log(av).mean())),
    }


def summarize_sampling(rows: list[dict]) -> dict:
    draws = np.asarray([int(r["sample_size"]) for r in rows], dtype=float)
    uniq = np.asarray([int(r["n_unique"]) for r in rows], dtype=float)
    return {
        "total_draws": int(draws.sum()),
        "mean_draws": float(draws.mean()),
        "median_draws": float(np.median(draws)),
        "max_draws": int(draws.max()),
        "total_unique_judgements": int(uniq.sum()),
        "mean_unique_judgements": float(uniq.mean()),
        "median_unique_judgements": float(np.median(uniq)),
        "max_unique_judgements": int(uniq.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimates", required=True, help="Path to estimates_single.json")
    ap.add_argument("--queries", required=True, help="Path to data/<table>/queries/single.json")
    ap.add_argument("--out", default="", help="Optional output directory for metrics_semceb.json & report_semceb.txt")
    args = ap.parse_args()

    est_path = Path(args.estimates)
    q_path = Path(args.queries)
    est_payload = json.loads(est_path.read_text(encoding="utf-8"))
    est_rows = est_payload["rows"]
    queries = json.loads(q_path.read_text(encoding="utf-8"))
    if len(est_rows) != len(queries):
        raise ValueError(f"Row count mismatch: {len(est_rows)} estimates vs {len(queries)} queries")

    rows = []
    for r, q in zip(est_rows, queries):
        if r["query"] != q["query"]:
            raise ValueError(f"Query order mismatch at {q['case_id']}")
        true_card = float(q["ground_truth_cardinality"])
        est_card = float(r["estimate"])
        qe = qerr(est_card, true_card)
        est_int, sq = semceb_signed_qerr(est_card, true_card)
        rows.append({
            "case_id": q["case_id"],
            "type": q["type"],
            "category": q["category"],
            "true": true_card,
            "estimate": est_card,
            "estimate_int": est_int,
            "qerr": qe,
            "semceb_signed_qerr": sq,
            "cand_size": r.get("cand_size"),
            "sample_size": r.get("sample_size"),
            "n_unique": r.get("n_unique"),
            "n_strata": r.get("n_strata"),
            "fallback": r.get("fallback"),
            "nodes": r.get("nodes"),
            "query": q["query"],
        })

    all_qerr = [r["qerr"] for r in rows]
    nonzero_qerr = [r["qerr"] for r in rows if r["true"] > 0]
    samp_stats = summarize_sampling(rows)
    signed_stats = summarize_semceb_signed(rows)
    summary = {
        "qerr_definition": "unsigned, clamped at 1.0: max(max(1, est), max(1, true)) / min(max(1, est), max(1, true))",
        "semceb_signed_qerr_definition": "SemCEB native signed q-error on rounded int estimate (benchmark.py:_calculate_q_error); negative for underestimation",
        "all_queries": summarize(all_qerr),
        "nonzero_truth_only": summarize(nonzero_qerr) if nonzero_qerr else None,
        "semceb_signed": signed_stats,
        "sampling": samp_stats,
        "zero_estimate_on_nonzero_truth": sum(1 for r in rows if r["true"] > 0 and r["estimate"] == 0.0),
        "fallback_count": sum(1 for r in rows if r["fallback"]),
    }

    lines = [
        f"Estimates: {est_path}",
        f"Queries:   {q_path}",
        "Metrics:   (1) UNSIGNED q-error (float, clamped >= 1.0)  (2) SemCEB SIGNED q-error (rounded int, negative=under)",
        "",
        f"{'subset':<22s}{'n':>4s}{'p50':>8s}{'p90':>8s}{'p95':>8s}{'p99':>8s}{'max':>9s}{'mean':>8s}{'geo':>8s}",
        "-" * 79,
    ]
    for label, key in (("All queries", "all_queries"), ("True > 0 only", "nonzero_truth_only")):
        s = summary[key]
        if s:
            lines.append(
                f"{label:<22s}{s['n']:>4d}{s['p50']:>8.3f}{s['p90']:>8.3f}{s['p95']:>8.3f}"
                f"{s['p99']:>8.3f}{s['max']:>9.3f}{s['mean']:>8.3f}{s['geo']:>8.3f}"
            )
    lines.append("")
    lines.append(
        f"SemCEB signed q-error (int est): mean={signed_stats['signed_mean']:.3f} "
        f"min={signed_stats['signed_min']:.3f} max={signed_stats['signed_max']:.3f} | "
        f"under={signed_stats['underestimate_count']} exact={signed_stats['exact_count']} "
        f"over={signed_stats['overestimate_count']} | "
        f"|q| p50={signed_stats['abs_p50']:.3f} p90={signed_stats['abs_p90']:.3f} geo={signed_stats['abs_geo']:.3f}"
    )
    lines.append(
        f"sampling_draws: total={samp_stats['total_draws']} mean={samp_stats['mean_draws']:.1f} "
        f"median={samp_stats['median_draws']:.1f} max={samp_stats['max_draws']}"
    )
    lines.append(
        f"unique_judgements: total={samp_stats['total_unique_judgements']} mean={samp_stats['mean_unique_judgements']:.1f} "
        f"median={samp_stats['median_unique_judgements']:.1f} max={samp_stats['max_unique_judgements']}"
    )
    lines.append(
        f"zero_estimate_on_nonzero_truth={summary['zero_estimate_on_nonzero_truth']}  "
        f"fallback_count={summary['fallback_count']}"
    )
    lines.append("")
    lines.append("Per-query breakdown (sorted by unsigned q-error descending):")
    for r in sorted(rows, key=lambda x: -x["qerr"]):
        lines.append(
            f"  {r['case_id']}  q={r['qerr']:7.2f}  sq={r['semceb_signed_qerr']:+8.2f}  "
            f"true={r['true']:6.0f}  est={r['estimate']:7.1f}  "
            f"cand={r['cand_size']:5d}  draws={r['sample_size']:4d}  uniq={r['n_unique']:4d}  "
            f"fb={int(bool(r['fallback']))}  {r['query'][:55]}"
        )

    report_text = "\n".join(lines)
    print(report_text)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics_semceb.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "report_semceb.txt").write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
