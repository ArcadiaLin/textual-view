"""Evaluate SemStats estimates_single.json against SemCEB sf1000 ground-truth cardinalities.

Reports UNSIGNED q-error clamped at 1.0:
    qerr(est, true) = max(max(1.0, est), max(1.0, true)) / min(max(1.0, est), max(1.0, true))
Note: SemCEB's native reporting uses signed q-error (negative for underestimation).
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
        rows.append({
            "case_id": q["case_id"],
            "type": q["type"],
            "category": q["category"],
            "true": true_card,
            "estimate": est_card,
            "qerr": qe,
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
    summary = {
        "qerr_definition": "unsigned, clamped at 1.0: max(max(1, est), max(1, true)) / min(max(1, est), max(1, true))",
        "all_queries": summarize(all_qerr),
        "nonzero_truth_only": summarize(nonzero_qerr) if nonzero_qerr else None,
        "sampling": samp_stats,
        "zero_estimate_on_nonzero_truth": sum(1 for r in rows if r["true"] > 0 and r["estimate"] == 0.0),
        "fallback_count": sum(1 for r in rows if r["fallback"]),
    }

    lines = [
        f"Estimates: {est_path}",
        f"Queries:   {q_path}",
        "Metric:    UNSIGNED q-error (est & true clamped to >= 1.0; SemCEB native reports signed q-error)",
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
    lines.append("Per-query breakdown (sorted by q-error descending):")
    for r in sorted(rows, key=lambda x: -x["qerr"]):
        lines.append(
            f"  {r['case_id']}  q={r['qerr']:7.2f}  true={r['true']:6.0f}  est={r['estimate']:7.1f}  "
            f"cand={r['cand_size']:5d}  draws={r['sample_size']:4d}  uniq={r['n_unique']:4d}  "
            f"fb={int(bool(r['fallback']))}  {r['query'][:65]}"
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
