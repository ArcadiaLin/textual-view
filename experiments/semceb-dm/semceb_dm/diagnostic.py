"""Offline diagnostic analysis and paired sampling replay for SemCEB x DM experiments.

Reads frozen Stage A (DM pairwise matrix) and Stage B (state proxy) artifacts only.
Makes zero API calls. Estimators receive strictly sampled labels/proxy values and
never see unsampled labels or population ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from .estimators import bilinear_total, q_error


def clamp_half_up(estimate: float, population: int) -> int:
    if estimate <= 0:
        return 0
    return max(0, min(population, int(math.floor(estimate + 0.5))))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_stage_a(run_dir: Path, threshold: float = 0.5) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    samples = _read_jsonl(run_dir / "samples.jsonl")
    requests = _read_jsonl(run_dir / "requests.jsonl")

    n_rows = manifest["row_counts"][manifest["query"]["datasets"][0].split(" as ")[0]]
    n_cols = manifest["row_counts"][manifest["query"]["datasets"][1].split(" as ")[0]]
    population = n_rows * n_cols

    if len(samples) != population:
        raise ValueError(f"Expected {population} samples, got {len(samples)}")
    if len(requests) != population:
        raise ValueError(f"Expected {population} request records, got {len(requests)}")

    seen: set[tuple[int, int]] = set()
    p_mat = np.full((n_rows, n_cols), np.nan, dtype=float)
    left_ids: dict[int, dict] = {}
    right_ids: dict[int, dict] = {}
    pair_keys: dict[tuple[int, int], str] = {}
    models = set()
    instructions = set()

    for sample, req in zip(samples, requests):
        if "error_type" in req:
            raise ValueError(f"Stage A contains failed request: {req['error_type']}")
        i, j = int(sample["left_index"]), int(sample["right_index"])
        if not (0 <= i < n_rows and 0 <= j < n_cols):
            raise ValueError(f"Index out of bounds: {(i, j)}")
        if (i, j) in seen:
            raise ValueError(f"Duplicate pair {(i, j)}")
        seen.add((i, j))

        prob = float(sample["dm_probability"])
        if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
            raise ValueError(f"Invalid probability {prob} at {(i, j)}")
        resp_prob = float(req["response"]["answers"]["decision"]["noul"])
        if not math.isclose(prob, resp_prob, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Sample/request probability mismatch at {(i, j)}")

        p_mat[i, j] = prob
        left_ids[i] = sample["left_id"]
        right_ids[j] = sample["right_id"]
        pair_keys[(i, j)] = req["key"]
        models.add(req["request"]["model"])
        instructions.add(req["request"]["questions"]["decision"]["instructions"])

    expected_pairs = {(i, j) for i in range(n_rows) for j in range(n_cols)}
    if seen != expected_pairs:
        raise ValueError(f"Incomplete pair coverage: missing {len(expected_pairs - seen)}")
    if len(models) != 1 or len(instructions) != 1:
        raise ValueError("Inconsistent model or prompt instructions across Stage A requests")

    y_mat = (p_mat >= threshold).astype(float)
    soft_sum = float(np.sum(p_mat))
    hard_sum = float(np.sum(y_mat))

    if not math.isclose(soft_sum, float(result["soft_estimate"]), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Stage A soft sum disagrees with result.json")
    if not math.isclose(hard_sum, float(result["hard_estimate"]), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Stage A hard sum disagrees with result.json")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "population": population,
        "threshold": threshold,
        "p_mat": p_mat,
        "y_mat": y_mat,
        "soft_sum": soft_sum,
        "hard_sum": hard_sum,
        "left_ids": [left_ids[i] for i in range(n_rows)],
        "right_ids": [right_ids[j] for j in range(n_cols)],
        "pair_keys": pair_keys,
        "reference": result["reference"],
        "model": next(iter(models)),
        "instructions": next(iter(instructions)),
        "manifest": manifest,
        "result": result,
    }


def load_stage_b(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    states_records = _read_jsonl(run_dir / "states.jsonl")
    requests = _read_jsonl(run_dir / "requests.jsonl")

    for req in requests:
        if "error_type" in req:
            raise ValueError(f"Stage B contains failed request: {req['error_type']}")

    left_rows: dict[int, list[float]] = {}
    right_rows: dict[int, list[float]] = {}
    left_state_names: list[str] | None = None
    right_state_names: list[str] | None = None
    k_mat = None

    for rec in states_records:
        if "compatibility_matrix" in rec:
            k_mat = np.asarray(rec["compatibility_matrix"], dtype=float)
        elif rec.get("side") == "left":
            left_rows[int(rec["row_index"])] = [float(x) for x in rec["q"]]
            left_state_names = list(rec["states"])
        elif rec.get("side") == "right":
            right_rows[int(rec["row_index"])] = [float(x) for x in rec["q"]]
            right_state_names = list(rec["states"])

    if k_mat is None or left_state_names is None or right_state_names is None:
        raise ValueError("Incomplete Stage B states.jsonl")

    q_left = np.asarray([left_rows[i] for i in range(len(left_rows))], dtype=float)
    q_right = np.asarray([right_rows[j] for j in range(len(right_rows))], dtype=float)
    mu_mat = q_left @ k_mat @ q_right.T

    agg_total = bilinear_total(q_left, k_mat, q_right)
    explicit_total = float(np.sum(mu_mat))
    if not math.isclose(agg_total, explicit_total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Bilinear aggregate {agg_total} != explicit sum {explicit_total}")
    if not math.isclose(agg_total, float(result["soft_estimate"]), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Stage B bilinear sum disagrees with result.json")

    # Extract raw product descriptions from left choice requests in row order
    left_texts = []
    for req in requests[: len(left_rows)]:
        left_texts.append(req["request"]["input"])

    return {
        "q_left": q_left,
        "q_right": q_right,
        "left_state_names": left_state_names,
        "right_state_names": right_state_names,
        "k_mat": k_mat,
        "mu_mat": mu_mat,
        "proxy_total": agg_total,
        "explicit_total": explicit_total,
        "left_texts": left_texts,
        "manifest": manifest,
        "result": result,
    }


def summarize_live_costs(run_dirs: Iterable[Path]) -> dict:
    """Aggregate real live HTTP calls (tokens, seconds, errors) across run directories.

    Cache hits are excluded from live latency/token averages so cached replay time
    is never misreported as live inference cost.
    """
    live_by_type: dict[str, list[dict]] = {"noul": [], "choice": []}
    failed_attempts: list[dict] = []
    seen_live_keys: set[str] = set()
    duplicate_live_keys = 0

    for run_dir in run_dirs:
        req_path = run_dir / "requests.jsonl"
        if not req_path.exists():
            continue
        for rec in _read_jsonl(req_path):
            if "error_type" in rec:
                failed_attempts.append(
                    {
                        "run": run_dir.name,
                        "key": rec["key"],
                        "error_type": rec["error_type"],
                        "status": rec.get("status"),
                        "seconds": float(rec["seconds"]),
                    }
                )
            elif not rec.get("cache_hit", False):
                key = rec["key"]
                if key in seen_live_keys:
                    duplicate_live_keys += 1
                    continue
                seen_live_keys.add(key)
                qtype = rec["request"]["questions"]["decision"]["type"]
                usage = rec["response"].get("usage", rec["response"].get("usageMetadata", {})) or {}
                live_by_type.setdefault(qtype, []).append(
                    {
                        "run": run_dir.name,
                        "key": key,
                        "seconds": float(rec["seconds"]),
                        "input_tokens": int(usage.get("total_input_tokens", 0)),
                        "output_tokens": int(usage.get("total_output_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }
                )

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {"count": 0, "total_seconds": 0.0, "mean_seconds": 0.0, "total_tokens": 0, "mean_tokens": 0.0, "min_tokens": 0, "max_tokens": 0}
        secs = [r["seconds"] for r in rows]
        toks = [r["total_tokens"] for r in rows]
        return {
            "count": len(rows),
            "total_seconds": float(sum(secs)),
            "mean_seconds": float(np.mean(secs)),
            "total_tokens": int(sum(toks)),
            "mean_tokens": float(np.mean(toks)),
            "min_tokens": int(min(toks)),
            "max_tokens": int(max(toks)),
        }

    all_live = [r for rows in live_by_type.values() for r in rows]
    return {
        "all_live_success": _agg(all_live),
        "by_primitive": {k: _agg(v) for k, v in live_by_type.items()},
        "failed_attempts_count": len(failed_attempts),
        "failed_attempts_seconds": float(sum(f["seconds"] for f in failed_attempts)),
        "failed_attempts": failed_attempts,
        "duplicate_live_keys": duplicate_live_keys,
        "usd": None,
    }


def proxy_relationship_metrics(target: np.ndarray, mu: np.ndarray) -> dict:
    y = np.asarray(target, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    if y.shape != m.shape or len(y) < 2:
        raise ValueError("Target and proxy arrays must have matching length >= 2")
    resid = y - m
    var_y = float(np.var(y, ddof=1))
    var_m = float(np.var(m, ddof=1))
    var_r = float(np.var(resid, ddof=1))
    cov_ym = float(np.cov(y, m, ddof=1)[0, 1])

    var_ratio = None if var_y == 0.0 else var_r / var_y
    corr = None if (var_y == 0.0 or var_m == 0.0) else cov_ym / math.sqrt(var_y * var_m)
    oracle_lambda = None if var_m == 0.0 else cov_ym / var_m
    if oracle_lambda is not None and var_y > 0.0:
        oracle_resid = y - oracle_lambda * m
        oracle_var_ratio = float(np.var(oracle_resid, ddof=1)) / var_y
    else:
        oracle_var_ratio = None

    return {
        "target_sum": float(np.sum(y)),
        "proxy_sum": float(np.sum(m)),
        "target_mean": float(np.mean(y)),
        "proxy_mean": float(np.mean(m)),
        "target_var_ddof1": var_y,
        "proxy_var_ddof1": var_m,
        "residual_mean": float(np.mean(resid)),
        "residual_var_ddof1": var_r,
        "variance_ratio_lambda1": var_ratio,
        "cov_target_proxy": cov_ym,
        "pearson_r": corr,
        "pair_rmse": float(math.sqrt(np.mean(resid ** 2))),
        "pair_mae": float(np.mean(np.abs(resid))),
        "diagnostic_oracle_upper_bound_not_deployable": {
            "oracle_lambda": oracle_lambda,
            "oracle_variance_ratio": oracle_var_ratio,
        },
    }


# --- Strictly isolated replay estimators (never see unsampled labels or C_DM) ---

def estimate_sample_hard(sampled_y: np.ndarray, population: int) -> float:
    return float(population * np.mean(sampled_y))


def estimate_sample_soft(sampled_p: np.ndarray, population: int) -> float:
    return float(population * np.mean(sampled_p))


def estimate_proxy_only(proxy_total: float) -> float:
    return float(proxy_total)


def estimate_residual(
    sampled_y: np.ndarray,
    sampled_mu: np.ndarray,
    proxy_total: float,
    population: int,
    lam: float = 1.0,
) -> float:
    return float(lam * proxy_total + population * np.mean(sampled_y - lam * sampled_mu))


def _summarize_estimates(estimates: np.ndarray, truth: float, population: int) -> dict:
    raw = np.asarray(estimates, dtype=float)
    clamped_int = np.asarray([clamp_half_up(x, population) for x in raw], dtype=float)
    err = raw - truth
    q_raw = np.asarray([q_error(max(0.0, float(x)), truth) for x in raw], dtype=float)
    q_int = np.asarray([q_error(float(x), truth) for x in clamped_int], dtype=float)

    return {
        "truth": float(truth),
        "mean_raw": float(np.mean(raw)),
        "bias_raw": float(np.mean(err)),
        "var_raw": float(np.var(raw, ddof=1)) if len(raw) > 1 else 0.0,
        "rmse_raw": float(math.sqrt(np.mean(err ** 2))),
        "mae_raw": float(np.mean(np.abs(err))),
        "negative_rate": float(np.mean(raw < 0.0)),
        "zero_or_neg_rate": float(np.mean(raw <= 0.0)),
        "q_error_raw_median": float(np.median(q_raw)),
        "q_error_raw_p90": float(np.quantile(q_raw, 0.90)),
        "q_error_raw_p95": float(np.quantile(q_raw, 0.95)),
        "q_error_raw_mean": float(np.mean(q_raw)),
        "q_error_int_median": float(np.median(q_int)),
        "q_error_int_p90": float(np.quantile(q_int, 0.90)),
        "q_error_int_p95": float(np.quantile(q_int, 0.95)),
        "q_error_int_mean": float(np.mean(q_int)),
        "rmse_int": float(math.sqrt(np.mean((clamped_int - truth) ** 2))),
    }


def run_paired_replay(
    y_mat: np.ndarray,
    p_mat: np.ndarray,
    mu_mat: np.ndarray,
    proxy_total: float,
    official_cardinality: float,
    sample_sizes: list[int],
    seeds: Iterable[int],
    oracle_lambda: float | None = None,
) -> dict:
    y_flat = np.asarray(y_mat, dtype=float).ravel()
    p_flat = np.asarray(p_mat, dtype=float).ravel()
    mu_flat = np.asarray(mu_mat, dtype=float).ravel()
    population = len(y_flat)
    c_dm = float(np.sum(y_flat))
    seed_list = list(seeds)

    # Pre-generate one permutation per seed so all methods/sample sizes share prefixes.
    perms = [random.Random(s).sample(range(population), population) for s in seed_list]

    methods = ["sample_hard", "proxy_only", "residual_lambda1", "sample_soft"]
    if oracle_lambda is not None:
        methods.append("diagnostic_oracle_lambda")

    by_sample_size: dict[int, dict] = {}
    for b in sample_sizes:
        if not (1 <= b <= population):
            raise ValueError(f"Sample size {b} out of bounds for population {population}")
        raw_by_method: dict[str, list[float]] = {m: [] for m in methods}
        all_zero_sample_count = 0

        for perm in perms:
            idx = perm[:b]
            sampled_y = y_flat[idx].copy()
            sampled_p = p_flat[idx].copy()
            sampled_mu = mu_flat[idx].copy()
            if np.all(sampled_y == 0.0):
                all_zero_sample_count += 1

            raw_by_method["sample_hard"].append(estimate_sample_hard(sampled_y, population))
            raw_by_method["sample_soft"].append(estimate_sample_soft(sampled_p, population))
            raw_by_method["proxy_only"].append(estimate_proxy_only(proxy_total))
            raw_by_method["residual_lambda1"].append(
                estimate_residual(sampled_y, sampled_mu, proxy_total, population, lam=1.0)
            )
            if oracle_lambda is not None:
                raw_by_method["diagnostic_oracle_lambda"].append(
                    estimate_residual(sampled_y, sampled_mu, proxy_total, population, lam=oracle_lambda)
                )

        entry = {
            "sample_size": b,
            "all_zero_hard_sample_rate": all_zero_sample_count / len(seed_list),
            "vs_c_dm": {},
            "vs_official": {},
        }
        for m, vals in raw_by_method.items():
            arr = np.asarray(vals, dtype=float)
            entry["vs_c_dm"][m] = _summarize_estimates(arr, c_dm, population)
            entry["vs_official"][m] = _summarize_estimates(arr, official_cardinality, population)
        by_sample_size[b] = entry

    return {
        "population": population,
        "c_dm": c_dm,
        "official_cardinality": float(official_cardinality),
        "num_seeds": len(seed_list),
        "sample_sizes": sample_sizes,
        "by_sample_size": {str(k): v for k, v in by_sample_size.items()},
    }


def run_budget_replay(
    y_mat: np.ndarray,
    mu_mat: np.ndarray,
    proxy_total: float,
    official_cardinality: float,
    build_requests: int,
    budgets: list[int],
    seeds: Iterable[int],
    mean_pair_tokens: float,
    mean_pair_seconds: float,
    build_total_tokens: int,
    build_total_seconds: float,
) -> list[dict]:
    """Compare methods under equal logical request budgets (Section 7)."""
    y_flat = np.asarray(y_mat, dtype=float).ravel()
    mu_flat = np.asarray(mu_mat, dtype=float).ravel()
    population = len(y_flat)
    c_dm = float(np.sum(y_flat))
    seed_list = list(seeds)
    perms = [random.Random(s).sample(range(population), population) for s in seed_list]

    rows = []
    for total_req in budgets:
        b_sample = min(total_req, population)
        b_res = min(max(total_req - build_requests, 0), population)

        sample_ests = []
        res_ests = []
        for perm in perms:
            if b_sample > 0:
                sample_ests.append(estimate_sample_hard(y_flat[perm[:b_sample]].copy(), population))
            if b_res == 0:
                res_ests.append(estimate_proxy_only(proxy_total))
            else:
                idx = perm[:b_res]
                res_ests.append(
                    estimate_residual(y_flat[idx].copy(), mu_flat[idx].copy(), proxy_total, population, lam=1.0)
                )

        rows.append(
            {
                "logical_request_budget": total_req,
                "sample_hard_pairs": b_sample,
                "residual_build_requests": build_requests,
                "residual_pairs": b_res,
                "sample_hard_vs_c_dm": _summarize_estimates(np.asarray(sample_ests), c_dm, population),
                "residual_lambda1_vs_c_dm": _summarize_estimates(np.asarray(res_ests), c_dm, population),
                "sample_hard_vs_official": _summarize_estimates(np.asarray(sample_ests), official_cardinality, population),
                "residual_lambda1_vs_official": _summarize_estimates(np.asarray(res_ests), official_cardinality, population),
                "replayed_cost_estimate": {
                    "note": "Replayed estimate from observed mean live tokens/seconds; not live end-to-end wall time.",
                    "sample_hard_est_tokens": b_sample * mean_pair_tokens,
                    "sample_hard_est_seconds": b_sample * mean_pair_seconds,
                    "residual_est_tokens": build_total_tokens + b_res * mean_pair_tokens,
                    "residual_est_seconds": build_total_seconds + b_res * mean_pair_seconds,
                },
            }
        )
    return rows


def extract_failure_cases(stage_a: dict, stage_b: dict, top_k: int = 12) -> dict:
    p_mat = stage_a["p_mat"]
    y_mat = stage_a["y_mat"]
    mu_mat = stage_b["mu_mat"]
    q_left = stage_b["q_left"]
    q_right = stage_b["q_right"]
    left_ids = stage_a["left_ids"]
    right_ids = stage_a["right_ids"]
    left_texts = stage_b["left_texts"]

    def _snippet(raw_json_str: str, limit: int = 160) -> str:
        try:
            obj = json.loads(raw_json_str)
            desc = obj.get("description_json", raw_json_str)
        except Exception:
            desc = raw_json_str
        desc = " ".join(str(desc).split())
        return desc if len(desc) <= limit else desc[: limit - 3] + "..."

    snippets = [_snippet(t) for t in left_texts]
    all_pairs = []
    for i in range(stage_a["n_rows"]):
        for j in range(stage_a["n_cols"]):
            all_pairs.append(
                {
                    "left_index": i,
                    "right_index": j,
                    "left_asin": left_ids[i]["parent_asin"],
                    "right_asin": right_ids[j]["parent_asin"],
                    "dm_prob": float(p_mat[i, j]),
                    "dm_hard": int(y_mat[i, j]),
                    "proxy_mu": float(mu_mat[i, j]),
                    "residual_hard": float(y_mat[i, j] - mu_mat[i, j]),
                    "abs_residual_hard": float(abs(y_mat[i, j] - mu_mat[i, j])),
                    "residual_soft": float(p_mat[i, j] - mu_mat[i, j]),
                    "abs_residual_soft": float(abs(p_mat[i, j] - mu_mat[i, j])),
                    "q_left": [round(float(x), 4) for x in q_left[i]],
                    "q_right": [round(float(x), 4) for x in q_right[j]],
                    "left_snippet": snippets[i],
                    "right_snippet": snippets[j],
                }
            )

    top_hard_resid = sorted(all_pairs, key=lambda x: (-x["abs_residual_hard"], x["left_index"], x["right_index"]))[:top_k]
    top_soft_resid = sorted(all_pairs, key=lambda x: (-x["abs_residual_soft"], x["left_index"], x["right_index"]))[:top_k]
    positive_dm_pairs = sorted([x for x in all_pairs if x["dm_hard"] == 1], key=lambda x: (-x["dm_prob"], x["left_index"], x["right_index"]))

    products = []
    for i in range(stage_a["n_rows"]):
        products.append(
            {
                "row_index": i,
                "parent_asin": left_ids[i]["parent_asin"],
                "q_left": {k: round(float(v), 4) for k, v in zip(stage_b["left_state_names"], q_left[i])},
                "q_right": {k: round(float(v), 4) for k, v in zip(stage_b["right_state_names"], q_right[i])},
                "dm_row_hard_sum": int(np.sum(y_mat[i, :])),
                "dm_col_hard_sum": int(np.sum(y_mat[:, i])),
                "dm_row_soft_sum": round(float(np.sum(p_mat[i, :])), 4),
                "dm_col_soft_sum": round(float(np.sum(p_mat[:, i])), 4),
                "snippet": snippets[i],
            }
        )

    return {
        "products": products,
        "positive_dm_pairs": positive_dm_pairs,
        "top_abs_residual_hard": top_hard_resid,
        "top_abs_residual_soft": top_soft_resid,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Stage C offline diagnostic analysis on frozen Stage A/B runs.")
    parser.add_argument("--stage-a", type=Path, required=True, help="Completed Stage A run directory")
    parser.add_argument("--stage-b", type=Path, required=True, help="Completed Stage B run directory")
    parser.add_argument("--stage-a-cost-dirs", type=Path, nargs="+", required=True, help="All run dirs contributing live Stage A calls")
    parser.add_argument("--stage-b-cost-dirs", type=Path, nargs="+", required=True, help="All run dirs contributing live Stage B calls")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for diagnostic artifacts")
    parser.add_argument("--seeds", type=int, default=1000, help="Number of paired seeds (0..seeds-1)")
    args = parser.parse_args()

    stage_a = load_stage_a(args.stage_a)
    stage_b = load_stage_b(args.stage_b)

    cost_a = summarize_live_costs(args.stage_a_cost_dirs)
    cost_b = summarize_live_costs(args.stage_b_cost_dirs)

    rel_hard = proxy_relationship_metrics(stage_a["y_mat"], stage_b["mu_mat"])
    rel_soft = proxy_relationship_metrics(stage_a["p_mat"], stage_b["mu_mat"])

    oracle_lam = rel_hard["diagnostic_oracle_upper_bound_not_deployable"]["oracle_lambda"]
    official_card = float(stage_a["reference"]["cardinality"])

    replay = run_paired_replay(
        y_mat=stage_a["y_mat"],
        p_mat=stage_a["p_mat"],
        mu_mat=stage_b["mu_mat"],
        proxy_total=stage_b["proxy_total"],
        official_cardinality=official_card,
        sample_sizes=[2, 5, 10, 20, 40, 60, 80, 100],
        seeds=range(args.seeds),
        oracle_lambda=oracle_lam,
    )

    # Census sanity check (Section 6)
    census_sample = replay["by_sample_size"]["100"]["vs_c_dm"]["sample_hard"]
    census_resid = replay["by_sample_size"]["100"]["vs_c_dm"]["residual_lambda1"]
    if census_sample["rmse_raw"] > 1e-12 or census_resid["rmse_raw"] > 1e-12:
        raise RuntimeError("Census endpoint check failed: B=N must return exact C_DM")

    budget_rows = run_budget_replay(
        y_mat=stage_a["y_mat"],
        mu_mat=stage_b["mu_mat"],
        proxy_total=stage_b["proxy_total"],
        official_cardinality=official_card,
        build_requests=32,
        budgets=[32, 34, 37, 42, 52, 72, 92, 100, 112, 132],
        seeds=range(args.seeds),
        mean_pair_tokens=cost_a["all_live_success"]["mean_tokens"],
        mean_pair_seconds=cost_a["all_live_success"]["mean_seconds"],
        build_total_tokens=cost_b["all_live_success"]["total_tokens"],
        build_total_seconds=cost_b["all_live_success"]["total_seconds"],
    )

    cases = extract_failure_cases(stage_a, stage_b, top_k=12)

    summary = {
        "stage_a_run": str(args.stage_a),
        "stage_b_run": str(args.stage_b),
        "totals": {
            "population": stage_a["population"],
            "official_reference": stage_a["reference"],
            "dm_soft_sum": stage_a["soft_sum"],
            "dm_soft_int": clamp_half_up(stage_a["soft_sum"], stage_a["population"]),
            "dm_hard_sum_c_dm": stage_a["hard_sum"],
            "proxy_total_m": stage_b["proxy_total"],
            "proxy_total_int": clamp_half_up(stage_b["proxy_total"], stage_a["population"]),
            "proxy_explicit_sum_diff": abs(stage_b["proxy_total"] - stage_b["explicit_total"]),
            "q_error_dm_hard_vs_official": q_error(stage_a["hard_sum"], official_card),
            "q_error_dm_soft_vs_official": q_error(stage_a["soft_sum"], official_card),
            "q_error_proxy_vs_official": q_error(stage_b["proxy_total"], official_card),
            "q_error_proxy_vs_c_dm": q_error(stage_b["proxy_total"], stage_a["hard_sum"]),
        },
        "state_summary": {
            "left_states": stage_b["left_state_names"],
            "right_states": stage_b["right_state_names"],
            "left_mean_q": [round(float(x), 4) for x in np.mean(stage_b["q_left"], axis=0)],
            "right_mean_q": [round(float(x), 4) for x in np.mean(stage_b["q_right"], axis=0)],
            "compatibility_matrix": [[round(float(v), 4) for v in row] for row in stage_b["k_mat"]],
        },
        "proxy_vs_dm_hard": rel_hard,
        "proxy_vs_dm_soft": rel_soft,
        "costs": {
            "stage_a_exhaustive_diagnostic": cost_a,
            "stage_b_proxy_construction": cost_b,
        },
        "paired_replay": replay,
        "budget_replay": budget_rows,
        "cases": cases,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    np.savez(
        args.output / "matrices.npz",
        p_mat=stage_a["p_mat"],
        y_mat=stage_a["y_mat"],
        mu_mat=stage_b["mu_mat"],
        resid_hard=stage_a["y_mat"] - stage_b["mu_mat"],
        q_left=stage_b["q_left"],
        q_right=stage_b["q_right"],
        k_mat=stage_b["k_mat"],
        asins=np.asarray([x["parent_asin"] for x in stage_a["left_ids"]]),
    )
    print(f"Wrote diagnostic artifacts to {args.output}")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print("proxy_vs_dm_hard:", json.dumps(summary["proxy_vs_dm_hard"], ensure_ascii=False, indent=2))
    print("proxy_vs_dm_soft:", json.dumps(summary["proxy_vs_dm_soft"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
