"""Tests for Stage C offline diagnostic and replay invariants."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from semceb_dm.diagnostic import (
    clamp_half_up,
    estimate_proxy_only,
    estimate_residual,
    estimate_sample_hard,
    load_stage_a,
    proxy_relationship_metrics,
    run_budget_replay,
    run_paired_replay,
)


def test_census_endpoint_returns_exact_c_dm():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, size=(4, 4))
    y = (p >= 0.5).astype(float)
    mu = rng.uniform(0, 1, size=(4, 4))
    m = float(np.sum(mu))
    c_dm = float(np.sum(y))

    replay = run_paired_replay(
        y_mat=y,
        p_mat=p,
        mu_mat=mu,
        proxy_total=m,
        official_cardinality=10.0,
        sample_sizes=[2, 8, 16],
        seeds=range(20),
    )
    census = replay["by_sample_size"]["16"]["vs_c_dm"]
    assert census["sample_hard"]["mean_raw"] == pytest.approx(c_dm)
    assert census["sample_hard"]["rmse_raw"] == pytest.approx(0.0, abs=1e-12)
    assert census["residual_lambda1"]["mean_raw"] == pytest.approx(c_dm)
    assert census["residual_lambda1"]["rmse_raw"] == pytest.approx(0.0, abs=1e-12)


def test_replay_reproducibility_and_prefix_consistency():
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    p = y * 0.8 + 0.1
    mu = np.array([[0.7, 0.2], [0.3, 0.6], [0.5, 0.1]])
    m = float(np.sum(mu))

    r1 = run_paired_replay(y, p, mu, m, 3.0, [2, 4, 6], range(50))
    r2 = run_paired_replay(y, p, mu, m, 3.0, [2, 4, 6], range(50))
    assert r1 == r2


def test_estimators_cannot_observe_unsampled_labels():
    # Perturbing unsampled entries must not change estimator output on a fixed sample slice.
    y1 = np.array([1.0, 0.0, 0.0, 0.0])
    y2 = np.array([1.0, 0.0, 1.0, 1.0])
    mu = np.array([0.6, 0.2, 0.3, 0.4])
    m = float(np.sum(mu))
    idx = [0, 1]

    assert estimate_sample_hard(y1[idx], 4) == estimate_sample_hard(y2[idx], 4)
    assert estimate_residual(y1[idx], mu[idx], m, 4, lam=1.0) == estimate_residual(y2[idx], mu[idx], m, 4, lam=1.0)
    assert estimate_proxy_only(m) == pytest.approx(m)


def test_zero_variance_denominator_handled_cleanly():
    y_const = np.zeros((3, 3))
    mu = np.linspace(0.1, 0.9, 9).reshape(3, 3)
    metrics = proxy_relationship_metrics(y_const, mu)
    assert metrics["target_var_ddof1"] == 0.0
    assert metrics["variance_ratio_lambda1"] is None
    assert metrics["pearson_r"] is None


def test_stage_a_coverage_validation(tmp_path: Path):
    # Missing one pair out of 4 must raise ValueError.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "query": {"datasets": ["t as p1", "t as p2"]},
                "row_counts": {"t": 2},
            }
        )
    )
    (tmp_path / "result.json").write_text(json.dumps({"soft_estimate": 1.0, "hard_estimate": 1.0, "reference": {"cardinality": 2}}))
    samples = [
        {"left_index": 0, "right_index": 0, "left_id": {"parent_asin": "a"}, "right_id": {"parent_asin": "a"}, "dm_probability": 0.8},
        {"left_index": 0, "right_index": 1, "left_id": {"parent_asin": "a"}, "right_id": {"parent_asin": "b"}, "dm_probability": 0.2},
        {"left_index": 1, "right_index": 0, "left_id": {"parent_asin": "b"}, "right_id": {"parent_asin": "a"}, "dm_probability": 0.0},
    ]
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    (tmp_path / "requests.jsonl").write_text("\n".join(json.dumps({"key": str(i)}) for i in range(3)) + "\n")
    with pytest.raises(ValueError, match="Expected 4 samples"):
        load_stage_a(tmp_path)


def test_budget_replay_and_clamp_half_up():
    assert clamp_half_up(-3.2, 100) == 0
    assert clamp_half_up(2.5, 100) == 3
    assert clamp_half_up(150.0, 100) == 100

    y = np.array([[1.0, 0.0], [0.0, 0.0]])
    mu = np.array([[0.4, 0.4], [0.4, 0.4]])
    rows = run_budget_replay(y, mu, float(np.sum(mu)), 2.0, build_requests=2, budgets=[2, 4, 6], seeds=range(10), mean_pair_tokens=100.0, mean_pair_seconds=1.0, build_total_tokens=150, build_total_seconds=2.0)
    assert rows[0]["residual_pairs"] == 0
    assert rows[1]["sample_hard_pairs"] == 4
    assert rows[1]["sample_hard_vs_c_dm"]["rmse_raw"] == pytest.approx(0.0, abs=1e-12)
    assert rows[2]["residual_pairs"] == 4
    assert rows[2]["residual_lambda1_vs_c_dm"]["rmse_raw"] == pytest.approx(0.0, abs=1e-12)
