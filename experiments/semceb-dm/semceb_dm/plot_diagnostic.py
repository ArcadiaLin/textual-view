"""Generate Stage C diagnostic figures from diagnostic_summary.json and matrices.npz.

Run with: uv run --with matplotlib --with numpy python semceb_dm/plot_diagnostic.py --input <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_matrices(npz_path: Path, out_path: Path) -> None:
    data = np.load(npz_path)
    p_mat = data["p_mat"]
    y_mat = data["y_mat"]
    mu_mat = data["mu_mat"]
    resid = data["resid_hard"]
    asins = [str(a) for a in data["asins"]]
    labels = [f"{i}:{a[-4:]}" for i, a in enumerate(asins)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    specs = [
        (axes[0, 0], p_mat, f"DM Raw Probability p_e (sum={p_mat.sum():.2f})", "YlGnBu", 0.0, 1.0),
        (axes[0, 1], y_mat, f"DM Hard Label Y^DM = 1[p_e>=0.5] (C_DM={int(y_mat.sum())})", "Blues", 0.0, 1.0),
        (axes[1, 0], mu_mat, f"Bilinear State Proxy mu_e (M={mu_mat.sum():.2f})", "YlOrRd", 0.0, 1.0),
        (axes[1, 1], resid, f"Hard Residual Y^DM - mu_e (mean={resid.mean():.3f})", "RdBu_r", -1.0, 1.0),
    ]

    for ax, mat, title, cmap, vmin, vmax in specs:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Right product (p2, tool)")
        ax.set_ylabel("Left product (p1, material)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = f"{int(val)}" if title.startswith("DM Hard") else f"{val:.2f}"
                dark = (abs(val) > 0.55) if vmin < 0 else (val > 0.55)
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Q63 sf10 (10x10 ordered pairs, incl. diagonal): DM vs State Proxy Matrices", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sampling_curves(summary: dict, out_path: Path) -> None:
    replay = summary["paired_replay"]
    bs = replay["sample_sizes"]
    by_b = replay["by_sample_size"]

    def series(target_key: str, method: str, metric: str) -> list[float]:
        return [by_b[str(b)][target_key][method][metric] for b in bs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Panel 1: RMSE vs C_DM (raw)
    ax = axes[0]
    ax.plot(bs, series("vs_c_dm", "sample_hard", "rmse_raw"), "o-", label="Uniform Sample Hard (lambda=0)", linewidth=2)
    ax.plot(bs, series("vs_c_dm", "residual_lambda1", "rmse_raw"), "s-", label="Residual Correction (lambda=1)", linewidth=2)
    ax.plot(bs, series("vs_c_dm", "proxy_only", "rmse_raw"), "--", label="Pure Proxy (M=39.04)", linewidth=1.8)
    ax.plot(bs, series("vs_c_dm", "diagnostic_oracle_lambda", "rmse_raw"), ":", label="Oracle lambda* (diag upper bound)", linewidth=1.8)
    ax.set_title("RMSE vs C_DM (5) at Fixed Sample Size B")
    ax.set_xlabel("Main sample size B (pairs)")
    ax.set_ylabel("Raw RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Median Clamped Int Q-error vs C_DM
    ax = axes[1]
    ax.plot(bs, series("vs_c_dm", "sample_hard", "q_error_int_median"), "o-", label="Uniform Sample Hard (lambda=0)", linewidth=2)
    ax.plot(bs, series("vs_c_dm", "residual_lambda1", "q_error_int_median"), "s-", label="Residual Correction (lambda=1)", linewidth=2)
    ax.plot(bs, series("vs_c_dm", "proxy_only", "q_error_int_median"), "--", label="Pure Proxy (M=39.04)", linewidth=1.8)
    ax.plot(bs, series("vs_c_dm", "sample_soft", "q_error_int_median"), "^-.", label="Sample Soft (targets sum p_e=17.66)", linewidth=1.5)
    ax.set_title("Median Clamped Q-error vs C_DM (5)")
    ax.set_xlabel("Main sample size B (pairs)")
    ax.set_ylabel("Median Q-error (int clamped)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: Median Clamped Int Q-error vs Official Reference (52)
    ax = axes[2]
    ax.plot(bs, series("vs_official", "sample_hard", "q_error_int_median"), "o-", label="Uniform Sample Hard (lambda=0)", linewidth=2)
    ax.plot(bs, series("vs_official", "residual_lambda1", "q_error_int_median"), "s-", label="Residual Correction (lambda=1)", linewidth=2)
    ax.plot(bs, series("vs_official", "proxy_only", "q_error_int_median"), "--", label="Pure Proxy (M=39.04)", linewidth=1.8)
    ax.plot(bs, series("vs_official", "sample_soft", "q_error_int_median"), "^-.", label="Sample Soft (targets sum p_e=17.66)", linewidth=1.5)
    ax.set_title("Median Clamped Q-error vs Official Ref (52)\n(includes DM vs gpt-5.4-nano bias)")
    ax.set_xlabel("Main sample size B (pairs)")
    ax.set_ylabel("Median Q-error (int clamped)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Stage C Offline Paired Replay (1,000 seeds, Q63 sf10)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_budget_curves(summary: dict, out_path: Path) -> None:
    rows = summary["budget_replay"]
    req_budgets = [r["logical_request_budget"] for r in rows]
    sample_rmse = [r["sample_hard_vs_c_dm"]["rmse_raw"] for r in rows]
    res_rmse = [r["residual_lambda1_vs_c_dm"]["rmse_raw"] for r in rows]

    sample_tok = [r["replayed_cost_estimate"]["sample_hard_est_tokens"] for r in rows]
    res_tok = [r["replayed_cost_estimate"]["residual_est_tokens"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    ax.plot(req_budgets, sample_rmse, "o-", label="Uniform Sample Hard (all R for pairs)", linewidth=2)
    ax.plot(req_budgets, res_rmse, "s-", label="Residual lambda=1 (32 build + R-32 pairs)", linewidth=2)
    ax.axvline(100, color="gray", linestyle=":", label="Full census N=100 pairs")
    ax.set_title("Equal Logical Request Budget vs RMSE (Target: C_DM=5)")
    ax.set_xlabel("Total logical requests R")
    ax.set_ylabel("Raw RMSE vs C_DM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(sample_tok, sample_rmse, "o-", label="Uniform Sample Hard (replayed token est.)", linewidth=2)
    ax.plot(res_tok, res_rmse, "s-", label="Residual lambda=1 (replayed token est.)", linewidth=2)
    ax.set_title("Replayed Token Cost Estimate vs RMSE (Target: C_DM=5)")
    ax.set_xlabel("Estimated total input tokens (replayed from live means)")
    ax.set_ylabel("Raw RMSE vs C_DM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    fig.suptitle("Cost-Matched Comparison (Q63 sf10, 1,000 seeds; Build=32 requests)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Diagnostic output directory")
    args = parser.parse_args()

    summary = json.loads((args.input / "diagnostic_summary.json").read_text())
    fig_dir = args.input / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_matrices(args.input / "matrices.npz", fig_dir / "matrices_4panel.png")
    plot_sampling_curves(summary, fig_dir / "sampling_error_curves.png")
    plot_budget_curves(summary, fig_dir / "budget_curves.png")
    print("Generated figures in", fig_dir)


if __name__ == "__main__":
    main()
