#!/usr/bin/env python3
"""Generate `notebooks/stark_prime_setwise_rerank.ipynb`."""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "notebooks" / "stark_prime_setwise_rerank.ipynb"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (setwise-decision-rerank)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# STaRK-Prime Top-20 Re-ranking with Decision Models (`AnyJev Tacit-9B`, `Kev-9B`, & `Gemini Decision 1`)

本 Notebook 将 Setwise vs. Pointwise 决策模型重排序实验扩展到半结构化生物医药知识库问答基准 **STaRK-Prime**（[Wu et al., NeurIPS 2024 Datasets & Benchmarks](../../../references/datasets/stark/README.md)）：

- **免检索直接重排（Zero-Retrieval Re-ranking Setup）**：
  - 直接读取本仓库已固化的官方 `test-0.1` 划分（共 280 道题）BM25 初筛结果 [`runs/stark-prime/test-10pct/baseline.json`](../../../runs/stark-prime/test-10pct/baseline.json) 与预渲染的 129,375 节点文档库 [`data/stark-prime/bm25/documents.sqlite`](../../../data/stark-prime/bm25/documents.sqlite)。
  - 在全部 280 道测试题中，有 **115 道题（41.1%）** 的 BM25 **Top-20** 候选列表中包含至少 1 个标准答案节点（Oracle-Covered Subset）；其余 165 道题的 Top-20 中无标准答案（任何 Top-20 重排序器在该 165 题上的得分恒为 0）。
  - 因此，对这 **115 道 Top-20 命中题** 进行重排，既能无噪声地评测重排序器在候选池内的区分能力，也能通过精确线性折算（$\text{Metric}_{280} = \text{Metric}_{115} \times \frac{115}{280}$）直接得到全部 280 道题的端到端 `Hit@1 / Hit@5 / Hit@10 / MRR@20` 指标，同时节省 59% 的无效计算与 API 成本。

---

## 三款被测决策模型与 $N=20$ 候选池的架构优势

与 CLERC（$N=30$，每段约 2,000 字符）相比，STaRK-Prime Top-20（$N=20$，每节点文档截断至前 1,200 字符，单题总长约 5,500 tokens）解锁了两个关键架构特性：
1. **`AnyJev (Tacit-9B)` 解锁单次 `One-shot Setwise Choice-20`**：AnyJev 的词表字母读出上限为 26 项（且 vLLM `top_logprobs=20`），在 $N=20$ 下无需分组即可用 **1 次前向调用** 直接输出全部 20 个候选节点的概率分布！
2. **`Kev-9B` 完全处于 8,192 验证上下文窗口内**：20 个候选节点总长度约 **5,500 tokens**，严格低于 `Kev-9B` 的 **7,552 训练长度 / 8,192 验证无退化阈值**，使我们可以公平对比 `Kev-9B` 的 `One-shot Setwise Choice-20 (criteria)` 与 `(tagged)` 模式。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

EXP_DIR = Path.cwd().resolve()
if EXP_DIR.name == "notebooks":
    EXP_DIR = EXP_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

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

load_env()
corpus, queries, golds, candidates, total_split_q = load_stark_prime_top20(
    top_k=20, max_doc_chars=1200, only_covered=True
)
cache = DiskCache(SCRATCH_DIR / "stark_decision_cache.json")

print(
    f"STaRK-Prime test-0.1 total queries: {total_split_q} | "
    f"Top-20 Oracle-Covered queries: {len(queries)} ({len(queries)/total_split_q:.1%}) | "
    f"Unique candidate nodes loaded: {len(corpus):,}"
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. 数据集特征与 BM25 Top-20 基线表现

我们先统计 115 道 Top-20 命中题的答案数量分布（STaRK-Prime 包含单答案题与多答案题），以及初始 BM25 排序在 **115 题可解子集** 和 **280 题全量 `test-0.1`** 上的基线指标。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """qids = list(queries)
bm25_rankings = {q: candidates[q] for q in qids}
bm25_metrics = evaluate_stark_rankings(bm25_rankings, golds, qids, total_split_queries=total_split_q)

gold_counts = [len(golds[q]) for q in qids]
multi_gold_q = sum(c > 1 for c in gold_counts)
non_top1_q = sum(r > 1 for r in bm25_metrics["best_ranks"].values())

print(f"Covered queries: {len(qids)} (Single-gold: {len(qids)-multi_gold_q}, Multi-gold: {multi_gold_q})")
print(f"Queries where BM25 Top-1 is already Gold: {len(qids)-non_top1_q} ({1 - non_top1_q/len(qids):.1%})")
print(f"Queries where Gold is in rank 2..20 (room for reranking gain): {non_top1_q} ({non_top1_q/len(qids):.1%})")
print(
    f"\\n[BM25 Baseline on 115 Covered Queries] "
    f"Hit@1: {bm25_metrics['hit1']:.1%} | Hit@3: {bm25_metrics['hit3']:.1%} | "
    f"Hit@5: {bm25_metrics['hit5']:.1%} | Hit@10: {bm25_metrics['hit10']:.1%} | "
    f"NDCG@10: {bm25_metrics['ndcg10']:.3f} | MRR: {bm25_metrics['mrr']:.3f} | "
    f"Mean Best Gold Rank: {bm25_metrics['mean_best_rank']:.2f}"
)
print(
    f"[BM25 Baseline on All 280 test-0.1 Queries] "
    f"Hit@1: {bm25_metrics['full280_hit1']:.2%} | Hit@5: {bm25_metrics['full280_hit5']:.2%} | "
    f"Hit@10: {bm25_metrics['full280_hit10']:.2%} | Recall@20: {bm25_metrics['full280_recall20']:.2%} | "
    f"MRR@20: {bm25_metrics['full280_mrr20']:.3f}"
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. 单条生物医药查询案例剖析（Case Study）

我们选取一条 BM25 未能把标准答案排在第 1 名的代表性生物医药查询，直观对比候选节点的半结构化文本形态以及三款决策模型（`AnyJev Tacit-9B`、`Kev-9B`、`Gemini Decision 1`）的 Pointwise 与 Setwise 重排结果。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# 选取第一条 BM25 最佳标准答案排名 >= 4 的查询
sample_qid = next(q for q in qids if bm25_metrics["best_ranks"][q] >= 4)
sample_query = queries[sample_qid]
sample_gold = golds[sample_qid]
sample_cids = candidates[sample_qid]

aj = make_stark_anyjev(cache=cache)
kv = make_stark_kev(cache=cache)
gm = make_stark_gemini(cache=cache)

print(f"Sample Query ID: {sample_qid}")
print(f"Query Text: {sample_query}")
print(f"Gold Node IDs: {sorted(sample_gold)}")
print(f"BM25 Gold Positions in Top-20: {[i+1 for i, c in enumerate(sample_cids) if c in sample_gold]}")
first_gold_cid = next(c for c in sample_cids if c in sample_gold)
print(f"\\n--- Sample Gold Node Document (node_id={first_gold_cid}) ---\\n{corpus[first_gold_cid][:500]}...")

for label, fn in [
    ("AnyJev Pointwise Noul (batch 20)", lambda: rerank_pointwise(aj, sample_query, sample_cids, corpus)),
    ("AnyJev One-shot Choice-20 (tagged)", lambda: rerank_oneshot_choice(aj, sample_query, sample_cids, corpus, mode="tagged")),
    ("Kev-9B Pointwise Noul (20 state-pairs)", lambda: rerank_pointwise(kv, sample_query, sample_cids, corpus)),
    ("Kev-9B One-shot Choice-20 (criteria)", lambda: rerank_oneshot_choice(kv, sample_query, sample_cids, corpus, mode="criteria")),
    ("Gemini Pointwise Fan-out (1x20 Nouls)", lambda: rerank_pointwise_fanout(gm, sample_query, sample_cids, corpus)),
    ("Gemini One-shot Choice-20 (criteria)", lambda: rerank_oneshot_choice(gm, sample_query, sample_cids, corpus, mode="criteria")),
]:
    ranked = fn()
    pos = [i + 1 for i, c in enumerate(ranked) if c in sample_gold]
    print(f"  {label:<40} -> Gold ranks: {pos} (Best: #{min(pos)})")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. 115 道 Top-20 命中题全量对比评测（`AnyJev` vs. `Kev-9B` vs. `Gemini Decision 1`）

下面在全部 115 道 Top-20 命中题上评测三款模型的 14 种重排序配置（所有响应均缓存于 `notebooks/_scratch/stark_decision_cache.json`，命中缓存时秒级重放），并同时汇报：
- **115 题可解子集指标**：`Hit@1`、`Hit@3`、`Hit@5`、`Hit@10`、`Recall@5`、`NDCG@10`、`MRR`、`Mean Best Rank`；
- **全量 280 题 `test-0.1` 端到端折算指标**：`280q Hit@1`、`280q Hit@5`、`280q MRR@20`（可直接与 [`runs/stark-prime/test-10pct/summary.json`](../../../runs/stark-prime/test-10pct/summary.json) 中的 BM25 基线 `13.93% Hit@1` 和 `pi` Agent 多轮检索结果 `29.29% Hit@1` 对照）。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """all_results = []
rankings_by_method = {"BM25 Fast Search (Top-20)": bm25_rankings}

all_results.append({
    "backend": "Baseline",
    "method": "BM25 Fast Search (Top-20)",
    **bm25_metrics,
    "calls_per_q": 0.0,
    "decisions_per_q": 0.0,
    "tokens_per_q": 0.0,
    "latency_s_per_q": 0.0,
})

# 1. Local AnyJev (Tacit-9B)
methods_aj = [
    ("AJ: Pointwise Noul (batch 20)", lambda q: rerank_pointwise(aj, queries[q], candidates[q], corpus)),
    ("AJ: One-shot Setwise Choice-20 (tagged)", lambda q: rerank_oneshot_choice(aj, queries[q], candidates[q], corpus, mode="tagged")),
    ("AJ: Setwise Tournament (4x5->8, tagged)", lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "tagged")),
    ("AJ: Coarse-to-Fine (Noul-20 -> Choice-10)", lambda q: rerank_coarse_to_fine(aj, queries[q], candidates[q], corpus, 10, "tagged")),
]

for name, fn in methods_aj:
    aj.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
    nq = len(qids)
    all_results.append({
        "backend": "AnyJev (Tacit-9B)",
        "method": name,
        **m,
        "calls_per_q": aj.calls / nq,
        "decisions_per_q": aj.decisions / nq,
        "tokens_per_q": 0.0,
        "latency_s_per_q": aj.total_latency_s / nq,
    })

# 2. Local Kev-9B
methods_kv = [
    ("Kev: Pointwise Noul (20 state-pairs)", lambda q: rerank_pointwise(kv, queries[q], candidates[q], corpus)),
    ("Kev: Pointwise Noul Fan-out (1x20 Nouls)", lambda q: rerank_pointwise_fanout(kv, queries[q], candidates[q], corpus)),
    ("Kev: One-shot Setwise Choice-20 (criteria)", lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="criteria")),
    ("Kev: One-shot Setwise Choice-20 (tagged)", lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="tagged")),
    ("Kev: Setwise Tournament (4x5->8, criteria)", lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "criteria")),
    ("Kev: Coarse-to-Fine (Noul-20 -> Choice-5, criteria)", lambda q: rerank_coarse_to_fine(kv, queries[q], candidates[q], corpus, 5, "criteria", use_fanout=False)),
]

for name, fn in methods_kv:
    kv.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
    nq = len(qids)
    all_results.append({
        "backend": "Kev-9B (Local)",
        "method": name,
        **m,
        "calls_per_q": kv.calls / nq,
        "decisions_per_q": kv.decisions / nq,
        "tokens_per_q": kv.total_tokens / nq,
        "latency_s_per_q": kv.total_latency_s / nq,
    })

# 3. Google Gemini Decision 1
methods_gm = [
    ("GM: Pointwise Noul Fan-out (1x20 Nouls)", lambda q: rerank_pointwise_fanout(gm, queries[q], candidates[q], corpus)),
    ("GM: One-shot Setwise Choice-20 (criteria)", lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="criteria")),
    ("GM: One-shot Setwise Choice-20 (tagged)", lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="tagged")),
    ("GM: Coarse-to-Fine (Noul-20 -> Choice-10)", lambda q: rerank_coarse_to_fine(gm, queries[q], candidates[q], corpus, 10, "criteria", use_fanout=True)),
]

for name, fn in methods_gm:
    gm.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_stark_rankings(ranked, golds, qids, total_split_queries=total_split_q)
    nq = len(qids)
    all_results.append({
        "backend": "Gemini Decision 1",
        "method": name,
        **m,
        "calls_per_q": gm.calls / nq,
        "decisions_per_q": gm.decisions / nq,
        "tokens_per_q": gm.total_tokens / nq,
        "latency_s_per_q": gm.total_latency_s / nq,
    })

lines = [
    "| Backend | Method | Hit@1 (115q) | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | MRR | Mean Rank | **280q Hit@1** | **280q Hit@5** | Calls/q | Tokens/q | Latency/q |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in all_results:
    tok_str = f"{r['tokens_per_q']:,.0f}" if r["tokens_per_q"] > 0 else "—"
    lat_str = f"{r['latency_s_per_q']:.2f}s" if r["latency_s_per_q"] > 0 else "—"
    lines.append(
        f"| {r['backend']} | **{r['method']}** | {r['hit1']:.1%} | {r['hit3']:.1%} | {r['hit5']:.1%} | "
        f"{r['hit10']:.1%} | {r['ndcg10']:.3f} | {r['mrr']:.3f} | {r['mean_best_rank']:.2f} | "
        f"**{r['full280_hit1']:.1%}** | **{r['full280_hit5']:.1%}** | {r['calls_per_q']:.1f} | {tok_str} | {lat_str} |"
    )

display(Markdown("\\n".join(lines)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. 可视化分析：三款模型的重排序提升与最佳标准答案排名分布"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=120)

# Plot 1: Hit@1, Hit@5, and MRR across key methods (115 Covered Queries)
selected_methods = [
    "BM25 Fast Search (Top-20)",
    "AJ: Pointwise Noul (batch 20)",
    "AJ: One-shot Setwise Choice-20 (tagged)",
    "AJ: Coarse-to-Fine (Noul-20 -> Choice-10)",
    "Kev: Pointwise Noul (20 state-pairs)",
    "Kev: One-shot Setwise Choice-20 (criteria)",
    "Kev: One-shot Setwise Choice-20 (tagged)",
    "Kev: Coarse-to-Fine (Noul-20 -> Choice-5, criteria)",
    "GM: Pointwise Noul Fan-out (1x20 Nouls)",
    "GM: One-shot Setwise Choice-20 (criteria)",
    "GM: One-shot Setwise Choice-20 (tagged)",
]
sel_rows = [next(r for r in all_results if r["method"] == m) for m in selected_methods]

y = np.arange(len(sel_rows))
h = 0.25
ax = axes[0]
ax.barh(y - h, [r["hit1"] * 100 for r in sel_rows], height=h, label="Hit@1 (%)", color="#3b82f6")
ax.barh(y, [r["hit5"] * 100 for r in sel_rows], height=h, label="Hit@5 (%)", color="#10b981")
ax.barh(y + h, [r["mrr"] * 100 for r in sel_rows], height=h, label="MRR (x100)", color="#f59e0b")

ax.set_yticks(y, [r["method"] for r in sel_rows], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Score on 115 Oracle-Covered Queries (%)")
ax.set_xlim(0, 108)
ax.set_title("STaRK-Prime Top-20 Re-ranking Effectiveness (115 Queries)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=8.5)

# Plot 2: Cumulative Hit@K curve (K = 1..20) for BM25 vs Best of Each Model
ax2 = axes[1]
ks = np.arange(1, 21)
curve_methods = [
    ("BM25 Fast Search (Top-20)", "#9ca3af", "o--"),
    ("AJ: One-shot Setwise Choice-20 (tagged)", "#3b82f6", "s-"),
    ("Kev: Pointwise Noul (20 state-pairs)", "#8b5cf6", "D-"),
    ("Kev: One-shot Setwise Choice-20 (criteria)", "#ec4899", "v--"),
    ("GM: One-shot Setwise Choice-20 (criteria)", "#10b981", "^-"),
]

for m_name, color, style in curve_methods:
    r = next(x for x in all_results if x["method"] == m_name)
    br = list(r["best_ranks"].values())
    curve = [sum(rank <= k for rank in br) / len(br) * 100 for k in ks]
    ax2.plot(ks, curve, style, label=m_name, color=color, markersize=4.5)

ax2.set_xticks(ks)
ax2.set_xlabel("Cutoff K")
ax2.set_ylabel("Hit@K on 115 Covered Queries (%)")
ax2.set_ylim(25, 103)
ax2.set_title("Cumulative Hit@K Curve (K = 1..20)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. 针对 76 道“BM25 未排第 1 名”困难题的专项净提升分析（Hard Subset: BM25 Best Rank $\\in [2, 20]$）

在 115 道 Top-20 命中题中，有 39 道题的 BM25 第 1 名已经是标准答案，另有 **76 道题** 的标准答案被 BM25 排在第 2~20 名（BM25 在这 76 题上的 `Hit@1 = 0.0%`）。我们专门考察各模型在这 76 道困难题上把标准答案“捞回第 1 名（Hit@1）”和“推入前 3 名（Hit@3）”的能力："""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """hard_qids = [q for q in qids if bm25_metrics["best_ranks"][q] > 1]
print(f"Hard-Covered Subset (BM25 Rank 2..20): {len(hard_qids)} queries")

hard_lines = [
    "| Backend | Method | Hard Hit@1 (76q) | Hard Hit@3 (76q) | Hard Hit@5 (76q) | Hard MRR | Mean Best Rank (BM25 = 5.59) |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for name, ranked in rankings_by_method.items():
    m_hard = evaluate_stark_rankings(ranked, golds, hard_qids, total_split_queries=total_split_q)
    backend_name = next(r["backend"] for r in all_results if r["method"] == name)
    hard_lines.append(
        f"| {backend_name} | **{name}** | **{m_hard['hit1']:.1%}** | {m_hard['hit3']:.1%} | "
        f"{m_hard['hit5']:.1%} | {m_hard['mrr']:.3f} | {m_hard['mean_best_rank']:.2f} |"
    )

display(Markdown("\\n".join(hard_lines)))"""
        )
    )

    nb.cells = cells
    return nb


if __name__ == "__main__":
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NB_PATH)
    print(f"Wrote notebook to {NB_PATH}")
