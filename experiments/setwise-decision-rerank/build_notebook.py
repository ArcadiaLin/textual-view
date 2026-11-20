#!/usr/bin/env python3
"""Generate `notebooks/setwise_decision_rerank.ipynb`."""
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "notebooks" / "setwise_decision_rerank.ipynb"


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
            """# Setwise Re-ranking with Decision Models (`AnyJev Tacit-9B`, `Kev-9B`, & `Gemini Decision 1`)

本实验在 [`cookbooks/rerank_typesafe`](../../../references/repos/typesafe-public-examples/cookbooks-zh-en/cookbooks/rerank_typesafe.md) 所使用的 **CLERC 法律判例检索数据集**（3,565 个判例段落，40 条查询，BM25 Top-30 初筛候选）上，将 [Zhuang et al. (SIGIR 2024) Setwise Reranking](../../../references/papers/2024-Setwise/analysis.md) 的思想迁移到三款不同读出机制的 **System One 决策模型（Decision Models）** 上进行实践对比：

1. **本地 AnyJev (`Tacit-9B`)**：通过 `infra/anyjev/docker-compose.yml` 部署于 `http://127.0.0.1:8100/v1/decide`（直连不走代理），基于 `Qwen3.5-9B` 自蒸馏，单次前向读取末尾词汇表 token `Yes/No` (`yes_no`) 或选项字母 `A..Z` (`choice`, 最多 26 项) 的归一化概率分布。
2. **本地 Kev-9B (`/srv/models/jaredpalmer/kev-9b`)**：通过 `infra/kev/docker-compose.yml` 部署于 `http://127.0.0.1:8009/v1/systemone`（直连不走代理，TypeSafe System One API 兼容），基于 `Qwen3.5-9B-Base` + LoRA ($r=16$) + 专用 **Pointer Head**（在 `<decide>` 隐状态与各 `</opt>` 选项结束隐状态之间做点积并除以校准温度 $T=2.19$），支持单次多问题 Fan-out 与最多 255 项 `Choice`。
3. **Google Gemini Decision 1 (`gemini-decision-1`)**：通过 `https://generativelanguage.googleapis.com/v1beta/decisions` 调用（显式经由本地 `http://127.0.0.1:10809` 代理访问；实验结果已完整缓存于 `notebooks/_scratch/decision_cache.json`，直接从缓存读取，无需重跑线上 API），输出 0 生成 token，直接返回校准后的 `noul` 或 `choice`（支持 2~255 项）概率分布。

---

## 核心实验问题

1. **Pointwise vs. Setwise (`Choice` / `listwise.likelihood`)**：在三款不同架构与读出机制（词表字母 Logits vs. 跨度 Pointer Head vs. 云端原生 Decision API）的 Decision Model 上，把候选段落放到同一上下文中做横向 Setwise 比较，相比孤立打分的 Pointwise `Noul` 各自表现如何？
2. **两种 Setwise 提示表示（`tagged` vs. `criteria/inline`）与模型架构的耦合规律**：把候选文本编号后放入 `state`、仅在选项中放短标签（`tagged`），与直接把候选文本放入 `Choice` 的选项跨度中（`criteria/inline`），在 **词汇表字母读出（AnyJev）** 与 **选项跨度指针读出（Kev-9B）** 上为何呈现完全相反的优劣反转？
3. **候选规模受限时的 Setwise 组织结构**：分组锦标赛（Tournament `6×5 -> 12`）、滑动窗口（Sliding Window）、最大堆（Heap Sort）与粗到细级联（Coarse-to-Fine）各自的准确率—效率权衡如何？"""
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

load_env()
corpus, queries, golds, candidates = load_clerc_slice()
cache = DiskCache(SCRATCH_DIR / "decision_cache.json")

print(f"CLERC Corpus size: {len(corpus):,} passages | Queries: {len(queries)} | Shortlist per query: {len(next(iter(candidates.values())))}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. 数据集与 BM25 快速检索基线

我们先检查全部 40 条查询以及实验子集（默认前 `N_EVAL_QUERIES = 10` 条查询；所有请求均自动持久化在 `notebooks/_scratch/decision_cache.json` 中，其中 `gemini-decision-1` 100% 命中本地缓存）上的 BM25 初筛表现。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """N_EVAL_QUERIES = 10  # 缓存命中时秒级重放，不会重新请求 Google Decision 1 API
qids = list(queries)[:N_EVAL_QUERIES]

bm25_all = evaluate_rankings(candidates, golds, list(queries))
bm25_sub = evaluate_rankings({q: candidates[q] for q in qids}, golds, qids)

print(f"BM25 (All 40 queries) -> Top-1: {bm25_all['top1']:.1%}, Top-5: {bm25_all['top5']:.1%}, Top-10: {bm25_all['top10']:.1%}, NDCG@10: {bm25_all['ndcg10']:.3f}, MRR: {bm25_all['mrr']:.3f}")
print(f"BM25 (Eval {len(qids)} queries) -> Top-1: {bm25_sub['top1']:.1%}, Top-5: {bm25_sub['top5']:.1%}, Top-10: {bm25_sub['top10']:.1%}, NDCG@10: {bm25_sub['ndcg10']:.3f}, MRR: {bm25_sub['mrr']:.3f}")
print("\\nEval query Gold ranks in BM25 top-30:", bm25_sub["ranks"])"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. 单条困难查询案例剖析（Case Study: BM25 排在第 21 名的 Query）

以第 1 条查询 `qid = 392149` 为例，BM25 将标准答案（Gold passage）排在了第 **21 名**。我们对比三款决策模型在该查询上的 Pointwise 与 Setwise 表现："""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """sample_qid = qids[0]
sample_query = queries[sample_qid]
sample_gold = golds[sample_qid]
sample_cids = candidates[sample_qid]
bm25_gold_pos = sample_cids.index(sample_gold) + 1

aj = AnyJevBackend(cache=cache)
kv = KevBackend(cache=cache)
gm = GeminiDecisionBackend(cache=cache)

# 1. AnyJev Pointwise vs Setwise Tournament (tagged)
aj_pw_scores = aj.score_pointwise_batch(sample_query, sample_cids, corpus)
aj_pw_ranked = sorted(sample_cids, key=lambda c: (-aj_pw_scores[c], sample_cids.index(c)))
aj_tourn_ranked = rerank_setwise_tournament(aj, sample_query, sample_cids, corpus, group_size=5, top_per_group=2, mode="tagged")

# 2. Kev-9B Pointwise (30 state-pairs vs Fan-out) & Setwise Tournament (criteria)
kv_pw_scores = kv.score_pointwise_batch(sample_query, sample_cids, corpus)
kv_pw_ranked = sorted(sample_cids, key=lambda c: (-kv_pw_scores[c], sample_cids.index(c)))
kv_fan_scores = kv.score_pointwise_fanout_one_call(sample_query, sample_cids, corpus)
kv_fan_ranked = sorted(sample_cids, key=lambda c: (-kv_fan_scores[c], sample_cids.index(c)))
kv_tourn_ranked = rerank_setwise_tournament(kv, sample_query, sample_cids, corpus, group_size=5, top_per_group=2, mode="criteria")

# 3. Gemini Pointwise Fan-out vs One-shot Choice-30 (from cache)
gm_pw_scores = gm.score_pointwise_fanout_one_call(sample_query, sample_cids, corpus)
gm_pw_ranked = sorted(sample_cids, key=lambda c: (-gm_pw_scores[c], sample_cids.index(c)))
gm_ch_probs = gm.choice_setwise(sample_query, sample_cids, corpus, mode="criteria")
gm_ch_ranked = sorted(sample_cids, key=lambda c: (-gm_ch_probs[c], sample_cids.index(c)))

print(f"Query ID: {sample_qid} | BM25 Gold Rank: #{bm25_gold_pos}/30")
print(f"  AnyJev Pointwise Noul              -> Gold Rank: #{aj_pw_ranked.index(sample_gold)+1}/30 (Gold P(Yes) = {aj_pw_scores[sample_gold]:.3f})")
print(f"  AnyJev Setwise Tournament (tagged) -> Gold Rank: #{aj_tourn_ranked.index(sample_gold)+1}/30")
print(f"  Kev-9B Pointwise Noul (30 pairs)   -> Gold Rank: #{kv_pw_ranked.index(sample_gold)+1}/30 (Gold Noul   = {kv_pw_scores[sample_gold]:.3f})")
print(f"  Kev-9B Pointwise Fan-out (1x30)    -> Gold Rank: #{kv_fan_ranked.index(sample_gold)+1}/30 (Gold Noul   = {kv_fan_scores[sample_gold]:.3f})")
print(f"  Kev-9B Setwise Tournament (crit.)  -> Gold Rank: #{kv_tourn_ranked.index(sample_gold)+1}/30")
print(f"  Gemini Pointwise Fan-out           -> Gold Rank: #{gm_pw_ranked.index(sample_gold)+1}/30 (Gold Noul   = {gm_pw_scores[sample_gold]:.3f})")
print(f"  Gemini Setwise Choice-30           -> Gold Rank: #{gm_ch_ranked.index(sample_gold)+1}/30 (Gold Prob   = {gm_ch_probs[sample_gold]:.3f})")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. 三款决策模型系统评测：Local AnyJev (`Tacit-9B`) vs. Local `Kev-9B` vs. Google `gemini-decision-1`

我们在评测查询集上统一对比三款模型：
- **Local AnyJev (`Tacit-9B`)**：
  1. `Pointwise Noul (1 batch x 30)`
  2. `Setwise Tournament (6x5 -> 12, tagged)`
  3. `Setwise Tournament (6x5 -> 12, inline)`
  4. `Setwise Sliding Window (w=6, s=4, tagged)`
  5. `Setwise Heap Sort (c=4, top-10, tagged)`
  6. `Coarse-to-Fine (Noul-30 -> Choice-10, tagged)`
- **Local Kev-9B (`Qwen3.5-9B` + Pointer Head)**：
  1. `Pointwise Noul (30 state-pairs, cookbook)`：完全遵循 `rerank_typesafe` Cookbook，每次将 `(query_excerpt, candidate_passage)` 放入 `state`（≤8,192 验证上下文区间），由 Kev 服务端自动合并为 CUDA Graph 批次并发执行。
  2. `Pointwise Noul Fan-out (1 call x 30 Nouls)`：单次请求共享 `state: query`，并发挂载 30 个 `noul` 分支（利用 Kev 的 State Prefix Cache 避免重复编码 Query）。
  3. `One-shot Setwise Choice-30 (1 call, tagged)`：单次请求把 30 个候选拼入 `state`（约 18.7k tokens，超出 Kev-9B 的 7,552 训练长度窗口，用于检验长上下文退化边界）。
  4. `Setwise Tournament (6x5 -> 12, tagged, 7 calls)`：分组锦标赛（段落放在 `state`，选项仅放短标签 `Passage [cid]`）。
  5. `Setwise Tournament (6x5 -> 12, criteria, 7 calls)`：分组锦标赛（段落直接放在 `criteria[cid]` 的 `<opt>...</opt>` 跨度内，让 Pointer Head 在 `</opt>` 处直接读取候选段落表征）。
  6. `Setwise Sliding Window (w=6, s=4, tagged)`
  7. `Coarse-to-Fine (Pointwise Noul -> Choice-5, criteria)`：先用 Pointwise Noul 选出前 5 名，再用 1 次 `Choice-5 (criteria)` 做横向精排。
- **Google Gemini Decision 1 (`gemini-decision-1`, cached)**：
  1. `Pointwise Noul Fan-out (1 call x 30 Nouls)`
  2. `One-shot Setwise Choice-30 (1 call, criteria)`
  3. `One-shot Setwise Choice-30 (1 call, tagged)`
  4. `Setwise Tournament (6x5 -> 12, criteria)`
  5. `Coarse-to-Fine (Fan-out Noul-30 -> Choice-10)`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """all_results = []
rankings_by_method = {"BM25 Fast Search": {q: candidates[q] for q in qids}}

all_results.append({
    "backend": "Baseline",
    "method": "BM25 Fast Search",
    **{k: bm25_sub[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
    "calls_per_q": 0.0,
    "decisions_per_q": 0.0,
    "tokens_per_q": 0.0,
    "latency_s_per_q": 0.0,
})

# --- 1. Run AnyJev ---
methods_aj = [
    ("AJ: Pointwise Noul (batch 30)", lambda q: rerank_pointwise(aj, queries[q], candidates[q], corpus)),
    ("AJ: Setwise Tournament (6x5->12, tagged)", lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "tagged")),
    ("AJ: Setwise Tournament (6x5->12, inline)", lambda q: rerank_setwise_tournament(aj, queries[q], candidates[q], corpus, 5, 2, "inline")),
    ("AJ: Setwise Sliding Window (w=6, s=4)", lambda q: rerank_setwise_sliding_window(aj, queries[q], candidates[q], corpus, 6, 4, 1, "tagged")),
    ("AJ: Setwise Heap Sort (c=4, top-10)", lambda q: rerank_setwise_heap(aj, queries[q], candidates[q], corpus, 3, 10, "tagged")),
    ("AJ: Coarse-to-Fine (Noul-30 -> Choice-10)", lambda q: rerank_coarse_to_fine(aj, queries[q], candidates[q], corpus, 10, "tagged")),
]

for name, fn in methods_aj:
    aj.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_rankings(ranked, golds, qids)
    nq = len(qids)
    all_results.append({
        "backend": "AnyJev (Tacit-9B)",
        "method": name,
        **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
        "calls_per_q": aj.calls / nq,
        "decisions_per_q": aj.decisions / nq,
        "tokens_per_q": 0.0,
        "latency_s_per_q": aj.total_latency_s / nq,
    })

# --- 2. Run Kev-9B ---
methods_kv = [
    ("Kev: Pointwise Noul (30 state-pairs)", lambda q: rerank_pointwise(kv, queries[q], candidates[q], corpus)),
    ("Kev: Pointwise Noul Fan-out (1x30 Nouls)", lambda q: rerank_pointwise_fanout(kv, queries[q], candidates[q], corpus)),
    ("Kev: One-shot Setwise Choice-30 (tagged)", lambda q: rerank_oneshot_choice(kv, queries[q], candidates[q], corpus, mode="tagged")),
    ("Kev: Setwise Tournament (6x5->12, tagged)", lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "tagged")),
    ("Kev: Setwise Tournament (6x5->12, criteria)", lambda q: rerank_setwise_tournament(kv, queries[q], candidates[q], corpus, 5, 2, "criteria")),
    ("Kev: Setwise Sliding Window (w=6, s=4, tagged)", lambda q: rerank_setwise_sliding_window(kv, queries[q], candidates[q], corpus, 6, 4, 1, "tagged")),
    ("Kev: Coarse-to-Fine (Noul-30 -> Choice-5, criteria)", lambda q: rerank_coarse_to_fine(kv, queries[q], candidates[q], corpus, 5, "criteria", use_fanout=False)),
]

for name, fn in methods_kv:
    kv.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_rankings(ranked, golds, qids)
    nq = len(qids)
    all_results.append({
        "backend": "Kev-9B (Local)",
        "method": name,
        **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
        "calls_per_q": kv.calls / nq,
        "decisions_per_q": kv.decisions / nq,
        "tokens_per_q": kv.total_tokens / nq,
        "latency_s_per_q": kv.total_latency_s / nq,
    })

# --- 3. Run Gemini Decision 1 (Cached) ---
methods_gm = [
    ("GM: Pointwise Noul Fan-out (1x30 Nouls)", lambda q: rerank_pointwise_fanout(gm, queries[q], candidates[q], corpus)),
    ("GM: One-shot Setwise Choice-30 (criteria)", lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="criteria")),
    ("GM: One-shot Setwise Choice-30 (tagged)", lambda q: rerank_oneshot_choice(gm, queries[q], candidates[q], corpus, mode="tagged")),
    ("GM: Setwise Tournament (6x5->12, criteria)", lambda q: rerank_setwise_tournament(gm, queries[q], candidates[q], corpus, 5, 2, "criteria")),
    ("GM: Coarse-to-Fine (Noul-30 -> Choice-10)", lambda q: rerank_coarse_to_fine(gm, queries[q], candidates[q], corpus, 10, "criteria", use_fanout=True)),
]

for name, fn in methods_gm:
    gm.reset_stats()
    ranked = {q: fn(q) for q in qids}
    rankings_by_method[name] = ranked
    m = evaluate_rankings(ranked, golds, qids)
    nq = len(qids)
    all_results.append({
        "backend": "Gemini Decision 1",
        "method": name,
        **{k: m[k] for k in ("top1", "top5", "top10", "ndcg10", "mrr", "mean_rank")},
        "calls_per_q": gm.calls / nq,
        "decisions_per_q": gm.decisions / nq,
        "tokens_per_q": gm.total_tokens / nq,
        "latency_s_per_q": gm.total_latency_s / nq,
    })

# Render Markdown comparison table
lines = [
    "| Backend | Method | Top-1 | Top-5 | Top-10 | NDCG@10 | MRR | Mean Rank | HTTP Calls/q | Decisions/q | Input Tokens/q | Latency/q |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in all_results:
    tok_str = f"{r['tokens_per_q']:,.0f}" if r["tokens_per_q"] > 0 else "—"
    lat_str = f"{r['latency_s_per_q']:.2f}s" if r["latency_s_per_q"] > 0 else "—"
    lines.append(
        f"| {r['backend']} | **{r['method']}** | {r['top1']:.1%} | {r['top5']:.1%} | {r['top10']:.1%} | "
        f"{r['ndcg10']:.3f} | {r['mrr']:.3f} | {r['mean_rank']:.1f} | {r['calls_per_q']:.1f} | "
        f"{r['decisions_per_q']:.1f} | {tok_str} | {lat_str} |"
    )

display(Markdown("\\n".join(lines)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. 可视化对比：三款决策模型的排序准确率与逐查询 Gold 排名轨迹"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=120)

# Plot 1: Top-1, Top-5, and MRR across key methods from all 3 backends
selected_methods = [
    "BM25 Fast Search",
    "AJ: Pointwise Noul (batch 30)",
    "AJ: Setwise Tournament (6x5->12, tagged)",
    "AJ: Coarse-to-Fine (Noul-30 -> Choice-10)",
    "Kev: Pointwise Noul (30 state-pairs)",
    "Kev: Pointwise Noul Fan-out (1x30 Nouls)",
    "Kev: Setwise Tournament (6x5->12, criteria)",
    "Kev: Setwise Tournament (6x5->12, tagged)",
    "GM: Pointwise Noul Fan-out (1x30 Nouls)",
    "GM: One-shot Setwise Choice-30 (tagged)",
]
sel_rows = [next(r for r in all_results if r["method"] == m) for m in selected_methods]

y = np.arange(len(sel_rows))
h = 0.25
ax = axes[0]
ax.barh(y - h, [r["top1"] * 100 for r in sel_rows], height=h, label="Top-1 (%)", color="#3b82f6")
ax.barh(y, [r["top5"] * 100 for r in sel_rows], height=h, label="Top-5 (%)", color="#10b981")
ax.barh(y + h, [r["mrr"] * 100 for r in sel_rows], height=h, label="MRR (x100)", color="#f59e0b")

ax.set_yticks(y, [r["method"] for r in sel_rows], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Score (%)")
ax.set_xlim(0, 110)
ax.set_title("Re-ranking Effectiveness Across 3 Decision Models")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.legend(loc="lower right", fontsize=8.5)

# Plot 2: Per-query Gold Rank shift (BM25 vs AnyJev vs Kev-9B vs Gemini Decision 1)
ax2 = axes[1]
x_idx = np.arange(len(qids))
bm25_ranks = [rankings_by_method["BM25 Fast Search"][q].index(golds[q]) + 1 for q in qids]
aj_ranks = [rankings_by_method["AJ: Coarse-to-Fine (Noul-30 -> Choice-10)"][q].index(golds[q]) + 1 for q in qids]
kv_ranks = [rankings_by_method["Kev: Pointwise Noul (30 state-pairs)"][q].index(golds[q]) + 1 for q in qids]
gm_ranks = [rankings_by_method["GM: One-shot Setwise Choice-30 (tagged)"][q].index(golds[q]) + 1 for q in qids]

ax2.plot(x_idx, bm25_ranks, "o--", label="BM25 Fast Search", color="#9ca3af", alpha=0.8)
ax2.plot(x_idx, aj_ranks, "s-", label="AnyJev Coarse-to-Fine (Noul->Choice-10)", color="#3b82f6")
ax2.plot(x_idx, kv_ranks, "D-", label="Kev-9B Pointwise Noul (30 state-pairs)", color="#8b5cf6")
ax2.plot(x_idx, gm_ranks, "^-", label="Gemini One-shot Setwise Choice-30", color="#10b981")

ax2.set_xticks(x_idx, [f"Q{i+1}\\n({q})" for i, q in enumerate(qids)], fontsize=8)
ax2.set_ylabel("Gold Passage Rank (1 is best)")
ax2.set_ylim(31, 0)
ax2.axhline(1, color="#10b981", linestyle=":", alpha=0.5)
ax2.axhline(5, color="#f59e0b", linestyle=":", alpha=0.5)
ax2.set_title("Gold Passage Rank per Query (Lower = Better)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="lower left", fontsize=8)

plt.tight_layout()
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Setwise 顺序敏感性诊断（Permutation Sensitivity Diagnostic）

在 [`references/papers/2024-Setwise/analysis.md`](../../../references/papers/2024-Setwise/analysis.md#L244-L250) 中我们专门讨论过：把候选作为集合输入给模型并不自动保证数学上的排列不变性（Permutation Invariance）。

下面我们在 `sample_qid` 上固定相同的候选集合，测试将输入顺序设为 **BM25 正序（Original）**、**BM25 逆序（Reversed）** 和 **随机打乱（Shuffled）** 时，三款模型的 Setwise 算子输出的 Gold 排名是否稳定。"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import random

orders = {
    "BM25 Original": list(sample_cids),
    "BM25 Reversed": list(reversed(sample_cids)),
    "Random Seed 42": random.Random(42).sample(sample_cids, len(sample_cids)),
}

perm_rows = []
for order_name, ord_cids in orders.items():
    # AnyJev Setwise Tournament (tagged)
    aj_ord = rerank_setwise_tournament(aj, sample_query, ord_cids, corpus, 5, 2, "tagged")
    # Kev-9B Setwise Tournament (criteria)
    kv_ord = rerank_setwise_tournament(kv, sample_query, ord_cids, corpus, 5, 2, "criteria")
    # Gemini One-shot Choice-30 (criteria, from cache)
    gm_probs = gm.choice_setwise(sample_query, ord_cids, corpus, mode="criteria")
    gm_ord = sorted(ord_cids, key=lambda c: (-gm_probs[c], ord_cids.index(c)))
    perm_rows.append({
        "Input Order": order_name,
        "Initial Gold Pos": ord_cids.index(sample_gold) + 1,
        "AnyJev Tournament (tagged)": aj_ord.index(sample_gold) + 1,
        "Kev-9B Tournament (criteria)": kv_ord.index(sample_gold) + 1,
        "Gemini Choice-30 (criteria)": gm_ord.index(sample_gold) + 1,
        "Gemini Gold Prob": gm_probs[sample_gold],
    })

perm_md = [
    "| Input Order | Initial Gold Pos | AnyJev Tournament (`tagged`) | Kev-9B Tournament (`criteria`) | Gemini `Choice-30` (`criteria`) | Gemini Gold Prob |",
    "|---|---:|---:|---:|---:|---:|",
]
for r in perm_rows:
    perm_md.append(
        f"| {r['Input Order']} | #{r['Initial Gold Pos']} | **#{r['AnyJev Tournament (tagged)']}** | "
        f"**#{r['Kev-9B Tournament (criteria)']}** | **#{r['Gemini Choice-30 (criteria)']}** | {r['Gemini Gold Prob']:.3f} |"
    )
display(Markdown("\\n".join(perm_md)))"""
        )
    )

    nb.cells = cells
    return nb


if __name__ == "__main__":
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NB_PATH)
    print(f"Wrote notebook to {NB_PATH}")
