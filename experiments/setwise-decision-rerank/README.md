# Setwise Decision Model Re-ranking (`experiments/setwise-decision-rerank`)

本实验在 [`cookbooks/rerank_typesafe`](../../references/repos/typesafe-public-examples/cookbooks-zh-en/cookbooks/rerank_typesafe.md) 的 **CLERC 法律判例检索数据集**（3,565 个段落，40 条查询，BM25 Top-30 初筛列表）上，探索将 [Zhuang et al. (SIGIR 2024) Setwise Reranking](../../references/papers/2024-Setwise/analysis.md) 迁移到两类 **System One 决策模型（Decision Models）** 上的效果与效率权衡：

1. **Local AnyJev (`Tacit-9B`)**：通过 `infra/anyjev/docker-compose.yml` 启动在本地 `http://127.0.0.1:8100/v1/decide`（代码显式使用无代理直连 `ProxyHandler({})`）。
2. **Google Gemini Decision 1 (`gemini-decision-1`)**：通过 `https://generativelanguage.googleapis.com/v1beta/decisions` 调用（代码显式经由 `http://127.0.0.1:10809` 代理访问，读取项目根目录 `.env` 中的 `GEMINI_API_KEY` / `X-goog-api-key`）。

## 目录结构

- [`pyproject.toml`](pyproject.toml)：独立实验环境依赖声明（Python 3.12，`bm25s`、`matplotlib`、`ipykernel`、`nbclient`、`nbformat`）。
- [`rerank_lib.py`](rerank_lib.py)：
  - 数据集与 BM25 Top-30 初筛加载（`load_clerc_slice`）；
  - 确定性磁盘缓存（`DiskCache`，保存在 `notebooks/_scratch/decision_cache.json`，符合 `.gitignore` 规范）；
  - 双后端封装（`AnyJevBackend` 与 `GeminiDecisionBackend`）；
  - 排序算子实现：
    - `rerank_pointwise` / `rerank_pointwise_fanout`（Pointwise `Noul` 基线）
    - `rerank_oneshot_choice`（One-shot Setwise `Choice-30`）
    - `rerank_setwise_tournament`（两阶段分组锦标赛 `6×5 -> 12`，在 AnyJev 上第一轮 6 组打包为 1 次 batch 请求）
    - `rerank_setwise_sliding_window`（滑动窗口 `listwise.likelihood`）
    - `rerank_setwise_heap`（$c$-叉最大堆 Top-$k$ 选优）
    - `rerank_coarse_to_fine`（Pointwise `Noul-30` 粗筛 $\to$ Setwise `Choice-10` 精排级联）
- [`run_experiment.py`](run_experiment.py)：命令行评测入口，支持 `--backend {anyjev,gemini,both}` 与 `--num-queries N`。
- [`notebooks/setwise_decision_rerank.ipynb`](notebooks/setwise_decision_rerank.ipynb)：完整交互式实验 Notebook（含单条困难查询剖析、多方法对照表、准确率/排名迁移可视化图表、以及候选顺序置换敏感性诊断）。

## 快速运行

```bash
# 1. 运行命令行实验（结果自动缓存至 notebooks/_scratch/decision_cache.json）
experiments/setwise-decision-rerank/.venv/bin/python experiments/setwise-decision-rerank/run_experiment.py --backend both --num-queries 10

# 2. 在 VS Code / Jupyter 中打开 Notebook，或使用 nbclient 顶到底部重跑
experiments/setwise-decision-rerank/.venv/bin/python -m nbclient --execute experiments/setwise-decision-rerank/notebooks/setwise_decision_rerank.ipynb --inplace
```
