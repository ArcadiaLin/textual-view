# Setwise Decision Model Re-ranking (`experiments/setwise-decision-rerank`)

本实验在 [`cookbooks/rerank_typesafe`](../../references/repos/typesafe-public-examples/cookbooks-zh-en/cookbooks/rerank_typesafe.md) 的 **CLERC 法律判例检索数据集**（3,565 个段落，40 条查询，BM25 Top-30 初筛列表）上，探索将 [Zhuang et al. (SIGIR 2024) Setwise Reranking](../../references/papers/2024-Setwise/analysis.md) 迁移到三类 **System One 决策模型（Decision Models）** 上的效果与效率权衡：

1. **Local AnyJev (`Tacit-9B`)**：通过 [`infra/anyjev/docker-compose.yml`](../../infra/anyjev/docker-compose.yml) 启动在本地 `http://127.0.0.1:8100/v1/decide`（代码显式使用无代理直连 `ProxyHandler({})`），基于词表末尾字母/Yes-No token 读出概率。
2. **Local Kev-9B (`/srv/models/jaredpalmer/kev-9b`)**：通过 [`infra/kev/docker-compose.yml`](../../infra/kev/docker-compose.yml) 启动在本地 `http://127.0.0.1:8009/v1/systemone`（无代理直连，完全兼容 TypeSafe System One API），基于 `Qwen3.5-9B-Base` + LoRA ($r=16$) + 专用 **Pointer Head**（在 `<decide>` 隐状态与各 `</opt>` 选项结束隐状态之间做点积并除以校准温度 $T=2.19$）。
3. **Google Gemini Decision 1 (`gemini-decision-1`)**：通过 `https://generativelanguage.googleapis.com/v1beta/decisions` 调用（代码显式经由 `http://127.0.0.1:10809` 代理访问，结果已持久化在 `notebooks/_scratch/decision_cache.json`，默认重放缓存不重复扣费）。

## 目录结构

- [`pyproject.toml`](pyproject.toml)：独立实验环境依赖声明（Python 3.12，`bm25s`、`matplotlib`、`ipykernel`、`nbclient`、`nbformat`）。
- [`rerank_lib.py`](rerank_lib.py)：
  - 数据集与 BM25 Top-30 初筛加载（`load_clerc_slice`）；
  - 线程安全确定性磁盘缓存（`DiskCache`，保存在 `notebooks/_scratch/decision_cache.json`，符合 `.gitignore` 规范）；
  - 三后端封装（`AnyJevBackend`、`KevBackend` 与 `GeminiDecisionBackend`）；
  - 排序算子实现：
    - `rerank_pointwise` / `rerank_pointwise_fanout`（Pointwise `Noul` 基线）
    - `rerank_oneshot_choice`（One-shot Setwise `Choice-30`）
    - `rerank_setwise_tournament`（两阶段分组锦标赛 `6×5 -> 12`）
    - `rerank_setwise_sliding_window`（滑动窗口 `listwise.likelihood`）
    - `rerank_setwise_heap`（$c$-叉最大堆 Top-$k$ 选优）
    - `rerank_coarse_to_fine`（Pointwise `Noul-30` 粗筛 $\to$ Setwise `Choice` 精排级联）
- [`run_experiment.py`](run_experiment.py)：CLERC 法律判例检索数据集命令行评测入口，支持 `--backend {anyjev,kev,gemini,all}` 与 `--num-queries N`。
- [`stark_rerank_lib.py`](stark_rerank_lib.py) & [`run_stark_experiment.py`](run_stark_experiment.py)：**STaRK-Prime 生物医药半结构化知识库问答** 免检索 Top-20 重排序评测库与命令行入口（直接读取 [`runs/stark-prime/test-10pct/baseline.json`](../../runs/stark-prime/test-10pct/baseline.json) 与 [`data/stark-prime/bm25/documents.sqlite`](../../data/stark-prime/bm25/documents.sqlite) 中 Top-20 命中的全部 115 道题，缓存持久化于 `notebooks/_scratch/stark_decision_cache.json`）。
- [`notebooks/setwise_decision_rerank.ipynb`](notebooks/setwise_decision_rerank.ipynb)：CLERC 法律判例数据集（Top-30）完整交互式实验 Notebook。
- [`notebooks/stark_prime_setwise_rerank.ipynb`](notebooks/stark_prime_setwise_rerank.ipynb)：**STaRK-Prime 数据集（Top-20 命中全部 115 题 & 280 题 `test-0.1` 折算）** 完整交互式实验 Notebook。
- [`REPORT.md`](REPORT.md)：涵盖 CLERC 与 STaRK-Prime 两大数据集、三款决策模型的实验详细分析报告。

## 快速运行

```bash
# 0. 确保本地 AnyJev (8100) 与 Kev-9B (8009) 容器已启动
docker compose -f infra/anyjev/docker-compose.yml up -d
docker compose -f infra/kev/docker-compose.yml up -d

# 1. 运行 CLERC Top-30 实验（结果自动缓存至 notebooks/_scratch/decision_cache.json）
experiments/setwise-decision-rerank/.venv/bin/python experiments/setwise-decision-rerank/run_experiment.py --backend all --num-queries 10

# 2. 运行 STaRK-Prime Top-20 (115 题) 实验（结果自动缓存至 notebooks/_scratch/stark_decision_cache.json）
experiments/setwise-decision-rerank/.venv/bin/python experiments/setwise-decision-rerank/run_stark_experiment.py --backend all --num-queries 115
```
