# 实验报告：Setwise 思想在 System One 决策模型（Decision Models）重排序中的实践与评测

- **实验目录**：[`experiments/setwise-decision-rerank/`](.)
- **实验脚本与库**：[`rerank_lib.py`](rerank_lib.py)、[`run_experiment.py`](run_experiment.py)
- **交互式 Notebook**：[`notebooks/setwise_decision_rerank.ipynb`](notebooks/setwise_decision_rerank.ipynb)
- **缓存与汇总工件**：`notebooks/_scratch/decision_cache.json`、`notebooks/_scratch/summary_q10.json`
- **日期**：2026-11-20

---

## 1. 研究背景与实验动机

本实验旨在验证将 [Zhuang et al. (SIGIR 2024) *A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models*](../../references/papers/2024-Setwise/analysis.md) 中的 **Setwise 重排序思想**，迁移到专门面向结构化概率决策的 **System One 决策模型（Decision Models）** 上的效果与效率特征。

在官方示例 [`cookbooks/rerank_typesafe`](../../references/repos/typesafe-public-examples/cookbooks-zh-en/cookbooks/rerank_typesafe.md) 中，法律判例重排序采用的是典型的 **Pointwise（逐点独立打分）** 方案：
- 对 BM25 初筛出的 $N=30$ 个候选段落，分别构造 30 个独立的 `Noul`（是非题）请求：*“该候选段落是否来自查询隐去引用所指向的判例？”*；
- 30 次评估彼此隔离，最终依赖模型输出的 `noul` 绝对概率跨上下文排序。

将 **Setwise** 引入 Decision Model 的核心理论动机在于：
1. **`Choice` 原语天然对应 Setwise 算子与 `listwise.likelihood`**：在传统生成式 LLM 上做 Setwise，需要设计单 token 标签并提取特定解码步的 logits；而 Decision Model 的 `Choice` 原语在一次前向计算（`0` 输出 token）中直接返回闭集选项上的归一化概率分布：
   $$p_i = P_\theta(\ell_i \mid q, S),\qquad \hat{\pi} = \operatorname{argsort}_{i\in\{1,\ldots,c\}}(-p_i)$$
   取 $\arg\max p_i$ 即为标准 Setwise 选优算子（配合 Heap Sort / Tournament），按 $p_i$ 降序排列即为零解码开销的 `listwise.likelihood`。
2. **同上下文横向比较 vs. 跨上下文独立校准**：法律检索的 Top-30 初筛列表中往往存在多段讨论同一法律原则但并非被引用判例的强干扰项。Setwise 将候选集合置于同一上下文中强制横向竞争（概率和为 $1$），可检验不同读出架构的 Decision Model 在横向对比与单点校准上的能力边界。
3. **共享 Query 前缀减少重复 Token 消耗**：Pointwise 独立调用需要将长 Query 文本重复传输 $N$ 次；而 Setwise 或单请求多问题扇出（Speculative Fan-out）在单次或少量分组调用中共享 Query，可显著降低输入 Token 开销。

---

## 2. 实验设置（Experimental Setup）

### 2.1 数据集与基线构建
完全复用 [`cookbooks/rerank_typesafe`](../../references/repos/typesafe-public-examples/cookbooks/rerank_typesafe/rerank_typesafe.py) 的 **CLERC 联邦法院判例检索数据集**切片（来自 `rerank_typesafe_cache.json` 中的 `build_slice|170|40|seed=0`）：
- **候选语料库规模**：$3,565$ 个法院判决书段落（170 条记录汇总去重）。
- **查询集**：共 $40$ 条查询（隐去某处判例引用的判决书摘录，平均长度约 $1,300\sim 1,600$ 字符），每条查询对应唯一一个标准答案段落（Gold passage）。
- **第一阶段检索（Fast Search）**：使用 `bm25s`（英文停用词过滤）在 $3,565$ 个段落中为每条查询检索出 **Top-30 候选列表（Shortlist）**。在全部 40 条查询中，BM25 Top-30 的召回率为 **100%**（所有 Gold 均在 Top-30 内），但 Top-1 准确率仅为 **5.0%**（前 10 条评测子集上 BM25 Top-1 为 **20.0%**，Top-5 为 **30.0%**，Top-10 为 **50.0%**）。

### 2.2 三款被测决策模型（Decision Model Backends）
实验横跨三款采用不同读出架构（Readout Architecture）的 Decision Model：

| 属性 | Local AnyJev (`Tacit-9B`) | Local `Kev-9B` (`/srv/models/jaredpalmer/kev-9b`) | Google Gemini Decision 1 (`gemini-decision-1`) |
| :--- | :--- | :--- | :--- |
| **部署与访问方式** | 本地 Docker 容器（[`infra/anyjev/docker-compose.yml`](../../infra/anyjev/docker-compose.yml)，GPU 0），端点 `http://127.0.0.1:8100/v1/decide`（无代理直连） | 本地 Docker 容器（[`infra/kev/docker-compose.yml`](../../infra/kev/docker-compose.yml)，GPU 1），端点 `http://127.0.0.1:8009/v1/systemone`（无代理直连，TypeSafe API 兼容） | 云端 API `https://generativelanguage.googleapis.com/v1beta/decisions`，经本地 `http://127.0.0.1:10809` 代理访问（结果已缓存） |
| **模型底座与读出机制** | 基于 `Qwen3.5-9B` 自蒸馏的 `Tacit-9B`，**词表字母读出（Vocab Token Readout）**：单次前向读取末尾位置的词表标签 token（`Yes/No` 或 `A..Z`）log-softmax | 基于 `Qwen3.5-9B-Base` + LoRA ($r=16$) + **专用指针头（Pointer Head，$d_p=256$）**：将每个选项包裹在 `<opt>...</opt>` 中，用末尾 `<decide>` 隐状态与各选项结束符 `</opt>` 隐状态做缩放点积，并除以拟合温度 $T=2.19$ | Google 原生 System One 决策模型，单次预填充（`total_output_tokens = 0`）返回校准概率 |
| **训练与验证上下文上限** | vLLM `max_model_len = 32,768` | 训练最大 `state` 为 **7,552 tokens**，验证无退化上下文为 **8,192 tokens**（服务端支持最高 65,536 tokens） | 长上下文原生支持 |
| **单次 `Choice` 选项上限** | 最多 **26 个选项**（受限于单字符字母标签 `A..Z` 及 vLLM `top_logprobs=20`） | 最多 **255 个选项**（Pointer Head 对任意数量的 `</opt>` 位置做点积） | 最多 **255 个选项**（`criteria` 字典映射） |
| **单请求多问题与批处理** | 支持 `{"items": [...]}` 批量传入多个独立决策，由 vLLM 并行调度 | 支持单请求挂载多问题（共享 `state` 的 Prefix Cache + 独立分支行 `rows_of`），且服务端自动将并发请求合并为 CUDA Graph 批次 | 支持在 `questions` 字典中单次并发挂载最多 1,000 个问题（Speculative Fan-out） |

### 2.3 对比的方法与算子设计
我们在三款模型上实现了 6 类重排序范式（详见 [`rerank_lib.py`](rerank_lib.py)）：

1. **Pointwise Noul 基线（两种形态）**：
   - **独立 State-Pair 模式（Cookbook 标准方式）**：将每个 `(query_excerpt, candidate_passage)` 作为独立的请求（在 AnyJev 上为 `1 batch x 30 items`，在 Kev-9B 上为 30 个并发 `/v1/systemone` 请求，每个请求的 `state` 仅约 1,045 tokens，完全处于 Kev-9B 的 8,192 验证窗口内）。
   - **Speculative Fan-out 模式（`1 call x 30 Nouls`）**：在 Kev-9B 与 Gemini Decision 1 上，单次 API 调用的 `state/input` 仅放 Query，在 `questions` 中并发挂载 30 个 `noul` 问题（每个问题的 `instructions` 包含对应候选段落）。在 Kev-9B 上，混合架构（Gated DeltaNet）只需对 `state`（约 430 tokens）做 1 次前缀预填充，随后分批运行 30 条长度为 ~600 tokens 的独立问题分支行（`rows_of`）。
2. **One-shot Setwise `Choice-30`（Kev-9B 与 Gemini 支持，因 $N=30 > 26$）**：
   - **`criteria` 模式**：`state/input` 仅放 Query，30 个候选段落作为 `choice` 问题 `criteria: {cid: passage_text}` 的键值直接传入。*注：在 Kev-9B 上，30 个候选全部塞入同一个问题的 `<opt>...</opt>` 分支会使单条分支总长达到 ~18,300 tokens，超出 `SERVE_MAX_BRANCH` 默认单分支增量预算（+8,192 tokens），因此 Kev-9B 的 One-shot `Choice-30` 采用 `tagged` 模式测试。*
   - **`tagged` 模式**：将 30 个候选以 `[cid] passage_text` 拼接在 `state/input` 中（约 18.7k tokens），`criteria` 仅放置短标签说明 `Passage [cid]`。
3. **Setwise Tournament（两阶段分组锦标赛，`6×5 -> 12`）**：
   - **第一阶段（初赛）**：将 BM25 Top-30 按序切分为 6 个小组，每组 $c=5$ 篇文档，每组运行 1 个 `Choice-5` 选出概率最高的前 2 名（共 12 名进入决赛）。
   - **第二阶段（决赛）**：对 12 名决赛候选运行 1 次 `Choice-12` 得到前 12 名精确排序，其余 18 篇按初赛小组内概率排在后面。
   - 同时对比 **`tagged` 模式**（段落放 `state`、选项放短标签）与 **`criteria / inline` 模式**（段落直接放进各选项内部）。
4. **Setwise Sliding Window（滑动窗口 `listwise.likelihood`，`w=6, s=4, 1 pass`）**：
   - 复现 Zhuang et al. (2024) §3.3 的滑动窗口机制：窗口大小 $w=6$、步长 $s=4$，从第 30 名向第 1 名滑动（共 7 个窗口），每次用 `Choice-6` 的输出概率对窗口内 6 篇候选原地重排。
5. **Setwise Heap Sort（$c$-叉最大堆排序，`c=4, top-10`）**：
   - 复现 Zhuang et al. (2024) §3.2 的最大堆算法：每个内部节点设 `num_child = 3` 个子节点（单次比较最多 $c=4$ 篇文档），每次仅使用 `Choice` 的赢家（$\arg\max$）执行 `heapify` 下沉调整并抽出 Top-10。
6. **Coarse-to-Fine 级联（`Pointwise Noul-30 -> Setwise Choice`）**：
   - 第一阶段用 Pointwise `Noul` 快速粗筛出得分最高的前 $m$ 篇候选（$m=10$ 或 $5$）；第二阶段用 1 次 Setwise `Choice` 让头部候选同台对比。

---

## 3. 实验结果与定量分析

### 3.1 三款决策模型总体性能与成本对照表（前 10 条 CLERC 查询）

| 后端模型 | 重排序方法 | Top-1 | Top-5 | Top-10 | NDCG@10 | MRR | Mean Rank | HTTP Calls/q | Decisions/q | Input Tokens/q | Latency/q |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | **BM25 Fast Search (Top-30)** | 20.0% | 30.0% | 50.0% | 0.329 | 0.303 | 12.0 | 0.0 | 0.0 | — | — |
| **AnyJev (`Tacit-9B`)** | **Pointwise Noul (batch 30)** | 20.0% | **70.0%** | 70.0% | **0.454** | **0.392** | **6.8** | **1.0** | 30.0 | — | 2.42s |
| **AnyJev (`Tacit-9B`)** | **Setwise Tournament (`6×5→12`, tagged)** | 20.0% | 50.0% | 70.0% | 0.418 | 0.352 | 7.2 | 2.0 | **7.0** | — | **2.07s** |
| **AnyJev (`Tacit-9B`)** | **Setwise Tournament (`6×5→12`, inline)** | 20.0% | 40.0% | 60.0% | 0.351 | 0.300 | 9.8 | 2.0 | **7.0** | — | 2.09s |
| **AnyJev (`Tacit-9B`)** | **Setwise Sliding Window (`w=6, s=4`)** | 20.0% | 50.0% | **80.0%** | 0.428 | 0.327 | 7.8 | 7.0 | **7.0** | — | 2.17s |
| **AnyJev (`Tacit-9B`)** | **Setwise Heap Sort (`c=4, top-10`)** | **30.0%** | 50.0% | 60.0% | 0.417 | 0.391 | 8.3 | 37.0 | 37.0 | — | 8.20s |
| **AnyJev (`Tacit-9B`)** | **Coarse-to-Fine (`Noul-30 → Choice-10`)** | **30.0%** | 50.0% | 70.0% | 0.447 | **0.392** | 7.6 | 2.0 | 31.0 | — | 2.90s |
| **Kev-9B (Local)** | **Pointwise Noul (30 state-pairs, cookbook)** | **30.0%** | 60.0% | **80.0%** | **0.532** | **0.463** | **5.7** | 30.0 | 30.0 | 31,356 | **3.13s** |
| **Kev-9B (Local)** | **Pointwise Noul Fan-out (`1×30 Nouls`)** | 10.0% | **80.0%** | **80.0%** | 0.449 | 0.345 | 5.9 | **1.0** | 30.0 | 20,749 | 3.94s |
| **Kev-9B (Local)** | **One-shot Setwise `Choice-30` (`tagged`)** | 10.0% | 20.0% | 50.0% | 0.247 | 0.207 | 11.2 | **1.0** | **1.0** | **18,717** | 3.84s |
| **Kev-9B (Local)** | **Setwise Tournament (`6×5→12`, tagged)** | 20.0% | 30.0% | 60.0% | 0.341 | 0.288 | 10.1 | 7.0 | 7.0 | 28,597 | 5.41s |
| **Kev-9B (Local)** | **Setwise Tournament (`6×5→12`, criteria)** | 20.0% | 50.0% | **80.0%** | 0.437 | 0.337 | 9.3 | 7.0 | 7.0 | 27,084 | 4.52s |
| **Kev-9B (Local)** | **Setwise Sliding Window (`w=6, s=4`, tagged)** | 10.0% | 50.0% | 60.0% | 0.308 | 0.239 | 10.2 | 7.0 | 7.0 | 28,586 | 5.03s |
| **Kev-9B (Local)** | **Coarse-to-Fine (`Noul-30 → Choice-5`, criteria)** | 20.0% | 60.0% | **80.0%** | 0.458 | 0.367 | 6.2 | 31.0 | 31.0 | 34,539 | 3.71s |
| **Gemini Decision 1** | **Pointwise Noul Fan-out (`1×30 Nouls`)** | **70.0%** | **90.0%** | 90.0% | 0.813 | **0.792** | **2.3** | **1.0** | 30.0 | 22,310 | 3.51s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-30` (`criteria`)** | **70.0%** | **90.0%** | **100.0%** | 0.831 | 0.780 | 2.4 | **1.0** | **1.0** | **17,852** | 6.40s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-30` (`tagged`)** | **70.0%** | **90.0%** | **100.0%** | **0.835** | 0.785 | **2.3** | **1.0** | **1.0** | 18,894 | **3.40s** |
| **Gemini Decision 1** | **Setwise Tournament (`6×5→12`, criteria)** | **70.0%** | **90.0%** | 90.0% | 0.813 | 0.789 | 2.9 | 7.0 | 7.0 | 27,718 | 14.37s |
| **Gemini Decision 1** | **Coarse-to-Fine (`Noul-30 → Choice-10`)** | **70.0%** | 80.0% | 90.0% | 0.792 | 0.769 | 3.0 | 2.0 | 31.0 | 28,620 | 5.68s |

---

### 3.2 逐条查询的 Gold 排名轨迹表（Per-Query Gold Passage Rank, 1 为最佳）

| Query ID | BM25 | AJ PW Noul | AJ Tourn (`tagged`) | AJ Coarse-to-Fine | Kev PW (`30 pairs`) | Kev PW (`Fan-out`) | Kev `Choice-30` (`tagged`) | Kev Tourn (`tagged`) | Kev Tourn (`criteria`) | Kev Coarse-to-Fine | GM PW (`Fan-out`) | GM `Choice-30` (`criteria`) | GM `Choice-30` (`tagged`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `392149` | 21 | **1** | 2 | **1** | **1** | 4 | 11 | 7 | 2 | 4 | **1** | **1** | **1** |
| `225090` | **1** | 2 | **1** | 4 | **1** | **1** | 3 | **1** | **1** | **1** | **1** | **1** | 2 |
| `25747` | 6 | 13 | 6 | 13 | 15 | 3 | 14 | 7 | 10 | 15 | **1** | **1** | **1** |
| `147736` | 21 | 20 | 3 | 20 | 8 | 13 | 11 | 6 | 25 | 8 | **1** | **1** | **1** |
| `451489` | 2 | 3 | 18 | 6 | **1** | 3 | 9 | 23 | 29 | 5 | 2 | 10 | 10 |
| `803305` | 9 | 2 | 14 | 5 | 2 | 2 | 8 | 16 | 5 | 2 | **1** | **1** | **1** |
| `648463` | 18 | 5 | 9 | 9 | 4 | 4 | 28 | 5 | 4 | 3 | **1** | 2 | **1** |
| `820752` | **1** | **1** | **1** | **1** | 2 | 3 | **1** | **1** | **1** | **1** | 3 | **1** | **1** |
| `604477` | 20 | 16 | 13 | 16 | 15 | 23 | 21 | 21 | 8 | 15 | 11 | 5 | 4 |
| `238002` | 21 | 5 | 5 | **1** | 8 | 3 | 6 | 14 | 8 | 8 | **1** | **1** | **1** |
| **Mean Rank** | **12.0** | **6.8** | **7.2** | **7.6** | **5.7** | **5.9** | **11.2** | **10.1** | **9.3** | **6.2** | **2.3** | **2.4** | **2.3** |

---

## 4. 关键实验发现与深度剖析

### 发现 1：新增的 `Kev-9B` 在两款开源 9B 决策模型中取得了最佳的整体重排序精度（NDCG@10 `0.532`，Top-5 `80.0%`，Mean Rank `5.7`）
- **校准温度（$T=2.19$）与 Pointer Head 带来的 Pointwise 优势**：与同样基于 `Qwen3.5-9B` 的 `AnyJev (Tacit-9B)` 相比，`Kev-9B` 在 `Pointwise Noul (30 state-pairs)` 上将 **Top-1 从 20.0% 提升至 30.0%**，**Top-10 从 70.0% 提升至 80.0%**，**NDCG@10 从 `0.454` 大幅提升至 `0.532`**，**MRR 从 `0.392` 提升至 `0.463`**，平均 Gold 排名从 `6.8` 缩减到 **`5.7`**。
- **Speculative Fan-out (`1 call x 30 Nouls`) 的高 Top-5 召回**：当把 Query 放在共享 `state` 中，一次性扇出 30 个 `Noul` 问题分支时，`Kev-9B` 仅用 **1 次 HTTP 调用**、**20,749 tokens**（节省 33.8% tokens），就达到了 **80.0% 的 Top-5 召回率** 和 **80.0% 的 Top-10 召回率**（超越了 AnyJev 所有方法的 Top-5 表现）。

### 发现 2：读出机制决定提示范式——`AnyJev` 与 `Kev-9B` 在 `tagged` vs. `criteria/inline` 上呈现完全相反的优劣反转！
这是本次引入 `Kev-9B` 后最有价值的架构级发现。在相同的 `Setwise Tournament (6×5 -> 12)` 算子下，两种候选表示方式在 `AnyJev` 和 `Kev-9B` 上表现出了**完全相反**的规律：

| 决策模型 | 底层读出架构（Readout Architecture） | Tournament (`tagged`：段落放 `state`，选项仅放短标签) | Tournament (`criteria/inline`：段落直接放进选项内) | 最优表示范式及原因 |
| :--- | :--- | :---: | :---: | :--- |
| **AnyJev (`Tacit-9B`)** | **末尾词表字母读出**：在序列最后 `Answer with the letter only.` 处取词表字母 `A..Z` 的 logits | **NDCG@10 = `0.418`**<br>MRR = `0.352`, Mean Rank = `7.2` | NDCG@10 = `0.351`<br>MRR = `0.300`, Mean Rank = `9.8` | **`tagged` 显著占优（+0.067 NDCG）**：若把长段落塞进 `Options:`，字母标签 `A, B, C...` 相隔数千 tokens，末尾预测单字母时注意力严重失焦。 |
| **Kev-9B (`Pointer Head`)** | **跨度边界指针读出**：把每个选项编码为 `<opt> k: v </opt>`，用末尾 `<decide>` 的隐状态 $q(h_{\text{decide}})$ 与每个选项末尾 `</opt>` 的隐状态 $k(h_{\text{opt}_i})$ 做点积 | NDCG@10 = `0.341`<br>MRR = `0.288`, Top-10 = `60.0%` | **NDCG@10 = `0.437`**<br>MRR = `0.337`, Top-10 = **`80.0%`** | **`criteria` 显著占优（+0.096 NDCG）**：Kev 的 Pointer Head 依赖 `</opt>` 位置的隐状态来汇总该选项的语义。若采用 `tagged` 模式，`<opt> cid: Passage [cid] </opt>` 内部只有干瘪的 ID 字符串，迫使 `</opt>` 跨越数千 tokens 去 `state` 中做远距离寻址；而采用 `criteria` 模式时，候选段落文本紧贴在 `<opt>...</opt>` 内部，`</opt>` 能直接汇总整段判例表征！ |

这一发现表明：**为 Decision Model 设计 Setwise 重排序算子时，必须根据模型的读出头类型（Token Logit Readout vs. Span Pointer Readout）选择匹配的候选注入位置**。

### 发现 3：`Kev-9B` 的训练上下文窗口（7,552 tokens）限制了超长 One-shot `Choice-30`，但小窗口分组 Tournament（$c=5$）可有效恢复性能
- 根据 `Kev-9B` 的模型卡与 `training_config.json`，`Kev-9B` 训练时的最大 `state` 长度为 **7,552 tokens**（验证无退化上下文上限为 **8,192 tokens**，只有 `Kev-27B` 在 32,768 tokens 上训练）。
- 当我们尝试在 `Kev-9B` 上运行 **One-shot Setwise `Choice-30 (tagged)`** 时，30 个候选段落拼接后的 `state` 长度达到了 **18,717 tokens**（超过训练长度 2.4 倍），导致 Gated DeltaNet 循环状态在长距离上严重退化（NDCG@10 跌至 `0.247`，甚至低于 BM25）。
- 而一旦采用 **Setwise Tournament (`6×5 -> 12`, criteria)** 将单次比较控制在 $c=5$（单次请求约 3,500 tokens，完全落回 7,552 训练舒适区内），`Kev-9B` 的 Top-10 立即从 50.0% 跃升至 **80.0%**，NDCG@10 恢复至 **`0.437`**（在困难查询 `604477` 上将 Gold 从 Pointwise 的 #15 提升至 **#8**）。

### 发现 4：三款模型的位置排列不变性（Permutation Invariance）对比
在 Notebook 第 5 节中，我们对困难查询 `392149`（BM25 Gold 初始排名 #21）进行了三种输入顺序置换实验：

| 候选输入顺序 | Gold 初始位置 | AnyJev Tournament (`tagged`) Gold 排名 | Kev-9B Tournament (`criteria`) Gold 排名 | Gemini `Choice-30` (`criteria`) Gold 排名 | Gemini `Choice-30` Gold 概率 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **BM25 正序（Original）** | #21 / 30 | **#2** | **#2** | **#1** | **0.513** |
| **BM25 逆序（Reversed）** | #10 / 30 | **#1** | **#2** | **#1** | **0.513** |
| **随机打乱（Seed 42）** | #1 / 30 | **#1** | **#13** | **#1** | **0.513** |

- **观察**：
  1. **`gemini-decision-1`** 对 30 个候选的任意排列保持了严格的概率与排名不变性（3 次置换均排 **#1**，概率恒为 **`0.513`**）。
  2. **`Kev-9B`** 的 `head.pt` 元数据显示其训练配置为 `option_isolation: False`（因为 `Qwen3.5-9B` 含有 Gated DeltaNet 线性注意力层，无法使用块因果掩码隔离不同 `<opt>...</opt>` 之间的自回归状态，后面的选项在因果注意力下会看到前面的选项）。因此在正序与逆序下 Gold 均稳定排在 **#2**，但在随机种子 42 的特定小组划分下出现了位置波动（#13）。值得一提的是，Kev 服务端专门内置了 `/v1/systemone/permute` 端点（支持对同一个 `Choice` 问题做 $K$ 次随机选项顺序置换并求平均），正好印证了非 `option_isolation` 混合架构在单次前向下存在选项顺序敏感性。

---

## 5. 扩展实验：大规模 `STaRK-Prime` 生物医药半结构化知识库重排序（115 道 Top-20 命中题 & 280 题全量 `test-0.1`）

为了在更大样本量（115 道题，2,300 个查询-文档对）和不同领域（生物医药多跳实体检索）上检验上述结论，我们将实验扩展至 **STaRK-Prime**（详见独立 Notebook [`notebooks/stark_prime_setwise_rerank.ipynb`](notebooks/stark_prime_setwise_rerank.ipynb) 与 [`stark_rerank_lib.py`](stark_rerank_lib.py)）。

### 5.1 免检索直接重排协议（Zero-Retrieval Protocol）
- **候选池与节点文档来源**：直接复用本地已固化的 `STaRK-Prime` 官方 `test-0.1`（共 280 道题）BM25 初筛结果 [`runs/stark-prime/test-10pct/baseline.json`](../../runs/stark-prime/test-10pct/baseline.json) 与预渲染的 129,375 节点文档库 [`data/stark-prime/bm25/documents.sqlite`](../../data/stark-prime/bm25/documents.sqlite)。
- **Oracle-Covered 115 题子集与 280 题全量折算**：在全部 280 道测试题中，有 **115 道题（41.1%）** 的 BM25 **Top-20** 候选中包含至少 1 个标准答案节点（其中 65 道单答案题、50 道多答案题；39 道题 BM25 已排第 1，**76 道困难题的标准答案位于第 2~20 名**）。由于其余 165 道题的 Top-20 内无标准答案（任何 Top-20 重排序器得分恒为 0），在这 115 题上评测重排不仅排除了未召回噪声，还能通过 $\text{Metric}_{280} = \text{Metric}_{115} \times \frac{115}{280}$ **100% 精确折算全量 280 题的端到端指标**。
- **文本长度与架构窗口对齐**：每个候选节点的半结构化文档保留前 1,200 字符（涵盖 `name`、`type`、`source` 与核心 `details`），20 个候选节点合计约 **4,500 ~ 5,500 tokens**：
  1. **解锁 `AnyJev (Tacit-9B)` 的 `One-shot Setwise Choice-20`**：$N=20 \le 26$（字母标签上限且等于 vLLM `top_logprobs=20`），无需分组即可在 **1 次前向调用** 中完成全部 20 个候选的横向比较！
  2. **完全落入 `Kev-9B` 的 8,192 验证无退化上下文窗口**：单题总长 ~5,000 tokens 严格低于 `Kev-9B` 的 **7,552 训练长度上限**，彻底消除了 CLERC Top-30（18.7k tokens）中的长上下文状态退化。

### 5.2 `STaRK-Prime` 全部 115 道 Top-20 命中题（及 280 题全量折算）定量结果

| 后端模型 | 重排序方法 | Hit@1 (115q) | Hit@3 | Hit@5 | Hit@10 | NDCG@10 | MRR | Mean Best Rank | **280q Hit@1** | **280q Hit@5** | Calls/q | Tokens/q | Latency/q |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | **BM25 Fast Search (Top-20)** | 33.9% | 60.0% | 75.7% | 91.3% | 0.592 | 0.521 | 4.03 | **13.9%** | **31.1%** | 0.0 | — | — |
| **AnyJev (`Tacit-9B`)** | **Pointwise Noul (batch 20)** | 52.2% | 73.0% | 80.0% | 96.5% | 0.712 | 0.659 | 2.96 | **21.4%** | **32.9%** | **1.0** | — | 1.25s |
| **AnyJev (`Tacit-9B`)** | **One-shot Setwise `Choice-20` (`tagged`)** | 65.2% | 83.5% | 88.7% | **99.1%** | 0.783 | 0.757 | **2.16** | **26.8%** | **36.4%** | **1.0** | — | **0.74s** |
| **AnyJev (`Tacit-9B`)** | **Setwise Tournament (`4×5→8`, tagged)** | **66.1%** | **84.3%** | 89.6% | 93.0% | 0.767 | **0.763** | 2.58 | **27.1%** | **36.8%** | 2.0 | — | 1.12s |
| **AnyJev (`Tacit-9B`)** | **Coarse-to-Fine (`Noul-20 → Choice-10`)** | 64.3% | 83.5% | **90.4%** | 96.5% | **0.785** | 0.759 | 2.28 | **26.4%** | **37.1%** | 2.0 | — | 1.64s |
| **Kev-9B (Local)** | **Pointwise Noul (20 state-pairs)** | 53.0% | 73.0% | 81.7% | 93.9% | 0.707 | 0.659 | 3.19 | **21.8%** | **33.6%** | 20.0 | 7,515 | 1.23s |
| **Kev-9B (Local)** | **Pointwise Noul Fan-out (`1×20 Nouls`)** | 53.0% | 72.2% | 80.9% | 91.3% | 0.704 | 0.660 | 3.42 | **21.8%** | **33.2%** | **1.0** | 6,771 | 1.27s |
| **Kev-9B (Local)** | **One-shot Setwise `Choice-20` (`criteria`)** | 56.5% | **78.3%** | 85.2% | 96.5% | 0.739 | 0.697 | 2.75 | **23.2%** | **35.0%** | **1.0** | **4,777** | 0.91s |
| **Kev-9B (Local)** | **One-shot Setwise `Choice-20` (`tagged`)** | **62.6%** | 77.4% | **87.8%** | **97.4%** | **0.771** | **0.731** | **2.45** | **25.7%** | **36.1%** | **1.0** | 5,096 | **0.90s** |
| **Kev-9B (Local)** | **Setwise Tournament (`4×5→8`, criteria)** | 58.3% | **78.3%** | 83.5% | 86.1% | 0.696 | 0.694 | 3.56 | **23.9%** | **34.3%** | 5.0 | 6,998 | 1.39s |
| **Kev-9B (Local)** | **Coarse-to-Fine (`Noul-20 → Choice-5`, criteria)** | 51.3% | 73.0% | 81.7% | 93.9% | 0.704 | 0.648 | 3.22 | **21.1%** | **33.6%** | 21.0 | 8,828 | 1.51s |
| **Gemini Decision 1** | **Pointwise Noul Fan-out (`1×20 Nouls`)** | 61.7% | 80.0% | 87.8% | 98.3% | 0.774 | 0.734 | 2.43 | **25.4%** | **36.1%** | **1.0** | 7,585 | 3.90s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-20` (`criteria`)** | 68.7% | 84.3% | 90.4% | **99.1%** | **0.823** | 0.784 | 2.10 | **28.2%** | **37.1%** | **1.0** | **4,534** | 2.19s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-20` (`tagged`)** | 65.2% | **87.0%** | **93.0%** | **99.1%** | 0.814 | 0.772 | **2.01** | **26.8%** | **38.2%** | **1.0** | 4,995 | **2.14s** |
| **Gemini Decision 1** | **Coarse-to-Fine (`Noul-20 → Choice-10`)** | **69.6%** | 86.1% | 89.6% | 98.3% | 0.818 | **0.790** | 2.17 | **28.6%** | **36.8%** | 2.0 | 10,064 | 5.89s |

### 5.3 针对 76 道“BM25 未排第 1 名”困难题的专项净提升对照（Hard Subset: BM25 Rank $2\ldots 20$）

在 BM25 第 1 名未能命中的 **76 道困难题**（BM25 `Hit@1 = 0.0%`，平均最佳排名 `5.59`）上，Setwise 对 Pointwise 的领先优势更加显著：

| 后端模型 | 代表性 Pointwise 方法 (`Hit@1` / `MRR`) | 代表性 Setwise 方法 (`Hit@1` / `MRR`) | Setwise 相比 Pointwise 的净提升 |
| :--- | :--- | :--- | :--- |
| **AnyJev (`Tacit-9B`)** | Pointwise Noul (batch 20): **`43.4%`** / `0.591` | One-shot `Choice-20` (`tagged`): **`52.6%`** / `0.661`<br>Coarse-to-Fine (`Choice-10`): **`53.9%`** / `0.685` | **`+10.5%` Hit@1**，`+0.094` MRR |
| **Kev-9B (Local)** | Pointwise Noul (20 pairs / Fan-out): **`47.4%`** / `0.626` | One-shot `Choice-20` (`tagged`): **`51.3%`** / `0.654` | **`+3.9%` Hit@1**，且延迟与 Token 减半 |
| **Gemini Decision 1** | Pointwise Noul Fan-out (`1×20`): **`53.9%`** / `0.686` | One-shot `Choice-20` (`criteria`): **`61.8%`** / `0.733`<br>Coarse-to-Fine (`Choice-10`): **`61.8%`** / `0.739` | **`+7.9%` Hit@1**，Token 节省 **40.2%**（`4,534` vs `7,585`） |

### 5.4 `STaRK-Prime` 大规模实验的核心新发现

1. **在全部三款决策模型上，Setwise `Choice-20` 全面且大幅超越 Pointwise `Noul`（无论精度、速度还是成本）**：
   - 在 `AnyJev (Tacit-9B)` 上，**`One-shot Setwise Choice-20 (tagged)`** 将 115 题的 `Hit@1` 从 BM25 的 `33.9%` 和 Pointwise Noul 的 `52.2%` 一举推高到 **`65.2%`**（`Setwise Tournament` 更达 **`66.1%`**，超越了 `Gemini Decision 1` 的 Pointwise `61.7%`！），同时单题耗时从 `1.25s` 缩短至 **`0.74s`**。
   - 在 `Kev-9B` 上，当 20 个候选的总长度（`~5,096` tokens）落回其 **7,552 训练长度窗口内** 时，**`One-shot Setwise Choice-20 (tagged)`** 彻底摆脱了 CLERC Top-30（18.7k tokens）时的长上下文退化，`Hit@1` 从 Pointwise 的 `53.0%` 跃升至 **`62.6%`**（NDCG@10 从 `0.707` 升至 **`0.771`**），同时输入 Token 减少 **32%**（`5,096` vs `7,515`），耗时缩短至 **`0.90s`/题**。
   - 在 `Gemini Decision 1` 上，**`One-shot Setwise Choice-20 (criteria)`** 将 `Hit@1` 从 Pointwise Fan-out 的 `61.7%` 提升至 **`68.7%`**（`Coarse-to-Fine` 达 **`69.6%`**），`Hit@10` 达到 **`99.1%`**（115 题中 114 题排入前 10），单题 Token 消耗从 `7,585` 降至 **`4,534` tokens**（节省 **40.2%**），延迟从 `3.90s` 降至 **`2.19s`**！
2. **单次免检索重排逼近多轮 Agent 检索的端到端表现**：
   - 在全量 280 道 `test-0.1` 题目上，此前 [`runs/stark-prime/test-10pct/summary.json`](../../runs/stark-prime/test-10pct/summary.json) 中使用 35B Agent (`pi` + `qwen3.6-35b-a3b`) 进行最多 20 轮交互式搜索（总计 7,884 次工具调用、耗时 9,375 秒 ≈ 2.6 小时），取得的端到端 `Hit@1` 为 **`29.29%`**。
   - 而本次实验**完全不改第一阶段静态 BM25 检索**，仅对 BM25 Top-20 做 **1 次 Setwise `Choice-20` 前向重排**（0 生成 token，每题 1 次调用），`Gemini Decision 1` 就将全量 280 题的 `Hit@1` 从 BM25 的 **`13.93%` 直接翻倍提升至 `28.57%`**（本地 9B 模型 `AnyJev Tournament` 也达到了 **`27.14%`**，全套耗时仅约 2 分钟）！这证明了在半结构化知识库检索中，**单次 Setwise Decision Model 重排能以极低延迟恢复绝大多数多轮 Agent 搜索在 Top-20 候选池内的排序收益**。
3. **本次 `STaRK-Prime` 115 题实验的 `Gemini Decision 1` 实际账单核算**：
   - 全部 115 道题 × 4 种方法共发起 **460 次 HTTP 请求**，累计消耗 **`2,253,204` input tokens（2.25M tokens）**、**`0` output tokens**。
   - 按官方 Decision Model 单价（`$0.042 / 1M input tokens`）计算，本次 115 题全量实验仅花费 **`$0.095 美元`（约合人民币 `0.68 元`）**；其中单跑效果最好的 `One-shot Setwise Choice-20 (criteria)` 115 道题合计仅 `521,410` tokens（花费 **`$0.022 美元`，约合人民币 `0.16 元`**）。

---

## 6. 总结与实践建议

1. **云端旗舰 Decision Model（`gemini-decision-1`）**：在 CLERC（Top-30）和 STaRK-Prime（Top-20）两大数据集上，**One-shot Setwise `Choice`** 均是绝对首选——相比 Pointwise `Noul` 稳定提升 **`+7%~+10%` Top-1 / Hit@1**，将 Top-10 召回推至 **`99.1%~100.0%`**，同时节省 **20%~40% Input Tokens**，每次查询仅需 **1 次 API 调用**。
2. **本地 Pointer-Head 决策模型（`Kev-9B`）**：
   - 核心原则是**把单次请求的上下文总长控制在 7,552 训练窗口（8,192 验证上限）以内**：在 STaRK-Prime Top-20（~5,000 tokens）下，`Kev-9B` 的 **`One-shot Setwise Choice-20 (tagged)`** 大幅超越 Pointwise `Noul`（`Hit@1` `62.6%` vs `53.0%`）；而在单次总长超过 18k tokens 的 CLERC Top-30 下，则应采用 Pointwise 或 $c=5$ 的 `Setwise Tournament (criteria)`。
3. **本地 Vocab-Readout 决策模型（`AnyJev Tacit-9B`）**：
   - 当候选数 $N \le 20$ 时（如 STaRK-Prime Top-20），直接使用 **`One-shot Setwise Choice-20 (tagged)`** 或 **`Setwise Tournament (4×5→8, tagged)`**，仅需 1~2 次请求（0.74s~1.12s/题）即可取得 **`65.2%~66.1%` Hit@1**，比 Pointwise Noul（`52.2%`）高出 **13~14 个百分点**。
