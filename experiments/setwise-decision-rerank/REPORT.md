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
2. **同上下文横向比较 vs. 跨上下文独立校准**：法律检索的 Top-30 初筛列表中往往存在多段讨论同一法律原则但并非被引用判例的强干扰项。Setwise 将候选集合置于同一上下文中强制横向竞争（概率和为 $1$），有望解决 Pointwise 孤立打分难以区分细微法理差异的问题。
3. **共享 Query 前缀减少重复 Token 消耗**：Pointwise 独立调用需要将长 Query 文本重复传输 $N$ 次；而 Setwise 在单次或少量分组调用中共享 Query，可显著降低输入 Token 开销。

---

## 2. 实验设置（Experimental Setup）

### 2.1 数据集与基线构建
完全复用 [`cookbooks/rerank_typesafe`](../../references/repos/typesafe-public-examples/cookbooks/rerank_typesafe/rerank_typesafe.py) 的 **CLERC 联邦法院判例检索数据集**切片（来自 `rerank_typesafe_cache.json` 中的 `build_slice|170|40|seed=0`）：
- **候选语料库规模**：$3,565$ 个法院判决书段落（170 条记录汇总去重）。
- **查询集**：共 $40$ 条查询（隐去某处判例引用的判决书摘录，平均长度约 $1,300\sim 1,600$ 字符），每条查询对应唯一一个标准答案段落（Gold passage）。
- **第一阶段检索（Fast Search）**：使用 `bm25s`（英文停用词过滤）在 $3,565$ 个段落中为每条查询检索出 **Top-30 候选列表（Shortlist）**。在全部 40 条查询中，BM25 Top-30 的召回率为 **100%**（所有 Gold 均在 Top-30 内），但 Top-1 准确率仅为 **5.0%**（前 10 条评测子集上 BM25 Top-1 为 **20.0%**，Top-5 为 **30.0%**，Top-10 为 **50.0%**）。

### 2.2 被测决策模型（Decision Model Backends）
实验同时在两款不同架构与规模的 Decision Model 上进行：

| 属性 | Local AnyJev (`Tacit-9B`) | Google Gemini Decision 1 (`gemini-decision-1`) |
| :--- | :--- | :--- |
| **部署与访问方式** | 本地 Docker 容器（[`infra/anyjev/docker-compose.yml`](../../infra/anyjev/docker-compose.yml)，`vLLM 0.21.0` + `anyjev.serve`），端点 `http://127.0.0.1:8100/v1/decide`（代码显式无代理直连） | 云端 API `https://generativelanguage.googleapis.com/v1beta/decisions`，经本地 `http://127.0.0.1:10809` 代理访问 |
| **模型底座与机制** | 基于 `Qwen3.5-9B` 自蒸馏的 `Tacit-9B`，单次前向（`one_forward`）读取末尾位置的标签 token log-softmax | Google 原生 System One 决策模型，单次预填充（`total_output_tokens = 0`）返回校准概率 |
| **单次 `Choice` 选项上限** | 最多 **26 个选项**（受限于单字符字母标签 `A..Z` 及 vLLM `top_logprobs=20`） | 最多 **255 个选项**（`criteria` 字典映射） |
| **单次请求批量支持** | 支持 `{"items": [...]}` 批量传入多个独立决策，由 vLLM 并行调度 | 支持在 `questions` 字典中单次并发挂载最多 1,000 个问题（Speculative Fan-out） |

### 2.3 对比的方法与算子设计
我们在两款模型上实现了 6 类重排序范式（详见 [`rerank_lib.py`](rerank_lib.py)）：

1. **Pointwise Noul 基线**：
   - **AnyJev `Pointwise Noul (1 batch x 30)`**：将 30 个 `(Query, Candidate)` 对打包进 1 次 `/v1/decide` 的 `items` 批量请求（30 个 `yes_no` 决策），按 `P(Yes)` 降序排列。
   - **Gemini `Pointwise Noul Fan-out (1 call x 30 Nouls)`**：利用 Speculative Fan-out 在 1 次 API 调用的 `questions` 中挂载 30 个 `noul` 问题（每个问题的 `instructions` 包含对应候选段落，`criteria` 包含 `true_description` 和 `false_description`），按返回的 `noul` 值降序排列。
2. **One-shot Setwise `Choice-30`（仅 Gemini 支持，因 $N=30 > 26$）**：
   - **`criteria` 模式**：`input` 仅放 Query，30 个候选段落作为 `choice` 问题 `criteria: {cid: passage_text}` 的键值直接传入，1 次请求直接读取 30 个候选的概率分布排序。
   - **`tagged` 模式**：将 30 个候选以 `[cid] passage_text` 拼接在 `input` 中，`criteria` 仅放置短标签说明 `Passage [cid]`。
3. **Setwise Tournament（两阶段分组锦标赛，`6×5 -> 12`）**：
   - **第一阶段（初赛）**：将 BM25 Top-30 按序切分为 6 个小组，每组 $c=5$ 篇文档，每组运行 1 个 `Choice-5` 选出概率最高的前 2 名（共 12 名进入决赛）。在 AnyJev 上，6 个小组被打包进 **1 次批量 HTTP 请求**并行完成。
   - **第二阶段（决赛）**：对 12 名决赛候选运行 1 次 `Choice-12` 得到前 12 名精确排序，其余 18 篇按初赛小组内概率排在后面。
   - 同时对比 **`tagged` 模式**（段落放 `state`、选项用 `[P01]..[P05]`）与 **`inline` 模式**（段落直接塞进选项行 `A. [cid] passage...`）。
4. **Setwise Sliding Window（滑动窗口 `listwise.likelihood`，`w=6, s=4, 1 pass`）**：
   - 复现 Zhuang et al. (2024) §3.3 的滑动窗口机制：窗口大小 $w=6$、步长 $s=4$，从第 30 名向第 1 名滑动（共 7 个窗口），每次用 `Choice-6` 的输出概率对窗口内 6 篇候选原地重排，使强候选逐窗上浮。
5. **Setwise Heap Sort（$c$-叉最大堆排序，`c=4, top-10`）**：
   - 复现 Zhuang et al. (2024) §3.2 的最大堆算法：每个内部节点设 `num_child = 3` 个子节点（单次比较最多 $c=4$ 篇文档），每次仅使用 `Choice` 的赢家（$\arg\max$）执行 `heapify` 下沉调整；先自底向上建堆，再连续弹出 `top_k = 10` 个根节点得到 Top-10。
6. **Coarse-to-Fine 级联（`Pointwise Noul-30 -> Setwise Choice-10`）**：
   - 借鉴 [`cookbooks/infinite_cardinality_choice`](../../references/repos/typesafe-public-examples/cookbooks-zh-en/cookbooks/infinite_cardinality_choice.md) 的多级架构：第一阶段用 Pointwise `Noul` 快速粗筛出得分最高的前 10 篇候选；第二阶段用 1 次 Setwise `Choice-10` 让这 10 篇最容易混淆的头部候选同台对比，重排 Top-10。

---

## 3. 实验结果与定量分析

### 3.1 总体性能与成本对照表（前 10 条 CLERC 查询）

| 后端模型 | 重排序方法 | Top-1 | Top-5 | Top-10 | NDCG@10 | MRR | Mean Rank | HTTP Calls/q | Decisions/q | Input Tokens/q | Latency/q |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | **BM25 Fast Search (Top-30)** | 20.0% | 30.0% | 50.0% | 0.329 | 0.303 | 12.0 | 0.0 | 0.0 | — | — |
| **AnyJev (`Tacit-9B`)** | **Pointwise Noul (batch 30)** | 20.0% | **70.0%** | 70.0% | **0.454** | **0.392** | **6.8** | **1.0** | 30.0 | — | 2.42s |
| **AnyJev (`Tacit-9B`)** | **Setwise Tournament (`6×5→12`, tagged)** | 20.0% | 50.0% | 70.0% | 0.418 | 0.352 | 7.2 | 2.0 | **7.0** | — | **2.07s** |
| **AnyJev (`Tacit-9B`)** | **Setwise Tournament (`6×5→12`, inline)** | 20.0% | 40.0% | 60.0% | 0.351 | 0.300 | 9.8 | 2.0 | **7.0** | — | 2.09s |
| **AnyJev (`Tacit-9B`)** | **Setwise Sliding Window (`w=6, s=4`)** | 20.0% | 50.0% | **80.0%** | 0.428 | 0.327 | 7.8 | 7.0 | **7.0** | — | 2.17s |
| **AnyJev (`Tacit-9B`)** | **Setwise Heap Sort (`c=4, top-10`)** | **30.0%** | 50.0% | 60.0% | 0.417 | 0.391 | 8.3 | 37.0 | 37.0 | — | 8.20s |
| **AnyJev (`Tacit-9B`)** | **Coarse-to-Fine (`Noul-30 → Choice-10`)** | **30.0%** | 50.0% | 70.0% | 0.447 | **0.392** | 7.6 | 2.0 | 31.0 | — | 2.90s |
| **Gemini Decision 1** | **Pointwise Noul Fan-out (`1×30 Nouls`)** | **70.0%** | **90.0%** | 90.0% | 0.813 | **0.792** | **2.3** | **1.0** | 30.0 | 22,310 | 3.51s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-30` (`criteria`)** | **70.0%** | **90.0%** | **100.0%** | 0.831 | 0.780 | 2.4 | **1.0** | **1.0** | **17,852** | 6.40s |
| **Gemini Decision 1** | **One-shot Setwise `Choice-30` (`tagged`)** | **70.0%** | **90.0%** | **100.0%** | **0.835** | 0.785 | **2.3** | **1.0** | **1.0** | 18,894 | **3.40s** |
| **Gemini Decision 1** | **Setwise Tournament (`6×5→12`, criteria)** | **70.0%** | **90.0%** | 90.0% | 0.813 | 0.789 | 2.9 | 7.0 | 7.0 | 27,718 | 14.37s |
| **Gemini Decision 1** | **Coarse-to-Fine (`Noul-30 → Choice-10`)** | **70.0%** | 80.0% | 90.0% | 0.792 | 0.769 | 3.0 | 2.0 | 31.0 | 28,620 | 5.68s |

---

### 3.2 逐条查询的 Gold 排名轨迹表（Per-Query Gold Passage Rank, 1 为最佳）

为了观察不同排序机制在具体样本上的行为，下表列出了全部 10 条评测查询在每种方法下的标准答案排名（`#1`~`#30`）：

| Query ID | BM25 | AJ Pointwise | AJ Tourn (`tagged`) | AJ Tourn (`inline`) | AJ Slide (`w=6`) | AJ Heap (`c=4`) | AJ Coarse-to-Fine | GM Pointwise Fan-out | GM `Choice-30` (`criteria`) | GM `Choice-30` (`tagged`) | GM Tourn (`6×5→12`) | GM Coarse-to-Fine |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `392149` | 21 | **1** | 2 | 6 | 4 | **1** | **1** | **1** | **1** | **1** | **1** | **1** |
| `225090` | **1** | 2 | **1** | **1** | 3 | **1** | 4 | **1** | **1** | 2 | **1** | **1** |
| `25747` | 6 | 13 | 6 | 14 | 8 | 11 | 13 | **1** | **1** | **1** | **1** | **1** |
| `147736` | 21 | 20 | 3 | 11 | 20 | 12 | 20 | **1** | **1** | **1** | **1** | **1** |
| `451489` | 2 | 3 | 18 | 23 | 5 | 12 | 6 | 2 | 10 | 10 | 17 | 10 |
| `803305` | 9 | 2 | 14 | 4 | 8 | 4 | 5 | **1** | **1** | **1** | **1** | **1** |
| `648463` | 18 | 5 | 9 | 7 | 7 | 5 | 9 | **1** | 2 | **1** | **1** | 2 |
| `820752` | **1** | **1** | **1** | **1** | **1** | **1** | **1** | 3 | **1** | **1** | **1** | **1** |
| `604477` | 20 | 16 | 13 | 26 | 21 | 30 | 16 | 11 | 5 | 4 | 3 | 11 |
| `238002` | 21 | 5 | 5 | 5 | **1** | 6 | **1** | **1** | **1** | **1** | 2 | **1** |

---

## 4. 关键实验发现与深度剖析

### 发现 1：在 `gemini-decision-1` 上，One-shot Setwise `Choice-30` 实现了“更省 Token + 更高 Top-10 召回 + 单次请求”的全面占优
- **对比原 Cookbook 的 30 次独立 Pointwise 调用**：原 Cookbook 对每条 Query 发起 30 次独立的 `Noul` 请求，Query 文本和指令被重复计费 30 次（每条 Query 耗费约 `32,000+` input tokens）。
- **对比 Speculative Fan-out（1 次请求挂载 30 个 `Noul`）**：即便利用 Decision API 的多问题扇出能力将 30 个 `Noul` 合并到 1 次请求中，每个 `Noul` 问题内部仍需嵌入各自候选文本并重复判断准则，平均每条 Query 消耗 **`22,310` tokens**。
- **One-shot Setwise `Choice-30` 的优势**：将 30 个候选直接放入单个 `Choice` 问题的 `criteria`（或 `input` 标签块）中，**每次 Query 仅需 1 次 API 调用、1 个 `Choice` 决策**，输入 Token 降至 **`17,852` tokens**（相比 Fan-out 节省 **20.0%**，相比独立调用节省 **44%+**）。
- **排序质量**：在保持 **70.0% Top-1** 和 **90.0% Top-5** 的同时，Setwise `Choice-30` 将 **Top-10 准确率提升到了 100.0%**（NDCG@10 从 `0.813` 提升至 `0.835`）。特别是对于困难查询 `604477`（BM25 排第 20 名），Pointwise `Noul` 仅能将其排到第 11 名（掉出 Top-10），而所有的纯 Setwise 方法（`Choice-30 criteria` #5、`Choice-30 tagged` #4、`Tournament` #3）全部通过横向对比将其成功推入 **Top-5**；对于 `820752`，Pointwise `Noul` 被另外两篇高分干扰项误导将 Gold 排到 #3，而 Setwise `Choice-30` 稳稳将其排在 **#1**。

### 发现 2：`gemini-decision-1` 的 Setwise `Choice` 具备严格的位置排列不变性（Permutation Invariance）
在 [2024-Setwise 论文精读](../../references/papers/2024-Setwise/analysis.md#L244-L250) 中我们指出，普通 Transformer 的自回归提示并不具备数学上的集合排列不变性。为此，我们在 Notebook 第 5 节对困难查询 `392149`（BM25 Gold 初始排名 #21）进行了三种输入顺序置换实验：

| 候选输入顺序 | Gold 在输入列表中的初始位置 | AnyJev Tournament (`tagged`) Gold 最终排名 | Gemini `Choice-30` (`criteria`) Gold 最终排名 | Gemini `Choice-30` 赋予 Gold 的概率 | Gemini Top-1 预测段落 ID |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **BM25 正序（Original）** | #21 / 30 | **#2** | **#1** | **0.513** | `e38b0f0c30dbd0e6` (Gold) |
| **BM25 逆序（Reversed）** | #10 / 30 | **#1** | **#1** | **0.513** | `e38b0f0c30dbd0e6` (Gold) |
| **随机打乱（Seed 42）** | #1 / 30 | **#1** | **#1** | **0.513** | `e38b0f0c30dbd0e6` (Gold) |

- **观察**：无论 30 个候选以何种顺序传入 `criteria`，`gemini-decision-1` 不仅每次都将 Gold 排在 **#1**，而且输出的概率值精确锁定在 **`0.513`**（小数点后三位零波动）。这表明 Google 的 `gemini-decision-1` 在底层实现 `choice` 原语时，对各 `criteria` 选项采用了位置对称/旋转消偏的读出机制，使其在经验上表现为一个**真正的集合算子（True Set Operator）**。
- 相比之下，本地 `AnyJev (Tacit-9B)` 在 `anyjev.serve` 默认运行于 `one_forward` 模式（只按给定顺序读一次 prompt，未开启 `L0` 循环旋转），因此在正序下排在 #2，逆序与随机顺序下排在 #1，存在轻微的位置与分组边界敏感性。

### 发现 3：在本地 `AnyJev (Tacit-9B)` 上，提示结构（`tagged` vs. `inline`）对 Setwise 质量起决定性作用
- 在 AnyJev 中，`readout.py` 会把 `Choice` 渲染为：
  ```text
  State:
  <state_text>

  Question: <question_text>
  Options:
  A. <option_1>
  B. <option_2>
  ...
  Answer with the letter only.
  ```
- 如果采用 **`inline` 模式**（直接把数百词的法院判例文本作为 `<option_i>`），`Options:` 区域会膨胀到数千 tokens，标签字母 `A, B, C...` 之间相隔太远，导致 9B 模型在末尾读取字母 token logits 时严重失焦：其 Tournament NDCG@10 仅为 **`0.351`**，MRR 降至 `0.300`（甚至不如 BM25 的 `0.303`）。
- 当改为 **`tagged` 模式**（将带编号的候选段落 `[P01] ... [P05]` 统一放在 `State:` 中，而 `Options:` 仅列出紧凑的符号 `A. P01`, `B. P02`...）时，模型可以在 `State` 编码阶段充分进行跨段落注意力交互，并在末尾聚焦于紧凑的短选项标签：NDCG@10 从 `0.351` 大幅跃升至 **`0.418`**，Mean Rank 从 `9.8` 改善至 **`7.2`**。

### 发现 4：9B 开源决策模型上的“Pointwise 粗筛 + Setwise 精排”互补效应
在 `AnyJev (Tacit-9B)` 上对比不同排序机制可以发现一个非常清晰的互补规律：
- **Pointwise `Noul` 擅长中段召回（Top-5），但缺乏 Top-1 决胜能力**：AnyJev 的 `Pointwise Noul` 将 Top-5 从 BM25 的 30% 大幅提升至 **70.0%**（Mean Rank `6.8`），但在 10 条查询中只有 2 条能排到第 1 名（Top-1 **20.0%**）。例如在 `238002`（BM25 #21）上，Pointwise Noul 把 Gold 提到了 #5，却无法击败前 4 个同样与主题高度相关的干扰判例。
- **Setwise 算子擅长在头部干扰项中决出 Top-1**：
  - **Setwise Heap Sort ($c=4$)** 与 **Coarse-to-Fine (`Noul-30 -> Choice-10`)** 均将 AnyJev 的 Top-1 准确率从 20.0% 提升至 **30.0%**；
  - 在 `238002` 上，先用 `Pointwise Noul` 选出 Top-10（此时 Gold 位于 #5），紧接着用 1 次 `Choice-10` 让这 10 篇候选同台做横向 Setwise 比较，AnyJev 立即将 Gold 从 **#5 直接推上了 #1**！
  - 在 `147736`（BM25 #21）上，Pointwise Noul 完全失效（排在 #20），而 **Setwise Tournament (`6×5→12`, tagged)** 通过 5 篇一组的局部相对比较，直接将 Gold 推到了 **#3**。

---

## 5. 局限性与下一步探索建议

1. **样本规模与统计显著性**：当前实验为快速原型验证，评测集为 CLERC 切片的前 10 条查询（300 个查询-文档对）。代码与缓存机制已完全支持全量 40 条查询（只需运行 `run_experiment.py --num-queries 40` 或在 Notebook 中将 `N_EVAL_QUERIES` 设为 `40`）。
2. **AnyJev 的旋转消偏（`L0` Rotation）与 Adaptive Thinking**：当前 `infra/anyjev/docker-compose.yml` 启动的网关使用 `Tacit` 默认的 `one_forward` 模式（未开启 `--adaptive` 或 `Decider(level="L0")` 的多排列旋转平均）。下一步可以在 `Coarse-to-Fine` 的第二阶段（仅需 1 次 `Choice-10` 决策）引入 **正反双排列平均（2-rotation L0）** 或开启 `--adaptive`，检验能否在不显著增加总延迟的前提下消除 9B 模型在 `Choice-10` 终局决策上的位置偏差。
3. **结合绝对门槛（`Choice` + `Noul` 联合校准）**：在 `451489` 上，Pointwise `Noul` 将 Gold 排在 #2/#3，而 Setwise `Choice` 将其排到了 #10。观察发现，当某个候选在单点法则匹配度很高、但在全局多选一中概率被平摊时，将 **Pointwise `Noul` 对数几率的先验得分** 与 **Setwise `Choice` 的横向竞争概率** 做加权融合（例如 $\text{Score}(d_i) = \alpha \cdot \text{Noul}(d_i) + (1-\alpha) \cdot \text{Choice}(d_i \mid S_{\text{top-}m})$），有望同时兼顾 Pointwise 的高 Top-5 稳健性与 Setwise 的高 Top-1 区分度。
