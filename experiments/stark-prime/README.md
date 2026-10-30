# STaRK-Prime：BM25 工具与 pi 试跑

范围：本机 pi 1.2.0 + SGLang `qwen3.6-35b-a3b`（`http://127.0.0.1:30000/v1`）。
MCP 只提供 `search` 和 `read_document`；不使用 embedding，不提供图遍历。
这是验证链路的小规模试跑，不是正式 benchmark 或 STaRK 论文结果复现。

## 启动

在本目录执行。独立 Python 3.11 环境及依赖版本由 `pyproject.toml` 和 `uv.lock` 固定。

```bash
cd /home/lzx/projs/textual-view/experiments/stark-prime
make setup
make index  # 只需一次；已有索引时会拒绝覆盖
make test
make pilot
```

`make pilot` 固定从 Prime `val` 抽取 10 题（seed 20261030），并行 2 个 pi，
每题最多 180 秒、20 个 agent turns。每次创建新的 `runs/stark-prime/<UTC时间>/`。
问题独立进程、配置、工作目录和会话；标准答案留在父进程，工具不提供答案访问。
运行目录位于仓库外的 `/tmp/stark-prime/`，最终复制回 `runs/` 保存。
这些配置/工作目录隔离不是操作系统文件访问沙箱。

单题可用：

```bash
.venv/bin/python run.py \
  --data-root ../../references/datasets/stark \
  --index ../../data/stark-prime/bm25 \
  --output /tmp/stark-prime/manual-query-11163 \
  --split val --query-id 11163 --timeout 180 --max-turns 20
```

## pi 配置与手动交互

`config/pi/` 保存 `models.json`、`settings.json`、`mcp.json` 和 `APPEND_SYSTEM.md`。
运行程序为每题复制前两份配置和附加指令，按实际 Python、服务代码、索引路径生成 MCP 配置。
模板的 MCP 绝对路径对应当前 checkout；搬迁时修改模板，运行程序会自动使用自己的位置。
保留 pi 默认 system prompt，附加指令只约束最终答案格式。只开放两个 MCP 工具。

```bash
prime_config_dir=/home/lzx/projs/textual-view/experiments/stark-prime/config/pi
mkdir -p /tmp/stark-prime
prime_run_dir=$(mktemp -d /tmp/stark-prime/manual.XXXXXX)
mkdir -p "$prime_run_dir/agent" "$prime_run_dir/work" "$prime_run_dir/sessions"
cp "$prime_config_dir/"* "$prime_run_dir/agent/"
cd "$prime_run_dir/work"
PI_CODING_AGENT_DIR="$prime_run_dir/agent" \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
pi --session-dir "$prime_run_dir/sessions" --tools 'mcp__stark_prime__*'
```

## 检索语义与来源

`retrieval.py` 直接读取本地 `skb/prime/processed.zip` 的节点数据，不联网、不解压全套图。
Prime 所有十种节点类型均为候选，共 129,375 个。
文档按官方 `PrimeSKB.get_doc_info(add_rel=False, compact=False)` 和 `clean_dict` 生成。
保留官方 `bm25s.BM25()` / `bm25s.tokenize()` 默认值、全候选集、top-k 排序及原始节点 ID。
`search(query, k=100)` 返回 ID、得分、名称、类型；原文由 `read_document` 分页读取。
无跨查询得分归一化、重排、关系拼接或语义扩写。

与下载的官方代码相比，唯一的 BM25 兼容性修正：不向 BM25 构造器传入文本 corpus，
让 `retrieve` 返回位置索引，再映射到 candidate ID。官方代码传入文本 corpus 后，
当前 bm25s 会返回文本，直接把它当位置会失败。索引 manifest 记录源数据/源码哈希、
上游 commit、参数、依赖版本、文档摘要、构建时间。索引约 165 MiB。

测试对照官方源码抽查 1,000 个节点及每种类型代表，验证文档一致；对照 DKGI 验证评分，
覆盖重复/无效 ID、非法输出、末轮工具调用和 `agent_end` 后重试再 `agent_settled` 的场景。
来源与许可见 `NOTICE`。

## 运行与评分记录

```text
runs/stark-prime/<run>/
├── selection.json             # 固定抽样和预算
├── baseline.json              # 同题原问题调用 BM25 一次，top-100
├── summary.json               # 宏平均；失败题计零
└── <query_id>/attempt_001/
    ├── agent/                 # 实际使用的 pi 配置
    ├── work/                  # 工作目录
    ├── sessions/              # pi 原生会话
    ├── prompt.txt, task.json
    ├── command.json, initial_state.json
    ├── events.jsonl           # 原始 RPC 事件，持续消费并落盘
    ├── stderr.txt, state.json
    ├── final.txt              # 最后一条 assistant 消息的原文
    ├── prediction.json        # 提取后的标准 JSON
    ├── metrics.json           # 父进程独立评分
    └── result.json            # 状态、耗时、usage、工具次数、版本与哈希
```

pi 1.2.0 的 RPC 等待 `agent_settled`，不在可重试的 `agent_end` 时提前结束。
最终答案仅取最后一条完整 assistant 消息的 text，不拼接整个会话的流式片段，
不把思考文本、工具调用、错误或截断回复当成答案。

模型偶尔在 JSON 前添加解释。保留原文，并只在末条消息里有唯一一个
`{"node_ids": [...]}` 对象时提取；多个答案对象或没有对象均失败。
`format_compliant` 独立记录是否严格遵守纯 JSON 格式，不调用模型修复，不修正 ID。
这比 DKGI 的纯 JSON 解析宽松，属于明确披露的输出包装处理差异。

评分遵循 DKGI 的排名列表语义，独立于 agent：Hit@k、Recall@k、MRR@100。
重复/非法 ID 占据其原排名并计不命中，不会删除后把后面的答案前移。
标准答案的分母不剔除不可达候选；单独报告 candidate coverage。
此处未直接调用官方 `Evaluator`，它对未返回候选补低分的处理与截断列表并不完全相同。
缺少格式或运行失败计零；空列表是合法预测。不会因低分或空答案重跑。

无需再调用模型即可重算单题：

```bash
.venv/bin/python score.py \
  --attempt ../../runs/stark-prime/<run>/<query_id>/attempt_001 \
  --data-root ../../references/datasets/stark
```

usage 来自模型报告的 assistant 消息；未报告的失败请求不推算费用。
图关系未进入文本索引，很多 Prime 问题的关系条件无法通过当前工具直接验证。
保留负面结果，不能把此试跑解释为 pi 或整个 Prime 知识库的能力上限。
