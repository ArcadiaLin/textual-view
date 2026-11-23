# SemCEB × Gemini Decision 1：最小研究工程

目前是开发和联通验证阶段，不是冻结的正式实验。目标是先比较 DM 逐对抽样预测与状态聚合代理；没有参照配对标签，因此没有启用残差纠偏，也不把 DM 自己的判断称为 SemCEB 参照标签。

## 环境与数据

从仓库根目录执行 `uv sync --project experiments/semceb-dm`。独立环境及 `uv.lock` 固定依赖，不修改其他实验环境。根目录 `.env` 提供：

```dotenv
X-goog-api-key=YOUR_KEY
GEMINI_DECISION_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/decisions
GEMINI_DECISION_MODEL=gemini-decision-1
GEMINI_PROXY=http://127.0.0.1:10809
```

密钥只进入 HTTP 请求头，不进入缓存、配置快照或日志。当前客户端串行执行，45 秒超时，不自动重试；`--max-requests` 限制实际 HTTP 尝试次数。价格尚未核实，USD 明确记为 null，不用 0 代替未知费用。原始 usage、实时请求耗时、缓存命中分别记录；缓存重放时间不冒充实际模型推理时间。

数据直接使用 `references/datasets/semceb/amazon-reviews`。复用官方 DataLoader：固定 seed=42 洗牌产品后取前 sf 行；评论按 `r.asin` 匹配所选商品的 `parent_asin`。sf 不是两张表各取 sf 行。左右别名、顺序和自连接对角线均保留。

来源：<https://github.com/utndatasystems/SemCEB>，commit `fec6bde7d1b23fc751f8701872fafda777a63942`。代码拒绝不同 revision，记录实际依赖源码哈希。当前仅支持文本双输入连接；不引入新的单侧过滤。

## 命令

在本目录执行，输出路径须尚不存在：

```bash
# 单元测试：含全部 40 条连接在 sf10/sf100 的真值键匹配检查
.venv/bin/python -m pytest -q

# 无 API 调用的数据/查询/参照覆盖检查
.venv/bin/python -m semceb_dm.runner --query 63 --scale-factor 10 \
  --dry-run --output ../../runs/semceb-dm/q63-dry

# 开发 smoke：最多两次请求；100 个有序配对中均匀无放回抽两对
.venv/bin/python -m semceb_dm.runner --query 63 --scale-factor 10 \
  --samples 2 --max-requests 2 --output ../../runs/semceb-dm/q63-smoke

# 状态聚合示例：先 dry-run 核对请求量，再决定建设预算
.venv/bin/python -m semceb_dm.runner --query 63 --scale-factor 10 \
  --method state-proxy --state-config configs/q63-states.json \
  --dry-run --output ../../runs/semceb-dm/q63-state-dry
```

实际状态聚合命令需移除 `--dry-run` 并明确给定请求预算。示例有左右 10+10 次行级 choice 和 3×4 次状态对 noul，共 32 次冷请求；不把这些在线建设费用计作免费离线准备。Q63 状态是手工研究假设，不是准确兼容规则；other 状态仍参与矩阵预测。

## 方法与输出

- `pair-sample`：对随机配对调用 noul；主输出为 N×样本平均 DM 分数，同时记录以 0.5 阈值构成的硬标签估计。前者是抽样估计全域 DM 分数之和，不是已校准的真实基数，也不是官方 Extrapolation Sampling 的复现。
- `state-proxy`：按指定输入字段逐行 choice 获得状态向量，以 noul 构造兼容矩阵，再计算 `(Σq_left)ᵀ K (Σq_right)`。保留左右方向；只计算有限状态对，不扫描所有记录对。
- 默认整数接口使用非负 half-up 取整并限制在总体大小内；原始浮点估计保留。报告原始与整数 clamped Q-error。
- 评测器在估计结束后读取官方总基数，使用模型、系统提示词、数据别名、规模和 LOTUS 谓词的完整缓存键匹配。真值不进入估计器或模型提示。
- 残差纠偏和 score 尚未实现；待参照配对标签与真实 API 契约落实后再加入，不伪造标签或用 choice 冒充 score。

每次运行保存 manifest、原始请求响应、配对或状态轨迹、结果和故障计数。全局请求缓存位于 `cache/semceb-dm/decisions`；仅保存成功且结构有效的响应。运行目录位于 `runs/semceb-dm`。两者均由根 `.gitignore` 排除。

数据记录文件大小、mtime 和有序切片身份哈希；尚非全数据内容哈希。版本一致不等于模型可完全复现：远程服务漂移、提示格式与参照模型差异需要单独评估。样本数 2 只用于验证链路，不能说明准确性或性能优势。

下一轮正式实验前应确定查询开发/留出划分、规模、种子数、总费用/时间预算、批处理及冷热缓存规则。最低比较为均匀 DM 抽样、纯状态代理；获得参照配对标签后再增加参照抽样和残差纠偏。多表优化不在当前工程范围。
