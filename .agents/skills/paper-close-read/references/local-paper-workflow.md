# Local Paper Workflow

本文件覆盖**本地 PDF 分支**的材料准备；arXiv 分支见 [arxiv-md-workflow.md](arxiv-md-workflow.md)，已有可读材料的分支见 SKILL.md。

## Goal

把用户给的单篇本地论文 PDF，整理成一套可写、可插图、可追踪证据的工作目录，而不是直接对着 PDF 生写。

## Recommended Workspace Layout

默认（`mineru-open-api` 可用并已配置 Token）：

```text
paper-workdir/
├── source.pdf          # 必备：原始材料副本
├── analysis.md         # 必备：图文精读成品
├── paper.md            # 推荐：MinerU 结构化 Markdown（含物理页码注释标记、LaTeX 公式与表格）
├── paper.json          # 推荐：MinerU 块级版面元数据（含 page_idx、bbox、图注）
├── prep_status.txt     # 准备状态与实际采用的后端记录
└── images/             # 推荐：自动裁切的论文插图、统计图、表格切图与索引
    ├── index.md
    └── <hash>.jpg
```

本地回退（无 `mineru-open-api`、未配置 Token、网络受限或指定 `--backend pymupdf`）：

```text
paper-workdir/
├── source.pdf          # 必备：原始材料副本
├── analysis.md         # 必备：图文精读成品
├── paper.txt           # 推荐：带页码标记的 PyMuPDF 提取文本
├── prep_status.txt     # 准备状态与回退原因记录
└── images/             # 推荐：整页渲染与嵌入图
    ├── index.md
    ├── page-001.png
    └── ...
```

如果已有更成熟的目录结构，可以兼容，保留原文来源（`source.pdf`）与精读成品（`analysis.md`）即可。

## Material Preparation Order

### 1. 运行统一准备脚本

优先顺序：

1. `uv run scripts/prepare_local_pdf.py <pdf> <out-dir>/` —— 默认入口：
   - **优先采用 `mineru-open-api extract --ocr -f md,json`**：当本机已安装 `mineru-open-api` 且 `mineru-open-api auth --verify`（或 `MINERU_TOKEN`）可用时自动启用。它会重排双栏阅读顺序、把行内与块级公式还原为 LaTeX、将表格转为结构化表格，并根据 `paper.json` 的 `page_idx` 在跨页段落边界插入 `<!-- ===== [page N] ===== -->` 标记。默认开启 `--ocr`，避免 PDF 内嵌字体 CMap 造成行内数学符号乱码（``）或连字丢失。
   - **自动回退 `PyMuPDF`**：若未安装 `mineru-open-api`、未配置 Token 或云端解析失败，脚本自动使用 PEP 723 声明的 `PyMuPDF` 提取带 `===== [page N] =====` 的 `paper.txt` 并渲染整页 PNG，同时把回退原因写入 `prep_status.txt`。
2. **离线或未公开稿件**：`mineru-open-api` 会将 PDF 上传至云端解析；处理审稿件、未公开草稿或无网环境时，显式加 `--backend pymupdf` 走纯本地提取。
3. **补充整页渲染**：MinerU 模式下若发现极少数纯矢量图漏裁、裁切不全，或需核对密集表格原貌，可运行 `uv run scripts/prepare_local_pdf.py <pdf> <out-dir>/ --render-pages` 额外产出 `page-NNN.png`。
4. 没有 `uv` 且无法使用 MinerU 时，可用 `python3 scripts/prepare_local_pdf.py <pdf> <out-dir>/ --backend pymupdf` 触发 `poppler` 兜底（能力较弱且无页码标记）。

注意：
- **不要**改用免认证的 `mineru-open-api flash-extract` 替代 `extract`：`flash-extract` 会将公式、表格、图片替换为占位符，丢失论文核心证据。
- **不要**手动 `pip install` 或 `apt install` PDF 工具来绕开脚本：本地依赖声明集中在脚本头部，绕开会让环境不可复现。

写正文时**直接用 `paper.md` 中的 `<!-- ===== [page N] ===== -->`（或 `paper.txt` 中的 `===== [page N] =====`）给关键判断与数字标物理页码**，不要靠估算数页；若需精确核对跨页段落归属，也可查阅 `paper.json` 中对应块的 `page_idx`（从 0 起算，物理页码为 `page_idx + 1`）。

如果提取失败：

- 明确记录当前有哪些材料可用（见 `prep_status.txt`）
- 继续寻找同篇论文的已提取文本、副本、笔记
- 不要因为缺少一个工具就直接只看标题和摘要

提取完成后，至少先定位：

- 摘要原文
- 章节标题
- 图号 / 表号
- 关键公式
- 关键数字所在页

### 2. 检查图片与表格资源

根据实际后端检查 `images/index.md`：

1. **MinerU 模式（默认）**：
   - `images/` 中直接包含自动裁切好的插图（`image`）、统计图（`chart`）和表格切图（`table`）。其中插图与统计图已自动内联在 `paper.md` 中；表格在 `paper.md` 正文里以结构化 `<table>` 呈现，同时 `images/` 中也保留了对应的原表切图（在 `images/index.md` 中标为 `table crop`），正文中既可整理成 Markdown 表格，也可直接引用原表切图。
   - `images/index.md` 末尾的 `Auxiliary / Formula Crops` 是 MinerU 识别块级公式时附带切出的公式小图，正文优先用 `$...$` / `$$...$$` 公式排版，仅在怀疑公式识别有误时用它核对原貌。
2. **PyMuPDF 回退模式**：
   - `page-NNN.png`（200dpi 整页渲染）是主要视觉材料；要用某张矢量图或表格时，从对应页面渲染里裁出图区。需要更高清晰度时加 `--dpi 300`。
   - `img-NNN.png`（嵌入位图，已过滤小于 120px 的装饰碎片与空遮罩）仅作补充。

目标不是“抽尽一切资源”，而是保证这些视觉证据可被正文使用：

- 方法框架图
- 核心表格或主结果图
- 消融图
- 案例图或可视化图

### 3. 维护图片索引

脚本已在 `images/index.md` 写好文件名、来源页码及（MinerU 模式下的）类型与原始图注；按需为实际采用的图片补充说明即可，无需逐张规划用途。

如果表格没有单独使用图片，在正文相关段落注明表号、页码与关键数字即可。

## Figure Usage Rules

- 不能只提“见图 3”，必须解释图支持哪条论点。
- 优先使用原论文图，不要无必要地重画。
- 挑选有助于解释核心机制和贡献的图表，不设图片数量或类型配额。
- 如果论文几乎没有图，允许使用页面截图或文本结构图补位。
- 按图表实际表达的内容使用，不将结果图改称为方法图以满足格式。

## Reading Strategy After Preparation

材料准备完成后，可沿以下路径理解研究叙事；按论文类型调整，不要求先写独立中间文件：

1. 标题、摘要、引言：确定问题、动机、主张
2. 方法章节：圈出关键模块、公式、训练推断流程
3. 实验章节：列出实验问题、数据集、基线、指标、关键表图
4. 结论与讨论：提炼局限、未来方向、适用边界
5. 在正文中将关键设计、结果与贡献连接起来，就近标注来源

如果论文不是标准实验论文，把第 3 步替换成：

- 实证论文：研究设计、变量、样本、关键表图
- 计量 / 历史论文：时间范围、统计对象、分段方式、关键图表
- 综述 / 理论论文：论点结构、证据来源、作者的归纳边界

## Failure Handling

- MinerU 未认证或超时：脚本自动回退 PyMuPDF，查看 `prep_status.txt` 确认当前后端
- 没有文本提取工具：改用已有文本或截图，不中断分析
- 没有图片提取工具：改用关键页面截图，不省略图文结构
- 图号不清晰：在正文里注明“依据页面内容判断”
- 表格或公式识别存疑：对照 `images/` 中的切图或加 `--render-pages` 对照原页确认数字与符号，不猜测
