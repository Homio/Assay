# Assay

高性能财报 PDF 表格解析器。基于 YOLO 版面检测 + SLANet 表格结构识别 + PP-LCNet 门控分类，输出 Markdown + Docling 兼容 JSON，并配套多维评估与回归门禁体系。

批量模式默认采用 **页粒度调度 + GPU 常驻推理服务** 架构（v2）：CPU worker 只做解析与重组，所有模型推理集中到常驻 GPU 服务，按页/裁剪块动态凑批。参考基准：13 本年报（3134 页）整批约 52 分钟（3144.1s，P34 配置，20 核 CPU + 单卡 GPU）。

## 架构

Assay 有两条执行路径，按输入自动选择：

### 批量模式（默认，v2 页粒度调度）

目录输入且 ≥2 个 PDF 时自动启用（`ASSAY_PIPELINE=legacy` 可强制回退）：

```
 pipeline_config.yaml（唯一阈值源）
        │
        ▼
 ┌─ scheduler_v2 ──────────────────────────────┐
 │  页粒度任务调度 · 预算阀门 · watchdog 水位控制 │
 └──────┬───────────────────────┬───────────────┘
        │ 页/裁剪块任务           │ GPU 请求（三通道）
        ▼                       ▼
 ┌─ worker × N（纯 CPU）─┐   ┌─ gpu_server（常驻）───────┐
 │  文本/裁剪/重组/输出    │   │  YOLO │ PP-LCNet │ SLANet │
 └───────────────────────┘   │  动态凑批，可拆双进程       │
                             └──────────────────────────┘
```

- **调度**（`assay/scheduler_v2.py`）：页粒度任务发派，在制页/裁剪块预算阀门，watchdog 监控内存与负载水位自动暂停/收缩/恢复，单实例文件锁。
- **GPU 服务**（`assay/gpu_server.py`）：常驻推理服务，YOLO / PP-LCNet / SLANet 三通道独立动态凑批（另有 detect+SLANet 合并通道），可拆分为 yolo 与 cls+slanet 双进程。
- **Worker**（`assay/worker_v2.py`）：纯 CPU 流水线，支持任务窃取式异步流水。
- **SLANet 解码器**（`assay/slanet_torch_decoder.py`）：默认启用 torch CUDA Graph 回放 + TF32，异常自动回退 ONNX Runtime 会话。

### 单文件模式（in-process，legacy 路径）

单个 PDF 输入时在进程内直接执行完整管线（每个 worker 独立加载模型的共享无关架构仅保留于此路径与 legacy 回退）。单文件处理流程：

1. 预检 — 数字密度 + 矢量图元双信号判断页面是否含表格
2. 渲染 — PyMuPDF 渲染页面（PAGE_ZOOM=2.0，即 144 DPI）
3. 检测 — YOLO doclayout_yolo 模型定位表格区域 + 磁性对齐 + 矢量线细化
4. 门控 — PP-LCNet 表/非表二分类过滤假阳
5. 识别 — SLANet 结构模型识别行列拓扑 + 合并单元格
6. 映射 — 坐标反向映射 (crop-local → image → PDF)
7. 提取 — PyMuPDF clip 按单元格精确提取文本
8. 补全 — 欠分割表格的几何重建（completion，仅内容层，带文本 recall 与结构一致性双重自检）

## 快速开始

```bash
# 单文件（进程内执行）
python assay.py -p 年报.pdf -o ./output

# 批量处理（自动启用 v2 页粒度调度）
python assay.py -p ./pdfs/ -o ./output

# worker 数量（批量 v2 模式下传 1 表示跟随 pipeline_config.yaml 的 workers）
python assay.py -p ./pdfs/ -o ./output -w 8

# 只出 Markdown / 强制 CPU / 生成标注版调试 PDF
python assay.py -p ./pdfs/ -o ./output --no-json
python assay.py -p 年报.pdf -o ./output --cpu
python assay.py -p 年报.pdf -o ./output --debug-pdf

# 详细日志
python assay.py -p ./pdfs/ -o ./output -v
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p, --path` | PDF 文件或目录（必填） | — |
| `-o, --output` | 输出目录（必填） | — |
| `-w, --workers` | CPU worker 数量；批量 v2 模式下传 1 表示跟随 `pipeline_config.yaml` 的 `workers` | 12 |
| `--gpu` | 保留参数——GPU 常驻服务当前固定绑定 GPU 0，该参数未参与调度 | 0 |
| `-v, --verbose` | 详细日志 | 否 |
| `--no-json` | 跳过 JSON 输出（只出 Markdown） | 否 |
| `--cpu` | 强制 CPU 推理（禁用 GPU） | 否 |
| `--debug-pdf` | 额外生成表格框标注的 `_layout.pdf` 供人工核对 | 否 |

## 输出格式

- **单文件模式**：写 `{output}/{stem}.md` 与 `{stem}.json`（根目录平铺）。
- **批量 v2 模式**：每本书一个子目录 `{output}/{stem}/{stem}.md` 与 `{stem}.json`（评分工具按此布局读取）。
- 超长文件名（派生文件名超 255 字节）自动截断并追加原名的 SHA-256 短哈希，保证唯一且确定性；正常短名逐字节不变。
- `--debug-pdf` 时额外输出 `{stem}_layout.pdf`。

**Markdown** — 文本段落与表格按阅读顺序交错输出，表格回到原文位置，使用标准 Markdown 表格语法（含表头分隔线）。交付前默认经过规范化通道（normalize，可用 `ASSAY_NORMALIZE=0` 关闭）。

**JSON** — Docling 兼容格式（`schema_name: "docling"`），包含完整的单元格 bbox、行列坐标、行列头标记、合并单元格 span 信息。

## 配置

`pipeline_config.yaml` 是 v2 管道全部调度/预算/watchdog 阈值的**唯一来源**（worker 数、三通道凑批窗口、SLANet 解码器开关、内存水位等）；算法阈值仍以 `assay/` 包内常量为准。环境变量：

| 环境变量 | 说明 |
|----------|------|
| `ASSAY_PIPELINE=legacy` | 强制批量模式走旧 per-file 路径（默认 `v2`） |
| `ASSAY_NORMALIZE=0` | 关闭 Markdown 交付规范化（默认启用） |

## 评估体系

`eval/` 提供多维评估与回归门禁。基准：`eval/ground_truth.json`，**13 本年报 3134 页 4563 框**人工标注。

```bash
PYTHONPATH=. python3 eval/score.py    # 三维评分 + 快照 (eval/runs/)

# 快照对比回归门禁：任意页面从干净变为失败即非零退出，杜绝"打地鼠"式修复
python3 eval/diff.py eval/baselines/r30_13books.json eval/runs/<新快照>.json --strict

# 全容差语义门禁：输出目录 vs 输出目录的语义级对比
python3 eval/semantic_diff.py <基线输出目录> <候选输出目录>

# 一键性能回归门禁：精度门禁 + 性能基准 + 内存画像三合一
python3 eval/perf/gate.py
```

- **框级** — P/R/F1 + meanIoU（与 eval.py 同一贪心 IoU 匹配）
- **覆盖** — 单元格面积对 GT 框的覆盖率（SLANet 欠分割可见化）
- **文本** — 单元格拼接文本 vs GT 框原生文本的字符袋 recall/precision
- **门禁工具链** — `diff.py`（快照对比）、`semantic_diff.py`（语义对比）、`gate_arbiter.py`（仲裁）、`checks/`（回归门禁 + 表格完整性不变量）、`perf/`（性能 harness）
- **烟雾模式** — 无标注书自动转入 GT-free 不变量统计（自覆盖率/自文本 recall/重叠率），监测泛化漂移

Label Studio 标注工具链（建项目、导任务、导入人工标注）见 `eval/ls_*.py`、`eval/make_tasks.py`、`eval/import_annotations.py`，工作流约定见 [HANDOVER.md](HANDOVER.md)。

## 模型

| 模型 | 格式 | 用途 | 推理引擎 |
|------|------|------|----------|
| doclayout_yolo_ft (YOLOv10, `.pt`) | PyTorch 权重 | 表格区域检测 | `doclayout_yolo` 包 |
| PP-LCNet_x1_0 | ONNX | 表/非表门控分类 | ONNX Runtime（CUDA EP） |
| SLANet-Plus | ONNX | 表格结构识别（行列拓扑） | ONNX Runtime（CUDA EP）；解码器默认走 torch CUDA Graph + TF32，异常回退 ORT |

三个模型文件随仓库分发（`model/` 目录），clone 后即可直接运行，无需额外下载。

## 环境要求

- Python ≥ 3.10
- NVIDIA GPU（v2 模式建议 ≥ 8GB 显存；也可 `--cpu` 纯 CPU 运行）
- 依赖：`torch`（含 CUDA，spawn 启动与 SLANet torch 解码器）、`onnxruntime-gpu`、`PyMuPDF`、`Pillow`、`numpy`、`PyYAML`、`doclayout_yolo`；eval 侧另需 `requests`

## 项目结构

```
Assay/
├── assay.py                    # CLI 入口（单文件 in-process / 批量分流）
├── pipeline_config.yaml        # v2 管道唯一阈值源
├── model/                      # 运行时权重（随仓库分发）
├── assay/
│   ├── __init__.py             # 公开 API
│   ├── pipeline.py             # 编排器（legacy 路径 + v2 分流）
│   ├── scheduler_v2.py         # 页粒度调度器（预算阀门 + watchdog）
│   ├── worker_v2.py            # CPU worker 流水线
│   ├── gpu_server.py           # GPU 常驻推理服务（三通道动态批）
│   ├── models.py               # PP-LCNet / SLANet ONNX 会话（CUDA EP）
│   ├── slanet_torch_decoder.py # SLANet 解码器 torch CUDA Graph 实现
│   ├── detection.py            # YOLO 检测 + 磁性对齐 + 矢量线细化
│   ├── rulings.py              # 矢量表格线检测
│   ├── splitting.py            # 网格行带 / 列边界算法
│   ├── extractor.py            # 8 步 PDF 处理管线
│   ├── completion.py           # 欠分割内容补全（几何重建 + 自检）
│   ├── normalize.py            # 交付通道规范化
│   ├── textpage_memo.py        # PyMuPDF textpage 记忆化
│   └── output.py               # Markdown / JSON / Debug PDF 输出
├── eval/
│   ├── metrics.py              # 共享指标库（框级/覆盖/文本）
│   ├── score.py                # 多维评分 + 快照
│   ├── diff.py                 # 快照对比回归门禁
│   ├── semantic_diff.py        # 全容差语义门禁
│   ├── gate_arbiter.py         # 门禁仲裁
│   ├── checks/                 # 回归门禁 + 表格完整性不变量
│   ├── perf/                   # 性能 harness（gate.py 主入口）
│   ├── ground_truth.json       # 人工标注基准
│   ├── baselines/              # 冻结基线快照
│   └── runs/                   # 评分快照
├── HANDOVER.md                 # 交接文档（工作流铁律、运维细节）
├── README.md
└── LICENSE
```

## License

GNU AGPL-3.0 — see [LICENSE](LICENSE).
