# Assay

高性能财报 PDF 表格解析器。基于 YOLO 版面检测 + SLANet 表格结构识别 + PP-LCNet 门控分类，采用共享无关 (shared-nothing) 多进程并行架构，一次性提取 PDF 中所有表格，输出 Markdown + JSON。

## 架构

```
                    ProcessPoolExecutor (N workers)
                    ┌─────────────────────────────────┐
                    │  Worker 0                       │
  PDF files ────→   │    TableModel (GPU detect+struc)│
  (N 份财报)         │    AssayExtractor (CPU 处理)      │──→  Markdown + JSON
                    │                                 │
                    │  Worker 1  (同上)                │
                    │  ...                            │
                    │  Worker N  (同上)                │
                    └─────────────────────────────────┘
```

每个 worker 进程独立加载完整模型管线（YOLO + PP-LCNet + SLANet），各自处理分配到的 PDF。CUDA 时间片轮转透明处理多进程 GPU 并发访问。

**单文件处理流程**:
1. 预检 — 数字密度 + 矢量图元双信号判断页面是否含表格
2. 渲染 — PyMuPDF 渲染页面 (150 DPI)
3. 检测 — YOLO doclayout_yolo 模型定位表格区域 + 磁性对齐
4. 门控 — PP-LCNet 表/非表二分类过滤假阳
5. 识别 — SLANet 结构模型识别行列拓扑 + 合并单元格
6. 映射 — 坐标反向映射 (crop-local → image → PDF)
7. 提取 — PyMuPDF clip 按单元格精确提取文本
8. 补全 — 欠分割表格的几何重建 (completion，仅内容层，带文本 recall 与结构一致性双重自检)

## 评估体系

`eval/` 提供多维评估与回归门禁（基准：`eval/ground_truth.json`，8 本年报 2028 页 2714 框人工标注）：

```bash
PYTHONPATH=. python3 eval/score.py          # 三维评分 + 快照 (eval/runs/)
python3 eval/diff.py eval/baselines/r23.json eval/runs/<新快照>.json --strict
```

- **框级** — P/R/F1 + meanIoU（与 eval.py 同一贪心 IoU 匹配）
- **覆盖** — 单元格面积对 GT 框的覆盖率（SLANet 欠分割可见化）
- **文本** — 单元格拼接文本 vs GT 框原生文本的字符袋 recall/precision
- **diff 门禁** — 任意页面从干净变为失败即 `--strict` 退出非零，杜绝"打地鼠"式修复
- **烟雾模式** — 无标注书自动转入 GT-free 不变量统计（自覆盖率/自文本 recall/重叠率），监测泛化漂移

## 快速开始

```bash
# 单文件
python assay.py -p 年报.pdf -o ./output

# 批量处理 (默认 12 worker)
python assay.py -p ./pdfs/ -o ./output

# 自定义 worker 数
python assay.py -p ./pdfs/ -o ./output -w 8

# 详细日志
python assay.py -p ./pdfs/ -o ./output -v
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p, --path` | PDF 文件或目录 (必填) | — |
| `-o, --output` | 输出目录 (必填) | — |
| `-w, --workers` | CPU worker 数量 | 12 |
| `--gpu` | GPU 设备 ID | 0 |
| `-v, --verbose` | 详细日志 | 否 |

## 输出格式

每个 PDF 生成两个文件：

**Markdown** (`.md`) — 按页面顺序，文本以段落输出，表格使用标准 Markdown 表格语法，包含表头分隔线。

**JSON** (`.json`) — Docling 兼容格式，包含完整的单元格 bbox、行列坐标、行列头标记、合并单元格 span 信息。

## 模型

| 模型 | 骨干网络 | 参数量 | 用途 |
|------|----------|--------|------|
| doclayout_yolo_ft | YOLOv10 | ~29M | 表格区域检测 |
| PP-LCNet | LCNet_x1_0 | ~6.5M | 表/非表门控分类 |
| SLANet-Plus | — | ~7.4M | 表格结构识别 (行列拓扑) |

模型均使用 ONNX Runtime 推理 (FP16)。模型文件存放于 `model/` 目录。

## 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0 + CUDA
- NVIDIA GPU (建议 ≥ 8GB 显存)
- 依赖: `torch`, `transformers`, `PyMuPDF`, `Pillow`

## 项目结构

```
Assay/
├── assay.py              # CLI 入口
├── assay/
│   ├── __init__.py       # 公开 API
│   ├── models.py         # GPU 模型加载与推理 (SLANet + PP-LCNet)
│   ├── detection.py      # YOLO 检测 + 磁性对齐 + 边缘扩展
│   ├── extractor.py      # 8步 PDF 处理管线
│   ├── completion.py     # 欠分割内容补全 (几何重建 + 自检)
│   ├── pipeline.py       # 多 worker 编排器
│   └── output.py         # Markdown / JSON / Debug PDF 输出
├── eval/
│   ├── metrics.py        # 共享指标库 (框级/覆盖/文本)
│   ├── score.py          # 多维评分 + 快照
│   ├── diff.py           # 快照对比回归门禁
│   └── baselines/        # 冻结基线快照
├── README.md
└── LICENSE
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
