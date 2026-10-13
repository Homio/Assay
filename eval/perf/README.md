# Assay perf harness (eval/perf/)

一键回归门禁——后续每个性能实施阶段用 `python3 eval/perf/gate.py` 一跑，
立刻拿到精度门禁 + 性能基准 + 内存画像三件事的 PASS/FAIL。

## 目录

```
eval/perf/
├── config.yaml        # 唯一阈值/参数/路径源（零硬编码铁律）
├── gate.py            # 主入口：串联三件事，落 gate_result.json + SUMMARY.txt
├── accuracy.py        # 精度门禁：extract → score → diff；含紫光 p154 仲裁
├── perf.py            # 性能基准：3 次中位数 vs P1 baseline.json（方向性容差）
├── memory.py          # 内存画像：进程树 RSS 采样（单 worker + N workers 批量）
├── utils.py           # 公共：GPU 检查、time -v 解析、tree_rss_mib 采样、JSON I/O
└── README.md          # 本文档
```

**不**改任何 `eval/`, `assay/`, `model/`, `assay.py` 现有代码。
`eval/perf/` 是**只新增**子包，不影响原有门禁流程。

## 入口

```bash
# 全量跑（精度 + 性能 + 内存）—— 一次大约 15-20 分钟
cd /home/homio/sda/1111/Assay
python3 eval/perf/gate.py

# 只跑精度门禁
python3 eval/perf/gate.py --only accuracy

# 只跑性能
python3 eval/perf/gate.py --only performance

# 只跑内存画像
python3 eval/perf/gate.py --only memory

# 跳过一个或多个节
python3 eval/perf/gate.py --skip memory

# 首次 / 重新生成 baseline.json
python3 eval/perf/gate.py --update-baseline
```

> 注意：`--only` / `--skip` 会重写 `gate_result.json`，未跑的节在 JSON 里
> 是空结果——要完整证据就全量跑。

输出落 `analysis/perf/track_p4/`：

- `gate_result.json` —— 机器可读（CI / 后续阶段用）
- `SUMMARY.txt` —— 人读摘要（一眼 PASS/FAIL）
- `perf_runs/<alias>/r{1,2,3}/` —— 每次单本跑的 assay 产物
- `memory_samples/single.csv`, `batch.csv` —— 进程树 RSS 时序
- `memory_single/assay_single.log`, `memory_batch/assay_batch.log` —— assay 运行日志
- `baseline.json` —— 性能基线（P1 报告播种；可用 `--update-baseline` 重生成）

## 判定口径

| 节 | PASS 条件 | 数据来源 |
|---|---|---|
| **精度门禁** | diff.py `REGRESSED + LOST + DEGRADED ≤ 0` 且 `CHANGED ≤ 0` 且 `SMOKE_DRIFT ≤ 0`；**紫光 p154 GPU 抖动仲裁**：若仅紫光 p154 一项红，自动同代码快照重抽紫光 → p154 IoU 落入 `iou_baseline ± iou_tolerance` → 判 PASS 并在 `arbitration` 字段写仲裁依据 | `eval/runs/r57_3books.json` 基线 + 本次 score snapshot |
| **性能基准** | 平安 / 石油各跑 3 次取中位数；**方向性容差（回归门禁只拦劣化）**：wall 慢于基线 ≤ `wall_pct_slower`（默认 15%）、user 慢于 ≤ `user_pct_slower`（默认 20%）、peak_rss 高于 ≤ `peak_rss_pct_slower`（默认 10%）；`*_pct_faster` 默认 null = 变快不限（改善不是回归；要双向门禁给数值即可） | `paths.track_root/baseline.json` |
| **内存画像** | 单 worker peak RSS ≤ `memory.single.peak_rss_mib_max`（默认 4000MiB）；批量 12 workers peak total RSS ≤ `memory.batch.peak_total_rss_mib_max`（默认 38000MiB）| ps 进程树 RSS 时序 |

> 整门禁 `verdict = PASS` ⇔ 三节**全** PASS。
> `config.driver.fail_on_any_section = true` 时任一节 FAIL → 整体 exit 1（CI 友好）。

## 内存采样机制（为什么按进程树而不是按 cmdline）

assay.py 用 `torch.multiprocessing` **spawn** 起 worker，worker 的 cmdline 是
`python3 -c "from multiprocessing.spawn import spawn_main; ..."`——**不含
"assay" 字样**。按 cmdline 过滤会把全部 worker 漏掉（批量采样只剩主进程，
曾因此把 12-worker 批量峰值误采成 534MiB）。

`utils.tree_rss_mib(root_pid)` 按 **ppid 进程树**聚合 RSS：以 assay.py 主进程
为根，BFS 全部后代求和，对 spawn/fork/嵌套子进程都鲁棒。单本与批量统一
走这一机制，采样间隔由 `memory.sample_interval_sec` 控制（默认 1Hz）。

批量超时保护：`memory.batch.max_wait_sec`（默认 1500s；P1 实测 13 本约 340s，
留 4 倍余量），超时自动 kill 并报 ERROR。

## 紫光 p154 GPU 抖动仲裁

**触发条件**（全部满足才进入仲裁）：
1. `accuracy.arbitration.enabled = true`
2. `accuracy.rerun_for_arbitration = true`
3. 本次 diff.py 输出里 `REGRESSED + LOST + DEGRADED` **仅** 紫光 p154 一项

**仲裁动作**：
1. 重新抽取紫光（同一个 GPU0、同一个代码快照 HEAD）
2. 跑 `eval/score.py` → 拿重抽快照
3. 取紫光 p154 的 matches 中各表 IoU 均值
4. 若均值落入 `iou_baseline ± iou_tolerance` → 视为 GPU 抖动，**判 PASS**
   （`verdict=FAIL_WITH_ARBITRATION`，`section_pass=true`）；否则真回归 → FAIL

**为什么不绑死到紫光**：
- `iou_baseline=0.9237` 与 `iou_tolerance=0.02` 是**通用参数**——可改、可换
- `target_book_alias=unisplendour` 来自 `books.alias`，可换其他书
- `target_page=154` 是通用页号
- 仲裁只判断「同代码下 p154 IoU 是否复现同量级」，与「紫光这本书」的语义无关

## 更新基线

**什么时候该更新**：
- 改完性能实施阶段、性能数字跨档（> 20% 偏移）需要重新设基线
- 换机器 / 换 GPU
- 跑新基线（首次跑也用这条）

**怎么更新**：
```bash
cd /home/homio/sda/1111/Assay
python3 eval/perf/gate.py --update-baseline
```

落 `analysis/perf/track_p4/baseline.json`，包含：
- meta（机器、code_rev、runs_per_book、aggregate）
- books.alias.wall_sec_median / user_sec_median / peak_rss_mib_median
- books.alias.*_runs（每次跑的原始值，方便审计）

**更新前自查**：
- 当前 HEAD 是不是你打算固化基线的代码版本
- GPU0 是不是空闲
- 没有其他用户进程在跑

**更新后建议**：
- 把 baseline.json 归档（或在 commit message 里写明：基线对应的 HEAD / 机器 / 跑批条件）

## 零硬编码铁律

任何「阈值」「路径」「页号」「书 alias」全部在 `config.yaml`。代码里**绝不**写：

```python
# ❌ 禁止
if iou < 0.9237:  # 绑死到紫光 p154 的 IoU
    pass

# ✅ 正确
iou_threshold = cfg["accuracy"]["arbitration"]["iou_baseline"]
```

新加规则时：
1. 先在 `config.yaml` 加 key
2. 代码读 `cfg[<section>][<key>]`
3. 写单测 / 集成验证「改 config 后行为符合预期」

## 已知限制 / 留口子

1. **精度门禁的仲裁基线 `iou_baseline=0.9237`** 是 P4 阶段的经验值；若换机器 / 换 GPU 后基线偏移，需改 config（不是改代码）。
2. **性能基线 `baseline.json`** 当前是 P1 报告播种值（wall 49.9s/45.0s）；
   实测同 HEAD 复现值可能持续偏快（见 track_p4/REPORT.md §2 残差分析），
   需要双向门禁时给 `*_pct_faster` 配数值，或 `--update-baseline` 重播种。
3. **批量内存画像** 默认 12 workers / 12 PDFs（取 pdf_root 排序后前 12 本）；
   若机器并发数变化，调 `memory.batch.workers` 与 `pdf_count`。
4. **GPU0 检查** `pre_check_max_used_mib=200` 是保守值；若跑批与 harness 并存，先释放 GPU0 再跑 harness。
5. **未触碰现有代码**——`eval/diff.py`、`eval/score.py`、`assay.py`、`assay/*` 全部未改；本 harness 是**外挂**门禁。

## 故障排查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `[pre-check] FAIL: GPU0 used=XMiB` | GPU0 已被其他进程占用 | `nvidia-smi` 查谁在用；释放后再跑 |
| `score failed: exit=1` | 抽取阶段某本书失败 | 看 `gate_result.json` 的 `accuracy.books[].stderr_tail` |
| `baseline r57_3books.json not found` | 路径错 | 改 `config.yaml::paths.baseline_runs_dir` |
| `extract 之后无 <stem>.json` | assay.py 单文件模式产物位置不对 | `utils.stage_for_score` 会自动 mv 到 `<out>/<stem>/` 子目录 |
| `arbitration: FAIL` | 同代码下 p154 IoU 不在仲裁范围内 | 不是 GPU 抖动，是真回归；看 `accuracy.arbitration.rationale` |
| 批量内存 `n_pids` 一直很小 | 采样没跟上 spawn worker | 不应再发生（已改进程树聚合）；若复现看 `memory_batch/assay_batch.log` |
| 批量 `exceeded max_wait_sec` | 机器负载过高 / 并发退化 | 调大 `memory.batch.max_wait_sec` 或空闲态重跑 |

## 跑一次实测的预期时长

| 节 | 时长（依机器空闲度）|
|---|---|
| 精度门禁 | 3 本各 35-50s + score 5-10s + diff 1s ≈ **3-4 min** |
| 精度仲裁（若触发） | 再 1 本 35-50s + score 5-10s ≈ **1 min** |
| 性能基准 | 2 本 × 3 次 × 35-50s ≈ **4-5 min** |
| 内存单 worker | 1 本 35-50s + 采样 ≈ **1 min** |
| 内存批量 12 workers | 12 本并行 ≈ 5-7 min（按 P1 报告 4-7 倍并发退化）|
| **合计** | **约 15-20 min**（不含仲裁）|

## 不在门禁内（边界）

- 13 本全量门禁（任务边界明确「13 本全量不在本任务」）——内存批量用 12 本
  是为了压并发内存，不是精度全量
- 精度全 13 本差异 vs `r30_13books.json` 老基线 —— 老基线在 `eval/baselines/`，是历史快照
- 性能/内存跨机器对比 —— 每次跑都是当台机器的局部数据
- 持久化模型复用收益（worker 单例化）—— 等实施阶段具体方案
- 真实业务语料质量指标（段落/标题最终一致性）—— 见 HANDOVER.md §7

## 关联文档

- 项目交接：`Assay/HANDOVER.md`（工作流铁律、代码机制地图）
- 性能基线（P1 报告）：`analysis/perf/track_p1/REPORT.md`
- 本 harness 建设报告：`analysis/perf/track_p4/REPORT.md`
- 历史快照：`Assay/eval/runs/r57_3books.json`（精度基线）
- 旧 13 本基线：`Assay/eval/baselines/r30_13books.json`（参考用）
