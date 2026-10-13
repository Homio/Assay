"""
Assay perf harness — one-click regression gate.

驱动 eval/perf/{accuracy,perf,memory}.py 三件套；按 config.yaml 阈值
输出机器可读 JSON（gate_result.json）+ 人读摘要（SUMMARY.txt）。

零硬编码铁律：本文件不绑任何书/阈值；只引用 config.yaml 字段。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# eval/ 在该仓库不是 Python 包（无 __init__.py）；以脚本形式运行。
# 把 _HERE 注入 sys.path，让 `import utils/accuracy/perf/memory` 可解析。
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (str(_HERE), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 同时把 perf 子包挂到 `eval.perf` 名下，方便 `python -m eval.perf.gate` 调用方。
import types as _types
if "eval" not in sys.modules:
    _eval_pkg = _types.ModuleType("eval")
    _eval_pkg.__path__ = [str(_PARENT)]
    sys.modules["eval"] = _eval_pkg
if "eval.perf" not in sys.modules:
    _perf_pkg = _types.ModuleType("eval.perf")
    _perf_pkg.__path__ = [str(_HERE)]
    sys.modules["eval.perf"] = _perf_pkg

import utils  # noqa: E402
import accuracy  # noqa: E402
import perf  # noqa: E402
import memory  # noqa: E402
sys.modules["eval.perf.utils"] = utils
sys.modules["eval.perf.accuracy"] = accuracy
sys.modules["eval.perf.perf"] = perf
sys.modules["eval.perf.memory"] = memory


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _section(title: str) -> str:
    line = "─" * 72
    return f"\n{line}\n  {title}\n{line}\n"


def _human_summary(ar, pr, mr, cfg: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"  Assay perf harness report — {_now()}")
    lines.append(f"  track  : {cfg['paths']['track_label']}")
    lines.append(f"  config : {_HERE / 'config.yaml'}")
    lines.append("=" * 72)

    # ── Accuracy
    lines.append(_section("1. 精度门禁 (Accuracy Gate)"))
    lines.append(f"  baseline  : {Path(ar.baseline_snapshot).name if ar.baseline_snapshot else 'N/A'}")
    lines.append(f"  candidate : {Path(ar.candidate_snapshot).name if ar.candidate_snapshot else 'N/A'}")
    lines.append(f"  verdict   : {ar.verdict}")
    if ar.diff_summary:
        d = ar.diff_summary
        lines.append(
            f"  regressed={d.get('REGRESSED', 0)} lost={d.get('LOST', 0)} "
            f"degraded={d.get('DEGRADED', 0)} changed_dirty={d.get('CHANGED', 0)} "
            f"smoke_drift={d.get('SMOKE_DRIFT', 0)}"
        )
    for b in ar.books:
        lines.append(
            f"  - {b.alias:14s} tables={b.table_count:4d} "
            f"wall={b.extract_wall_sec:6.1f}s "
            f"peak_rss={b.extract_peak_rss_mib:6.0f}MiB "
            f"status={b.status}"
        )
    if ar.arbitration:
        a = ar.arbitration
        lines.append(
            f"  arbitration: {a['verdict']} — {a['rationale']}"
        )
    if ar.overall_baseline and ar.overall_candidate:
        b = ar.overall_baseline
        c = ar.overall_candidate
        lines.append(
            f"  overall: F1 {b.get('F1', 0):.4f}→{c.get('F1', 0):.4f} | "
            f"P {b.get('P', 0):.4f}→{c.get('P', 0):.4f} | "
            f"R {b.get('R', 0):.4f}→{c.get('R', 0):.4f}"
        )

    # ── Performance
    lines.append(_section("2. 性能基准 (Performance Baseline)"))
    lines.append(f"  baseline : {pr.baseline_file}")
    lines.append(f"  verdict  : {pr.verdict}  violations={pr.violations}")
    metric_keys = [
        k for k, mc in cfg["performance"].get("metric", {}).items()
        if mc.get("required")
    ]
    for b in pr.books:
        lines.append(f"  - {b.alias} ({b.pages}p)")
        for k in metric_keys:
            if k not in b.median:
                continue
            v = b.median[k]
            base = b.baseline.get(k + "_median")
            if base is None:
                base = b.baseline.get(k)
            d = b.delta_pct.get(f"{k}_pct", 0.0)
            ok = b.within_tolerance.get(k, True)
            mark = "✓" if ok else "✗"
            base_s = f"{base:.2f}" if isinstance(base, (int, float)) else "N/A"
            lines.append(
                f"      {k:18s} median={v:8.2f}  base={base_s:>8}  "
                f"Δ={d:+6.1f}% {mark}"
            )
        for r in b.runs:
            lines.append(
                f"      r{r.run_index}: wall={r.wall_sec:.1f}s user={r.user_sec:.1f}s "
                f"peak_rss={r.peak_rss_mib:.0f}MiB exit={r.exit_code}"
            )

    # ── Memory
    lines.append(_section("3. 内存画像 (Memory Profile)"))
    lines.append(f"  verdict: {mr.verdict}")
    if mr.single.get("status") == "OK":
        s = mr.single
        lines.append(
            f"  single ({s['book_alias']}): peak={s['peak_rss_mib']:.0f}MiB "
            f"avg={s['avg_rss_mib']:.0f}MiB p99={s['p99_rss_mib']:.0f}MiB "
            f"(threshold={s['max_rss_mib_threshold']:.0f}MiB) "
            f"{'✓' if s['within_threshold'] else '✗'}"
        )
        lines.append(f"    samples: {s['samples_csv']}")
    else:
        lines.append(f"  single: {mr.single.get('status')} — {mr.single.get('reason')}")
    if mr.batch.get("status") == "OK":
        s = mr.batch
        lines.append(
            f"  batch ({s['n_workers']} workers / {s['n_pdfs']} pdfs): "
            f"peak_total={s['peak_total_rss_mib']:.0f}MiB "
            f"avg={s['avg_total_rss_mib']:.0f}MiB p99={s['p99_total_rss_mib']:.0f}MiB "
            f"(threshold={s['max_total_rss_mib_threshold']:.0f}MiB) "
            f"{'✓' if s['within_threshold'] else '✗'}"
        )
        lines.append(f"    samples: {s['samples_csv']}")
    else:
        lines.append(f"  batch: {mr.batch.get('status')} — {mr.batch.get('reason')}")

    # ── Overall
    lines.append(_section("OVERALL"))
    overall_pass = _overall_pass(ar, pr, mr)
    lines.append(f"  {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _overall_pass(ar, pr, mr) -> bool:
    """overall 只统计实际跑过的节：--only/--skip 跳过的节（verdict=UNKNOWN
    且 section_pass=False 的默认空结果）不参与判定，否则单节重跑永远 FAIL。"""
    parts = []
    for r in (ar, pr, mr):
        if r.verdict == "UNKNOWN" and not r.section_pass:
            continue
        parts.append(r.section_pass)
    return all(parts) if parts else False


def _gate_dict(ar, pr, mr, cfg: dict, t0: float) -> dict:
    return {
        "ts": _now(),
        "duration_sec": round(time.time() - t0, 1),
        "config_path": str(_HERE / "config.yaml"),
        "track_label": cfg["paths"]["track_label"],
        "track_root": cfg["paths"]["track_root"],
        "verdict": "PASS" if _overall_pass(ar, pr, mr) else "FAIL",
        "accuracy": ar.as_dict(),
        "performance": pr.as_dict(),
        "memory": mr.as_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", default=str(_HERE / "config.yaml"),
        help="path to config.yaml",
    )
    ap.add_argument(
        "--skip", choices=["accuracy", "performance", "memory"],
        action="append", default=[],
        help="skip a section (repeatable)",
    )
    ap.add_argument(
        "--update-baseline", action="store_true",
        help="generate / refresh baseline.json for the perf gate, then exit",
    )
    ap.add_argument(
        "--only", choices=["accuracy", "performance", "memory"],
        help="run only one section (others skipped)",
    )
    args = ap.parse_args()

    cfg = utils.load_config(args.config)

    if args.update_baseline:
        path = perf.update_baseline(cfg)
        print(f"baseline written: {path}")
        return 0

    skip = set(args.skip or [])
    if args.only:
        skip = {"accuracy", "performance", "memory"} - {args.only}

    track_root = Path(cfg["paths"]["track_root"])
    track_root.mkdir(parents=True, exist_ok=True)

    # Pre-check GPU
    if cfg["driver"].get("fail_fast_on_gpu_busy", True):
        try:
            st = utils.assert_gpu_free(cfg)
            print(f"[pre-check] GPU{st.index} ({st.name}) free={st.free_mib}MiB "
                  f"used={st.used_mib}MiB util={st.util_pct}% — OK")
        except Exception as e:
            print(f"[pre-check] FAIL: {e}", file=sys.stderr)
            return 2

    t0 = time.time()
    ar = accuracy.AccuracyResult()
    pr = perf.PerfResult()
    mr = memory.MemoryResult()

    if "accuracy" not in skip:
        print("[1/3] accuracy gate ...")
        ar = accuracy.run_accuracy_gate(cfg)
        print(f"      → {ar.verdict}")
    if "performance" not in skip:
        print("[2/3] performance gate ...")
        pr = perf.run_perf_gate(cfg)
        print(f"      → {pr.verdict}")
    if "memory" not in skip:
        print("[3/3] memory profile ...")
        mr = memory.run_memory_profile(cfg)
        print(f"      → {mr.verdict}")

    out = _gate_dict(ar, pr, mr, cfg, t0)
    if cfg["driver"]["write_machine_json"]:
        utils.write_json(
            track_root / cfg["driver"]["machine_json_name"], out
        )
    if cfg["driver"]["write_human_summary"]:
        summary = _human_summary(ar, pr, mr, cfg)
        (track_root / cfg["driver"]["human_summary_name"]).write_text(
            summary, encoding="utf-8"
        )
        print("\n" + summary)

    overall = out["verdict"]
    if cfg["driver"].get("fail_on_any_section", False) and overall != "PASS":
        return 1
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
