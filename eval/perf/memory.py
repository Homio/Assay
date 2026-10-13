"""
Memory profile — process-tree RSS sampling, single worker + batch N workers.

零硬编码铁律：sample_interval / peak_rss_mib_max / batch_workers /
max_wait_sec 全部来自 config.yaml；本模块只提供通用采样与判定机制。

采样机制说明：assay.py 用 torch.multiprocessing spawn 起 worker，
worker 的 cmdline 是 `python3 -c "from multiprocessing.spawn import
spawn_main; ..."`，不含 "assay" 字样——按 cmdline 过滤会漏采。
因此单本与批量统一用 utils.tree_rss_mib（按 ppid 树聚合 RSS），
对 spawn/fork/嵌套子进程都鲁棒。
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Module alias shim (see gate.py for the why).
import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

try:
    import utils as _utils
    if "eval.perf.utils" not in _sys.modules:
        _sys.modules["eval.perf.utils"] = _utils
except Exception:
    from . import utils as _utils  # type: ignore[no-redef]
utils = _utils


@dataclass
class RssSample:
    ts: float
    total_rss_mib: float
    n_pids: int

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 2),
            "total_rss_mib": round(self.total_rss_mib, 1),
            "n_pids": self.n_pids,
        }


@dataclass
class MemoryResult:
    single: Dict[str, Any] = field(default_factory=dict)
    batch: Dict[str, Any] = field(default_factory=dict)
    section_pass: bool = False
    verdict: str = "UNKNOWN"
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "single": self.single,
            "batch": self.batch,
            "section_pass": self.section_pass,
            "verdict": self.verdict,
            "errors": self.errors,
        }


# ── Shared sampling loop ────────────────────────────────────────────────


def _spawn_assay(assay_root: Path, target: Path, out_dir: Path,
                 workers: int, gpu_index: int, log_path: Path) -> subprocess.Popen:
    """Spawn assay.py with stdout/stderr redirected to a log file.

    不用 PIPE：-v 输出可能灌满 64KB 管道缓冲造成子进程阻塞。
    """
    cmd = [
        "python3", "assay.py",
        "-p", str(target),
        "-o", str(out_dir),
        "-w", str(workers),
        "--gpu", str(gpu_index),
        "-v",
    ]
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd, cwd=str(assay_root), env=os.environ.copy(),
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    # 持有方负责关闭；把 fh 挂到 proc 上，调用方 wait 后统一关。
    proc._assay_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _sample_until_exit(proc: subprocess.Popen, samples_csv: Path,
                       interval: float, deadline_sec: Optional[float],
                       header: List[str]) -> Dict[str, Any]:
    """
    1Hz(可配) 对 proc 的整棵进程树采样 RSS，写 CSV，直到 proc 退出。
    超时则 kill 并标记 timed_out。返回采样统计。
    """
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    totals: List[float] = []
    t0 = time.time()
    timed_out = False
    with samples_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        while True:
            n, mib = utils.tree_rss_mib(proc.pid)
            ts = time.time() - t0
            totals.append(mib)
            w.writerow([round(ts, 2), n, round(mib, 1)])
            f.flush()
            if proc.poll() is not None:
                break
            if deadline_sec is not None and ts > deadline_sec:
                proc.kill()
                timed_out = True
                break
            time.sleep(interval)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_fh = getattr(proc, "_assay_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass
    return {
        "n_samples": len(totals),
        "duration_sec": round(time.time() - t0, 2),
        "peak": max(totals) if totals else 0.0,
        "avg": sum(totals) / len(totals) if totals else 0.0,
        "p50": utils.percentile(totals, 50) if totals else 0.0,
        "p99": utils.percentile(totals, 99) if totals else 0.0,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
    }


# ── Single-worker profile ──────────────────────────────────────────────


def _profile_single(cfg: dict, samples_csv: Path) -> Dict[str, Any]:
    assay_root = Path(cfg["paths"]["assay_root"])
    pdf_root = Path(cfg["paths"]["pdf_root"])
    gpu_index = cfg["gpu"]["allowed_index"]
    interval = float(cfg["memory"]["sample_interval_sec"])
    books = cfg["memory"]["single"]["books"]
    out_dir = Path(cfg["paths"]["track_root"]) / "memory_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not books:
        return {"status": "SKIPPED", "reason": "no books in single.books"}

    alias = books[0]
    info = cfg["books"].get(alias)
    if not info:
        return {"status": "ERROR", "reason": f"unknown alias {alias}"}
    pdf = pdf_root / f"{info['stem']}.pdf"
    if not pdf.exists():
        return {"status": "ERROR", "reason": f"PDF missing: {pdf}"}

    # Clean prior run artifacts
    for f in out_dir.glob(f"{pdf.stem}*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    proc = _spawn_assay(assay_root, pdf, out_dir, workers=1,
                        gpu_index=gpu_index,
                        log_path=out_dir / "assay_single.log")
    time.sleep(2.0)
    stats = _sample_until_exit(
        proc, samples_csv, interval, deadline_sec=None,
        header=["ts_sec", "n_pids", "total_rss_mib"],
    )

    cfg_max = float(cfg["memory"]["single"]["peak_rss_mib_max"])
    return {
        "status": "OK",
        "book_alias": alias,
        "n_samples": stats["n_samples"],
        "duration_sec": stats["duration_sec"],
        "peak_rss_mib": round(stats["peak"], 1),
        "avg_rss_mib": round(stats["avg"], 1),
        "p50_rss_mib": round(stats["p50"], 1),
        "p99_rss_mib": round(stats["p99"], 1),
        "max_rss_mib_threshold": cfg_max,
        "within_threshold": stats["peak"] <= cfg_max,
        "samples_csv": str(samples_csv),
        "exit_code": stats["exit_code"],
    }


# ── Batch profile (N workers via assay.py process pool) ─────────────────


def _profile_batch(cfg: dict, samples_csv: Path) -> Dict[str, Any]:
    assay_root = Path(cfg["paths"]["assay_root"])
    pdf_root = Path(cfg["paths"]["pdf_root"])
    gpu_index = cfg["gpu"]["allowed_index"]
    interval = float(cfg["memory"]["sample_interval_sec"])
    n_workers = int(cfg["memory"]["batch"]["workers"])
    pdf_count = int(cfg["memory"]["batch"]["pdf_count"])
    max_wait = float(cfg["memory"]["batch"].get("max_wait_sec", 1500))
    out_dir = Path(cfg["paths"]["track_root"]) / "memory_batch"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Memory batch mode: sample concurrent worker memory, not gate accuracy.
    # Pull PDFs from pdf_root directly so we can pick `pdf_count` files
    # regardless of how many are in the books registry (perf/accuracy books
    # are 3; memory wants 12 to push batch-mode pressure).
    candidates = [(pdf.stem, pdf) for pdf in sorted(pdf_root.glob("*.pdf"))]
    if len(candidates) < pdf_count:
        return {"status": "ERROR",
                "reason": f"need {pdf_count} PDFs, only {len(candidates)} available under {pdf_root}"}
    chosen = candidates[:pdf_count]
    chosen_dir = out_dir / "pdfs"
    chosen_dir.mkdir(parents=True, exist_ok=True)
    for alias, pdf in chosen:
        link = chosen_dir / pdf.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(pdf, link)

    proc = _spawn_assay(assay_root, chosen_dir, out_dir, workers=n_workers,
                        gpu_index=gpu_index,
                        log_path=out_dir / "assay_batch.log")
    stats = _sample_until_exit(
        proc, samples_csv, interval, deadline_sec=max_wait,
        header=["timestamp_sec", "n_pids", "total_rss_mib"],
    )
    if stats["timed_out"]:
        return {"status": "ERROR",
                "reason": f"batch exceeded max_wait_sec={max_wait:.0f}; killed. "
                          f"see {out_dir / 'assay_batch.log'}"}

    cfg_max = float(cfg["memory"]["batch"]["peak_total_rss_mib_max"])
    return {
        "status": "OK",
        "n_workers": n_workers,
        "n_pdfs": pdf_count,
        "n_samples": stats["n_samples"],
        "duration_sec": stats["duration_sec"],
        "peak_total_rss_mib": round(stats["peak"], 1),
        "avg_total_rss_mib": round(stats["avg"], 1),
        "p99_total_rss_mib": round(stats["p99"], 1),
        "max_total_rss_mib_threshold": cfg_max,
        "within_threshold": stats["peak"] <= cfg_max,
        "samples_csv": str(samples_csv),
        "exit_code": stats["exit_code"],
        "pdf_aliases": [a for a, _ in chosen],
    }


# ── Section entry ───────────────────────────────────────────────────────


def run_memory_profile(cfg: dict) -> MemoryResult:
    result = MemoryResult()
    track_root = Path(cfg["paths"]["track_root"])
    track_root.mkdir(parents=True, exist_ok=True)
    samples_root = track_root / "memory_samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    try:
        result.single = _profile_single(cfg, samples_root / "single.csv")
    except Exception as e:
        result.errors.append(f"single failed: {e}")
        result.single = {"status": "ERROR", "reason": str(e)}

    try:
        result.batch = _profile_batch(cfg, samples_root / "batch.csv")
    except Exception as e:
        result.errors.append(f"batch failed: {e}")
        result.batch = {"status": "ERROR", "reason": str(e)}

    s_ok = result.single.get("status") == "OK" and result.single.get("within_threshold", True)
    b_ok = result.batch.get("status") == "OK" and result.batch.get("within_threshold", True)
    if s_ok and b_ok:
        result.verdict = "PASS"
        result.section_pass = True
    else:
        result.verdict = "FAIL"
    return result
