"""
Performance gate — single-book 3-run median vs P1 baseline.

零硬编码铁律：所有 runs_per_book / aggregate / tolerance 来自 config.yaml；
基线数值落盘在 paths.track_root/baseline.json（用户可改、可重生成），
不绑任何特定书的字面值到代码。

容差语义（方向性，回归门禁只拦劣化）：
- <metric>_pct_slower：比基线慢超过该百分比 → 违规；
- <metric>_pct_faster：比基线快超过该百分比 → 违规；null 表示不限
  （性能变快是改善不是回归，默认不拦；若需要双向门禁给数值即可）。
- 向后兼容：若只配了旧的 <metric>_pct（对称值），按对称处理。
"""
from __future__ import annotations

import json
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
class BookPerfRun:
    alias: str
    stem: str
    pdf_path: Path
    run_index: int
    wall_sec: float
    user_sec: float
    sys_sec: float
    peak_rss_mib: float
    exit_code: int

    def as_dict(self) -> dict:
        return {
            "alias": self.alias,
            "stem": self.stem,
            "pdf": str(self.pdf_path),
            "run_index": self.run_index,
            "wall_sec": round(self.wall_sec, 3),
            "user_sec": round(self.user_sec, 3),
            "sys_sec": round(self.sys_sec, 3),
            "peak_rss_mib": round(self.peak_rss_mib, 1),
            "exit_code": self.exit_code,
        }


@dataclass
class BookPerfResult:
    alias: str
    stem: str
    pages: int
    runs: List[BookPerfRun]
    median: Dict[str, float]
    baseline: Dict[str, float]
    delta_pct: Dict[str, float]
    within_tolerance: Dict[str, bool]
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "alias": self.alias,
            "stem": self.stem,
            "pages": self.pages,
            "runs": [r.as_dict() for r in self.runs],
            "median": self.median,
            "baseline": self.baseline,
            "delta_pct": self.delta_pct,
            "within_tolerance": self.within_tolerance,
            "errors": self.errors,
        }


@dataclass
class PerfResult:
    books: List[BookPerfResult] = field(default_factory=list)
    baseline_file: str = ""
    verdict: str = "UNKNOWN"        # PASS | FAIL
    violations: int = 0
    section_pass: bool = False
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "books": [b.as_dict() for b in self.books],
            "baseline_file": self.baseline_file,
            "verdict": self.verdict,
            "violations": self.violations,
            "section_pass": self.section_pass,
            "errors": self.errors,
        }


# ── Baseline I/O ────────────────────────────────────────────────────────


def load_baseline(track_root: Path, baseline_name: str) -> Dict[str, Any]:
    path = track_root / baseline_name
    if not path.exists():
        return {}
    return utils.read_json(path)


def write_baseline(track_root: Path, baseline_name: str, data: dict) -> Path:
    path = track_root / baseline_name
    utils.write_json(path, data)
    return path


# ── Directional tolerance ───────────────────────────────────────────────


def _within_directional(d_pct: float, tol: dict, prefix: str) -> bool:
    """
    d_pct > 0 → 比基线慢/高（劣化方向）；d_pct < 0 → 比基线快/低（改善方向）。
    读取 <prefix>_pct_slower / <prefix>_pct_faster；缺省回退对称 <prefix>_pct。
    None 表示该方向不限。
    """
    slower = tol.get(f"{prefix}_pct_slower")
    faster = tol.get(f"{prefix}_pct_faster")
    if slower is None and faster is None:
        sym = float(tol.get(f"{prefix}_pct", 0.0))
        slower = sym
        faster = sym
    if d_pct > 0 and slower is not None and d_pct > float(slower):
        return False
    if d_pct < 0 and faster is not None and -d_pct > float(faster):
        return False
    return True


# ── Single run ──────────────────────────────────────────────────────────


def _run_one(assay_root: Path, pdf: Path, output_dir: Path, gpu_index: int,
             run_index: int, alias: str) -> BookPerfRun:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem
    for f in output_dir.glob(f"{stem}*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    res = utils.run_with_time_v(
        ["python3", "assay.py",
         "-p", str(pdf),
         "-o", str(output_dir),
         "-w", "1", "--gpu", str(gpu_index), "-v"],
        cwd=assay_root,
    )
    return BookPerfRun(
        alias=alias, stem=stem, pdf_path=pdf,
        run_index=run_index,
        wall_sec=res.wall_sec, user_sec=res.user_sec, sys_sec=res.sys_sec,
        peak_rss_mib=res.peak_rss_mib, exit_code=res.exit_code,
    )


# ── Section entry ───────────────────────────────────────────────────────


def run_perf_gate(cfg: dict) -> PerfResult:
    result = PerfResult()
    assay_root = Path(cfg["paths"]["assay_root"])
    pdf_root = Path(cfg["paths"]["pdf_root"])
    track_root = Path(cfg["paths"]["track_root"])
    gpu_index = cfg["gpu"]["allowed_index"]
    runs_per_book = int(cfg["performance"]["runs_per_book"])
    aggregate_name = cfg["performance"]["aggregate"]
    tolerance = cfg["performance"]["tolerance"]
    metric_cfg = cfg["performance"]["metric"]
    pass_cfg = cfg["performance"]["pass_conditions"]
    baseline_name = cfg["performance"]["baseline_file"]

    track_root.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(track_root, baseline_name)
    result.baseline_file = str(track_root / baseline_name)
    if not baseline:
        result.errors.append(
            f"baseline {baseline_name} not found at {result.baseline_file}; "
            f"运行 gate.py --update-baseline 首次生成"
        )
        result.verdict = "FAIL"
        return result

    perf_books = [
        (alias, info) for alias, info in cfg["books"].items()
        if "performance" in info.get("role", [])
    ]

    for alias, info in perf_books:
        stem = info["stem"]
        pages = int(info.get("pages", 0))
        pdf = pdf_root / f"{stem}.pdf"
        if not pdf.exists():
            result.errors.append(f"PDF missing: {pdf}")
            continue
        book_dir = track_root / "perf_runs" / alias
        runs: List[BookPerfRun] = []
        for i in range(runs_per_book):
            run = _run_one(assay_root, pdf, book_dir / f"r{i+1}",
                           gpu_index, run_index=i + 1, alias=alias)
            runs.append(run)
            if run.exit_code != 0:
                result.errors.append(f"{alias} r{i+1} exit={run.exit_code}")
        if not runs:
            continue
        per_run_dicts = [r.as_dict() for r in runs]
        med = utils.aggregate_runs(per_run_dicts, aggregate_name)
        base_for_book = baseline.get("books", {}).get(alias, {})
        delta: Dict[str, float] = {}
        within: Dict[str, bool] = {}
        for metric_key, tol_prefix in (
            ("wall_sec", "wall"),
            ("user_sec", "user"),
            ("peak_rss_mib", "peak_rss"),
        ):
            if not metric_cfg.get(metric_key, {}).get("required"):
                continue
            base_val = base_for_book.get(metric_key + "_median")
            if base_val is None or base_val == 0:
                base_val = base_for_book.get(metric_key)
            if base_val is None:
                continue
            med_val = med.get(metric_key)
            if med_val is None:
                continue
            d_pct = (med_val - base_val) / base_val * 100.0
            delta[f"{metric_key}_pct"] = round(d_pct, 2)
            within[metric_key] = _within_directional(d_pct, tolerance, tol_prefix)
        bpr = BookPerfResult(
            alias=alias, stem=stem, pages=pages, runs=runs,
            median=med, baseline=base_for_book,
            delta_pct=delta, within_tolerance=within,
        )
        result.books.append(bpr)

    violations = sum(
        1 for b in result.books
        for ok in b.within_tolerance.values() if not ok
    )
    result.violations = violations
    if violations <= pass_cfg["within_tolerance_max_violations"]:
        result.verdict = "PASS"
        result.section_pass = True
    else:
        result.verdict = "FAIL"
    return result


# ── Update baseline (operator) ──────────────────────────────────────────


def update_baseline(cfg: dict) -> Path:
    """
    生成 / 更新 baseline.json：每个 performance role 的书跑 runs_per_book 次，采
    /usr/bin/time -v 的 wall/user/sys/peak_rss，落盘为 baseline.json。
    """
    assay_root = Path(cfg["paths"]["assay_root"])
    pdf_root = Path(cfg["paths"]["pdf_root"])
    track_root = Path(cfg["paths"]["track_root"])
    gpu_index = cfg["gpu"]["allowed_index"]
    runs_per_book = int(cfg["performance"]["runs_per_book"])
    aggregate_name = cfg["performance"]["aggregate"]
    baseline_name = cfg["performance"]["baseline_file"]

    track_root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {
        "meta": {
            "tag": "p1_baseline",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "machine": "home (Intel Ultra 7 265K 20c / 46GB / GPU0 RTX PRO 6000 96GB)",
            "code_rev": _git_head(assay_root),
            "runs_per_book": runs_per_book,
            "aggregate": aggregate_name,
        },
        "books": {},
    }

    perf_books = [
        (alias, info) for alias, info in cfg["books"].items()
        if "performance" in info.get("role", [])
    ]
    for alias, info in perf_books:
        stem = info["stem"]
        pages = int(info.get("pages", 0))
        pdf = pdf_root / f"{stem}.pdf"
        if not pdf.exists():
            continue
        book_dir = track_root / "perf_runs" / alias
        runs: List[BookPerfRun] = []
        for i in range(runs_per_book):
            run = _run_one(assay_root, pdf, book_dir / f"baseline_r{i+1}",
                           gpu_index, run_index=i + 1, alias=alias)
            runs.append(run)
        if not runs:
            continue
        per_run = [r.as_dict() for r in runs]
        med = utils.aggregate_runs(per_run, aggregate_name)
        out["books"][alias] = {
            "alias": alias, "stem": stem, "pages": pages,
            "wall_sec_median": med.get("wall_sec"),
            "user_sec_median": med.get("user_sec"),
            "sys_sec_median": med.get("sys_sec"),
            "peak_rss_mib_median": med.get("peak_rss_mib"),
            "wall_sec_runs": [r["wall_sec"] for r in per_run],
            "user_sec_runs": [r["user_sec"] for r in per_run],
            "peak_rss_mib_runs": [r["peak_rss_mib"] for r in per_run],
        }
    path = write_baseline(track_root, baseline_name, out)
    return path


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True, timeout=10
        ).strip()
    except Exception:
        return "unknown"
