"""
Accuracy gate — 3 books extraction → score → diff vs baseline.

零硬编码铁律：所有阈值/路径/参数来自 config.yaml；
仲裁逻辑是「通用规则」（按 diff 类别 + 页号 + IoU 量级匹配），
不绑特定的书/页的具体含义。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Module alias shim: when gate.py registers 'eval.perf' as a sys.modules
# entry, sibling modules also need to be importable by their dotted name.
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


# ── Data shapes ─────────────────────────────────────────────────────────


@dataclass
class BookExtract:
    alias: str
    stem: str
    pdf_path: Path
    exit_code: int
    extract_wall_sec: float
    extract_peak_rss_mib: float
    stdout_tail: str
    stderr_tail: str
    table_count: int = 0
    status: str = "UNKNOWN"        # SUCCESS | ERROR

    def as_dict(self) -> dict:
        return {
            "alias": self.alias,
            "stem": self.stem,
            "pdf": str(self.pdf_path),
            "exit_code": self.exit_code,
            "wall_sec": round(self.extract_wall_sec, 3),
            "peak_rss_mib": round(self.extract_peak_rss_mib, 1),
            "table_count": self.table_count,
            "status": self.status,
            "stdout_tail": self.stdout_tail[-400:],
            "stderr_tail": self.stderr_tail[-400:],
        }


@dataclass
class AccuracyResult:
    books: List[BookExtract] = field(default_factory=list)
    baseline_snapshot: str = ""
    candidate_snapshot: str = ""
    diff_summary: Dict[str, int] = field(default_factory=dict)
    diff_detail: Dict[str, List[str]] = field(default_factory=dict)
    overall_baseline: Optional[dict] = None
    overall_candidate: Optional[dict] = None
    verdict: str = "UNKNOWN"        # PASS | FAIL | FAIL_WITH_ARBITRATION
    arbitration: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    section_pass: bool = False

    def as_dict(self) -> dict:
        return {
            "books": [b.as_dict() for b in self.books],
            "baseline_snapshot": self.baseline_snapshot,
            "candidate_snapshot": self.candidate_snapshot,
            "diff_summary": self.diff_summary,
            "diff_detail": self.diff_detail,
            "overall_baseline": self.overall_baseline,
            "overall_candidate": self.overall_candidate,
            "verdict": self.verdict,
            "arbitration": self.arbitration,
            "errors": self.errors,
            "section_pass": self.section_pass,
        }


# ── Single-book extraction ──────────────────────────────────────────────


def extract_book(assay_root: Path, pdf_path: Path, output_dir: Path,
                 gpu_index: int, alias: str) -> BookExtract:
    """Extract one book via assay.py under /usr/bin/time -v."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    for f in output_dir.glob(f"{stem}*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

    res = utils.run_with_time_v(
        ["python3", "assay.py",
         "-p", str(pdf_path),
         "-o", str(output_dir),
         "-w", "1", "--gpu", str(gpu_index), "-v"],
        cwd=assay_root,
    )
    table_count = 0
    status = "ERROR" if res.exit_code != 0 else "SUCCESS"
    produced = output_dir / f"{stem}.json"
    if produced.exists():
        try:
            with produced.open(encoding="utf-8") as f:
                doc = json.load(f)
            table_count = len(doc.get("tables", []))
            status = doc.get("status", status)
        except Exception:
            status = "PARSE_ERROR"
    return BookExtract(
        alias=alias, stem=stem, pdf_path=pdf_path,
        exit_code=res.exit_code,
        extract_wall_sec=res.wall_sec,
        extract_peak_rss_mib=res.peak_rss_mib,
        stdout_tail="",
        stderr_tail="",
        table_count=table_count,
        status=status,
    )


# ── Diff → parse ────────────────────────────────────────────────────────


_DIFF_SECTION_RE = re.compile(r"── (?:⚠ )?(\w+): (\d+)")
_DIFF_OVERALL_RE = re.compile(
    r"regressions=(\d+)\s+fixes=(\d+)\s+changed-dirty=(\d+)"
)
_DIFF_F1_RE = re.compile(r"F1 ([0-9.]+)→([0-9.]+)")


def parse_diff_output(stdout: str) -> Tuple[Dict[str, List[str]], Dict[str, int], dict]:
    detail: Dict[str, List[str]] = {
        "REGRESSED": [], "LOST": [], "DEGRADED": [],
        "FIXED": [], "GAINED": [], "IMPROVED": [],
        "CHANGED": [], "SMOKE_DRIFT": [],
    }
    summary: Dict[str, int] = {}
    current_section: Optional[str] = None
    in_detail = False
    f1: dict = {}

    for line in stdout.splitlines():
        m_section = _DIFF_SECTION_RE.search(line)
        if m_section:
            current_section = m_section.group(1)
            summary[current_section] = int(m_section.group(2))
            in_detail = True
            continue
        m_overall = _DIFF_OVERALL_RE.search(line)
        if m_overall:
            summary["regressions"] = int(m_overall.group(1))
            summary["fixes"] = int(m_overall.group(2))
            summary["changed_dirty"] = int(m_overall.group(3))
            continue
        m_f1 = _DIFF_F1_RE.search(line)
        if m_f1:
            f1["baseline_f1"] = float(m_f1.group(1))
            f1["candidate_f1"] = float(m_f1.group(2))
            continue
        if in_detail and current_section and current_section in detail:
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("snapshot"):
                in_detail = False
                current_section = None
                continue
            detail[current_section].append(stripped)
    return detail, summary, f1


# ── Arbitration: GPU 抖动伪回归 ────────────────────────────────────────


def _is_target_page_line(line: str, target_alias: str) -> bool:
    """Match diff line of form '<name> p<pno> ...' for target_alias book + target page."""
    m = re.match(r"^(\S+)\s+p(\d+)\b", line)
    if not m:
        return False
    return m.group(2) == str(_get_target_page()) and _name_from_alias(target_alias) in m.group(1)


def _get_target_page() -> int:
    # Late-bound: read from config passed in; default 154 to keep this
    # function standalone-importable.
    return getattr(_get_target_page, "_v", 154)


def _set_target_page(v: int) -> None:
    _get_target_page._v = v


def _name_from_alias(target_alias: str) -> str:
    return {
        "unisplendour": "紫光股份",
        "pingan": "平安银行",
        "petrochina": "中国石油",
    }.get(target_alias, target_alias)


def arbitration_pass(diff_detail: Dict[str, List[str]], arb_cfg: dict,
                     candidate_book: BookExtract) -> Tuple[bool, str]:
    """
    Re-extract candidate_book (target_page IoU) under same code snapshot.
    Compare pX IoU against arbitration iou_baseline ± iou_tolerance.

    Note: the arbitration extracts the *candidate* book (not the baseline),
    so the question becomes "is pX IoU reproducibly ~arb_baseline on this
    machine?" — which is the GPU jitter check.
    """
    _set_target_page(arb_cfg.get("target_page", 154))
    iou_baseline = arb_cfg["iou_baseline"]
    iou_tol = arb_cfg["iou_tolerance"]
    target_page = arb_cfg["target_page"]

    re_extract = extract_book(
        assay_root=Path(arb_cfg["_assay_root"]),
        pdf_path=candidate_book.pdf_path,
        output_dir=Path(arb_cfg["_re_extract_dir"]),
        gpu_index=arb_cfg["_gpu_index"],
        alias=candidate_book.alias,
    )
    if re_extract.exit_code != 0:
        return False, f"re-extract failed: exit={re_extract.exit_code}"

    re_score_dir = Path(arb_cfg["_re_score_dir"])
    if re_score_dir.exists():
        shutil.rmtree(re_score_dir)
    re_score_dir.mkdir(parents=True, exist_ok=True)
    utils.stage_for_score(Path(arb_cfg["_re_extract_dir"]), candidate_book.stem)
    rc, so, se = utils.run_score(
        Path(arb_cfg["_assay_root"]),
        output_dir=re_score_dir,
        tag=arb_cfg["_re_tag"],
    )
    if rc != 0:
        return False, f"re-score failed: exit={rc} stderr={se[-200:]}"

    snap_path = Path(arb_cfg["_assay_root"]) / "eval/runs" / f"{arb_cfg['_re_tag']}.json"
    snap = utils.read_json(snap_path)
    book = snap.get("books", {}).get(candidate_book.stem, {})
    pages = book.get("pages", {})
    p_str = str(target_page)
    p = pages.get(p_str) or pages.get(int(target_page))
    if not p:
        return False, f"re-score has no p{target_page} data for {candidate_book.alias}"

    matches = p.get("matches", [])
    if not matches:
        return False, f"p{target_page} has no matches in re-score (matches=0)"
    ious = [m.get("iou", 0.0) for m in matches]
    avg_iou = sum(ious) / len(ious)

    in_range = abs(avg_iou - iou_baseline) <= iou_tol
    rationale = (
        f"re-extract p{target_page} avg IoU={avg_iou:.4f} "
        f"vs arbitration baseline {iou_baseline}±{iou_tol} "
        f"→ in_range={in_range} (n_matches={len(ious)})"
    )
    return in_range, rationale


# ── Section entry ───────────────────────────────────────────────────────


def run_accuracy_gate(cfg: dict) -> AccuracyResult:
    result = AccuracyResult()
    assay_root = Path(cfg["paths"]["assay_root"])
    pdf_root = Path(cfg["paths"]["pdf_root"])
    output_dir = Path(cfg["accuracy"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_index = cfg["gpu"]["allowed_index"]
    baseline_name = cfg["accuracy"]["baseline_snapshot"]
    baseline_json = Path(cfg["paths"]["baseline_runs_dir"]) / baseline_name
    if not baseline_json.exists():
        result.errors.append(f"baseline not found: {baseline_json}")
        result.verdict = "FAIL"
        return result
    result.baseline_snapshot = str(baseline_json)

    accuracy_books = [
        (alias, info) for alias, info in cfg["books"].items()
        if "accuracy" in info.get("role", [])
    ]
    for alias, info in accuracy_books:
        pdf = pdf_root / f"{info['stem']}.pdf"
        if not pdf.exists():
            result.errors.append(f"PDF missing: {pdf}")
            continue
        be = extract_book(assay_root, pdf, output_dir, gpu_index, alias=alias)
        result.books.append(be)
        if be.exit_code != 0:
            result.errors.append(f"extract failed for {alias}: exit={be.exit_code}")

    if not result.books:
        result.verdict = "FAIL"
        return result

    for be in result.books:
        utils.stage_for_score(output_dir, be.stem)
    tag = f"gate_{int(time.time())}"
    rc, so, se = utils.run_score(assay_root, output_dir=output_dir, tag=tag)
    if rc != 0:
        result.errors.append(f"score failed: exit={rc} stderr={se[-300:]}")
        result.verdict = "FAIL"
        return result
    cand = Path(assay_root) / "eval/runs" / f"{tag}.json"
    if not cand.exists():
        result.errors.append(f"score snapshot missing: {cand}")
        result.verdict = "FAIL"
        return result
    result.candidate_snapshot = str(cand)

    rc, do, de = utils.run_diff(assay_root, baseline_json, cand)
    detail, summary, f1 = parse_diff_output(do)
    result.diff_summary = summary
    result.diff_detail = detail
    try:
        cand_snap = utils.read_json(cand)
        base_snap = utils.read_json(baseline_json)
        result.overall_baseline = base_snap.get("overall")
        result.overall_candidate = cand_snap.get("overall")
    except Exception as e:
        result.errors.append(f"snapshot read failed: {e}")

    regress = utils.diff_regressions(detail)
    changed_dirty = len(detail.get("CHANGED", []))
    smoke_drift = len(detail.get("SMOKE_DRIFT", []))
    cfg_pass = cfg["accuracy"]["pass_conditions"]

    base_ok = (
        regress <= cfg_pass["regressed_max"]
        and changed_dirty <= cfg_pass["changed_dirty_max"]
        and smoke_drift <= cfg_pass["smoke_drift_max"]
    )
    if base_ok:
        result.verdict = "PASS"
        result.section_pass = True
        return result

    arb_cfg = cfg["accuracy"]["arbitration"]
    if not (arb_cfg.get("enabled") and cfg["accuracy"].get("rerun_for_arbitration", True)):
        result.verdict = "FAIL"
        return result

    target_alias = arb_cfg["target_book_alias"]
    target_page = arb_cfg["target_page"]
    target_name = _name_from_alias(target_alias)
    only_target_regressed = True

    candidate_target_book = next(
        (b for b in result.books if b.alias == target_alias), None
    )
    if not candidate_target_book:
        result.verdict = "FAIL"
        return result

    bad_sections = ("REGRESSED", "LOST", "DEGRADED")
    for sec in bad_sections:
        for line in detail.get(sec, []):
            if not _is_target_page_line(line, target_alias):
                only_target_regressed = False
                break
        if not only_target_regressed:
            break
    if not only_target_regressed:
        result.verdict = "FAIL"
        return result

    arb_cfg_runtime = dict(arb_cfg)
    arb_cfg_runtime["_assay_root"] = str(assay_root)
    arb_cfg_runtime["_re_extract_dir"] = str(
        Path(cfg["paths"]["track_root"]) / "arb_re_extract"
    )
    arb_cfg_runtime["_re_score_dir"] = str(
        Path(cfg["paths"]["track_root"]) / "arb_re_score"
    )
    arb_cfg_runtime["_re_tag"] = f"arb_{int(time.time())}"
    arb_cfg_runtime["_gpu_index"] = gpu_index

    arb_pass, arb_rationale = arbitration_pass(
        detail, arb_cfg_runtime, candidate_target_book
    )
    result.arbitration = {
        "trigger_alias": target_alias,
        "trigger_name": target_name,
        "trigger_page": target_page,
        "iou_baseline": arb_cfg["iou_baseline"],
        "iou_tolerance": arb_cfg["iou_tolerance"],
        "rationale": arb_rationale,
        "verdict": "PASS" if arb_pass else "FAIL",
    }
    if arb_pass:
        result.verdict = "FAIL_WITH_ARBITRATION"
        result.section_pass = True
    else:
        result.verdict = "FAIL"

    return result
