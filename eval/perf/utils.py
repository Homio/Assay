"""
Assay perf harness — shared utilities.

零硬编码铁律：所有阈值/路径/参数必须来自 config.yaml；本模块只提供
通用机制（GPU 检查、ps 采样、JSON 落盘、time -v 解析），不绑特定
的书或页号。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ── Config loader ───────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    _require(cfg, ["paths", "gpu", "books", "accuracy", "performance", "memory", "driver"])
    return cfg


def _require(cfg: dict, keys: list) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")


# ── GPU check ───────────────────────────────────────────────────────────


@dataclass
class GpuStatus:
    index: int
    name: str
    used_mib: int
    total_mib: int
    util_pct: int

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


def query_gpu(index: int) -> GpuStatus:
    """Return current GPU status via nvidia-smi. No hard-coded fallback."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, timeout=10)
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and int(parts[0]) == index:
            return GpuStatus(
                index=int(parts[0]),
                name=parts[1],
                used_mib=int(parts[3]),
                total_mib=int(parts[4]),
                util_pct=int(parts[2]),
            )
    raise RuntimeError(f"GPU index {index} not present in nvidia-smi output")


def assert_gpu_free(cfg: dict) -> GpuStatus:
    """Pre-check GPU0 is free enough; fail-fast if pre_check_max_used_mib exceeded."""
    gpu = cfg["gpu"]
    st = query_gpu(gpu["allowed_index"])
    if st.used_mib > gpu["pre_check_max_used_mib"]:
        raise RuntimeError(
            f"GPU{st.index} ({st.name}) 已占用 {st.used_mib}MiB "
            f"> 上限 {gpu['pre_check_max_used_mib']}MiB；"
            f"GPU{gpu['reserved_index']} 永远不许触碰。"
        )
    return st


# ── time -v wrapper ─────────────────────────────────────────────────────


@dataclass
class TimeVResult:
    wall_sec: float
    user_sec: float
    sys_sec: float
    peak_rss_mib: float
    exit_code: int

    def as_dict(self) -> dict:
        return {
            "wall_sec": round(self.wall_sec, 3),
            "user_sec": round(self.user_sec, 3),
            "sys_sec": round(self.sys_sec, 3),
            "peak_rss_mib": round(self.peak_rss_mib, 1),
            "exit_code": self.exit_code,
        }


_TIME_V_RE = {
    "wall": re.compile(r"^\s*Elapsed \(wall clock\) time.*:\s*([0-9:.]+)"),
    "user": re.compile(r"^\s*User time.*:\s*([0-9.]+)"),
    "sys": re.compile(r"^\s*System time.*:\s*([0-9.]+)"),
    "rss": re.compile(r"^\s*Maximum resident set size.*:\s*([0-9]+)"),
}


def run_with_time_v(cmd: list, cwd: str | Path, env: Optional[dict] = None) -> TimeVResult:
    """Run cmd under /usr/bin/time -v; return parsed wall/user/sys/rss."""
    if not shutil.which("/usr/bin/time"):
        raise RuntimeError("/usr/bin/time not found (need util-linux)")
    full = ["/usr/bin/time", "-v"] + list(cmd)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        full, cwd=str(cwd), env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    parsed = {"wall": None, "user": None, "sys": None, "rss": None}
    for line in proc.stderr.splitlines():
        for k, rx in _TIME_V_RE.items():
            if parsed[k] is not None:
                continue
            m = rx.match(line)
            if not m:
                continue
            if k == "wall":
                # h:mm:ss or m:ss.ss
                raw = m.group(1)
                parts = raw.split(":")
                if len(parts) == 3:
                    parsed["wall"] = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    parsed["wall"] = int(parts[0]) * 60 + float(parts[1])
                else:
                    parsed["wall"] = float(parts[0])
            elif k == "rss":
                parsed["rss"] = int(m.group(1)) / 1024.0  # KB → MiB
            else:
                parsed[k] = float(m.group(1))
    missing = [k for k, v in parsed.items() if v is None]
    if missing:
        raise RuntimeError(
            f"/usr/bin/time -v parse failed for {missing}; "
            f"stderr (head)={proc.stderr.splitlines()[:8]!r} exit={proc.returncode}"
        )
    return TimeVResult(
        wall_sec=parsed["wall"],
        user_sec=parsed["user"],
        sys_sec=parsed["sys"],
        peak_rss_mib=parsed["rss"],
        exit_code=proc.returncode,
    )


# ── Process-tree RSS sampler ────────────────────────────────────────────


def tree_rss_mib(root_pid: int) -> Tuple[int, float]:
    """
    Sum RSS (MiB) over root_pid and its whole descendant tree.

    为什么按 ppid 树聚合而不是按 cmdline 过滤：assay.py 用
    torch.multiprocessing spawn 起 worker，worker 的 cmdline 是
    `python3 -c "from multiprocessing.spawn import spawn_main; ..."`
    不含 "assay" 字样——按 cmdline 过滤会把 12 个 worker 全部漏掉。
    ppid 树聚合对 spawn/fork/再嵌套的子进程都鲁棒。

    Returns (n_pids, total_rss_mib). root 已退出时返回 (0, 0.0)。
    """
    try:
        out = subprocess.check_output(
            ["ps", "-e", "-o", "pid=,ppid=,rss="], text=True, timeout=10,
        )
    except Exception:
        return 0, 0.0
    children: Dict[int, List[int]] = {}
    rss_kb: Dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss_kb[pid] = rss
    total_kb = 0
    n = 0
    stack = [root_pid]
    seen = set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in rss_kb:
            total_kb += rss_kb[p]
            n += 1
        stack.extend(children.get(p, []))
    return n, total_kb / 1024.0


@dataclass
class ProcessSample:
    pid: int
    rss_mib: float
    cmd: str


def sample_processes(predicate: Optional[callable] = None,
                     include_children: bool = True) -> List[ProcessSample]:
    """Sample current process tree. Predicate filters by cmd substring."""
    if include_children:
        cmd = ["ps", "-e", "-o", "pid=,ppid=,rss=,args="]
    else:
        cmd = ["ps", "-o", "pid=,rss=,args="]
    out = subprocess.check_output(cmd, text=True, timeout=10)
    samples: List[ProcessSample] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        if include_children:
            pid, ppid, rss, args = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
        else:
            pid, rss, args = int(parts[0]), int(parts[1]), parts[2]
        if predicate and not predicate(args):
            continue
        samples.append(ProcessSample(pid=pid, rss_mib=rss / 1024.0, cmd=args[:200]))
    return samples


# ── JSON I/O ────────────────────────────────────────────────────────────


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ── Median / aggregate ──────────────────────────────────────────────────


def median(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate_runs(runs: List[dict], aggregate: str) -> dict:
    """Apply aggregate function to each numeric field. Returns dict-of-aggregated."""
    if not runs:
        return {}
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        if aggregate == "median":
            out[k] = round(median(vals), 4)
        elif aggregate == "mean":
            out[k] = round(sum(vals) / len(vals), 4)
        elif aggregate == "min":
            out[k] = round(min(vals), 4)
        else:
            raise ValueError(f"unknown aggregate: {aggregate}")
        out[f"{k}__runs"] = [round(v, 4) for v in vals]
    return out


# ── Diff summary helpers ────────────────────────────────────────────────


def diff_regressions(diff_result: dict) -> int:
    """Total regression count = REGRESSED + LOST + DEGRADED."""
    return (
        len(diff_result.get("REGRESSED", []))
        + len(diff_result.get("LOST", []))
        + len(diff_result.get("DEGRADED", []))
    )


def diff_fixes(diff_result: dict) -> int:
    return (
        len(diff_result.get("FIXED", []))
        + len(diff_result.get("GAINED", []))
        + len(diff_result.get("IMPROVED", []))
    )


# ── Subprocess for assay.py ─────────────────────────────────────────────


def run_assay_single(assay_root: Path, pdf_path: Path, output_dir: Path,
                     gpu_index: int, env: Optional[dict] = None) -> Tuple[int, str, str]:
    """Run assay.py -p <pdf> -o <out> --gpu 0. Returns (exit, stdout, stderr)."""
    cmd = [
        "python3", "assay.py",
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "-w", "1",         # single-file path doesn't matter but keep consistent
        "--gpu", str(gpu_index),
        "-v",
    ]
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd, cwd=str(assay_root), env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_score(assay_root: Path, output_dir: Path, tag: str) -> Tuple[int, str, str]:
    cmd = [
        "python3", "eval/score.py",
        "--output-dir", str(output_dir),
        "--tag", tag,
        "--quiet",
    ]
    proc = subprocess.run(
        cmd, cwd=str(assay_root), env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_diff(assay_root: Path, baseline_json: Path, candidate_json: Path) -> Tuple[int, str, str]:
    cmd = [
        "python3", "eval/diff.py",
        str(baseline_json), str(candidate_json),
        "--strict",
    ]
    proc = subprocess.run(
        cmd, cwd=str(assay_root), env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ── Extracted-json staging for score.py ─────────────────────────────────


def stage_for_score(extract_root: Path, stem: str) -> Path:
    """
    assay.py 单文件模式把 <stem>.json 写到 extract_root 根目录；score.py 读
    <extract_root>/<stem>/<stem>.json。本函数把根目录下的 .json/.md 移到
    子目录里。返回 score 后会在 eval/runs/<tag>.json 落盘。
    """
    sub = extract_root / stem
    sub.mkdir(parents=True, exist_ok=True)
    for ext in (".json", ".md"):
        src = extract_root / f"{stem}{ext}"
        dst = sub / f"{stem}{ext}"
        if src.exists() and src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
    return sub
