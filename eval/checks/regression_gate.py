#!/usr/bin/env python3
"""
回归门禁脚本 — Assay C1-C8 修复阶段验收工具
=============================================

用法:
    # 门禁模式（三书零红色判定）
    python3 eval/checks/regression_gate.py \
        eval/runs/r56_13books.json eval/runs/r57_13books.json

    # 自检模式
    python3 eval/checks/regression_gate.py --self-test

    # 指定书名过滤（仅检查指定的几本书）
    python3 eval/checks/regression_gate.py baseline.json candidate.json \
        --books 中国石油 平安银行 紫光股份

返回码:
    0 = PASS（零 REGRESSED/LOST/DEGRADED，或在允许清单内）
    1 = FAIL（存在不允许的红色项）
    2 = 脚本自身错误

设计原则:
    - 依赖 eval/diff.py 的 diff() 函数（同 PYTHONPATH）
    - 不重复实现 diff 逻辑
    - 自检模式独立验证脚本正确性
    - 允许清单机制：已知的结构改善副作用可豁免（需逐条登记理由）
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


# ── 豁免清单 ──────────────────────────────────────────────────
# 格式: (book_name, page, gt_index, reason)
# 仅登记已验证为"结构改善副作用"或"度量口径变化"的项
# 每项必须有 CTO 签字或视觉复核证据链接
ALLOWLIST: Set[Tuple[str, str, str, str]] = {
    # 示例（当前为空，各轨道修复后 CTO 逐条审核追加）:
    # ("平安银行", "8", "*", "度量口径：cov 下降因标题从表体拆分，内容无损 — r56 视觉复核"),
    # ("紫光股份", "250", "0", "C2 页眉分离导致 trec 下降 0.03，表格内容完整 — PM 复核"),
}


# ── 核心逻辑 ──────────────────────────────────────────────────

def _resolve_repo_root() -> Path:
    """定位 Assay 仓库根目录。"""
    candidates = [
        Path("/home/homio/sda/1111/Assay"),
    ]
    for c in candidates:
        if (c / "eval" / "diff.py").exists():
            return c
    # 回退：从当前脚本位置推断
    script_dir = Path(__file__).resolve().parent.parent.parent
    if (script_dir / "eval" / "diff.py").exists():
        return script_dir
    raise RuntimeError("无法定位 Assay 仓库根目录（找不到 eval/diff.py）")


def run_diff(baseline: str, candidate: str, repo_root: Path) -> dict:
    """调用 eval/diff.py 并解析 JSON 输出。"""
    diff_script = repo_root / "eval" / "diff.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, str(diff_script), baseline, candidate],
        capture_output=True, text=True, timeout=120, cwd=str(repo_root), env=env
    )
    if result.returncode not in (0, 1):  # diff.py exit 1 on --strict regressions
        print(f"[ERROR] diff.py 异常退出 (code={result.returncode}):",
              file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    # 从文本输出解析 diff 结果（diff.py 没有 --json 模式，解析其标准输出）
    return _parse_diff_output(result.stdout)


def _parse_diff_output(output: str) -> dict:
    """解析 diff.py 的文本输出为结构化字典。"""
    sections = {}
    current_section = None
    for line_raw in output.split("\n"):
        line = line_raw.strip()
        if not line:
            continue
        if line.startswith("── "):
            # e.g. "── ⚠ REGRESSED: 2" or "── FIXED: 5"
            clean = line[3:].lstrip()  # strip "── "
            if clean.startswith("⚠ "):
                clean = clean[2:].lstrip()
            if ":" in clean:
                parts = clean.split(":", 1)
                current_section = parts[0].strip()
                sections[current_section] = []
        elif line.startswith("=") and len(line) > 10:
            # separator line — stop collecting items for current section
            current_section = None
        elif current_section:
            # items are indented with 2 spaces in the raw line
            if line_raw.startswith("  ") and "..." not in line_raw[:6]:
                sections[current_section].append(line.strip())
            elif "regressions=" in line or "F1 " in line:
                continue

    # 汇总
    regressed = sections.get("REGRESSED", [])
    lost = sections.get("LOST", [])
    degraded = sections.get("DEGRADED", [])

    return {
        "REGRESSED": regressed,
        "LOST": lost,
        "DEGRADED": degraded,
        "FIXED": sections.get("FIXED", []),
        "GAINED": sections.get("GAINED", []),
        "IMPROVED": sections.get("IMPROVED", []),
        "CHANGED": sections.get("CHANGED", []),
        "SMOKE_DRIFT": sections.get("SMOKE_DRIFT", []),
        "total_red": len(regressed) + len(lost) + len(degraded),
        "total_green": (len(sections.get("FIXED", [])) +
                        len(sections.get("GAINED", [])) +
                        len(sections.get("IMPROVED", []))),
    }


def _parse_item(item: str) -> Optional[Tuple[str, str, str]]:
    """解析单条 diff 项为 (book_name, page, gt_index)。

    示例输入:
        "平安银行 p122: +0/-1"
        "紫光股份 p250 gt#0: trec 1.00→0.95"
        "平安银行 p122 gt#3 (was IoU=0.85)"
    """
    parts = item.split(None, 1)
    if not parts:
        return None
    identifier = parts[0]  # e.g. "平安银行" or "平安银行p122"
    book = identifier
    page = "*"
    # 尝试分离书名和页码
    import re
    m = re.match(r"^(.+?)\s*p(\d+)$", identifier)
    if m:
        book = m.group(1)
        page = m.group(2)
    else:
        m = re.match(r"^(.+?)\s*p(\d+)\s*gt#(\d+)$", identifier)
        if m:
            book = m.group(1)
            page = m.group(2)
        elif "gt#" in item:
            # Try to extract from the full line
            m2 = re.match(r"^(.+?)\s*p(\d+)\s*gt#(\d+)", item)
            if m2:
                book = m2.group(1)
                page = m2.group(2)
    # 提取 gt# index
    gt_match = re.search(r"gt#(\d+)", item)
    gt_index = gt_match.group(1) if gt_match else "*"

    return (book, page, gt_index)


def check_allowlist(item: str) -> Tuple[bool, str]:
    """检查某条红色项是否在豁免清单中。"""
    parsed = _parse_item(item)
    if parsed is None:
        return False, ""
    book, page, gt = parsed
    for abook, apage, agt, reason in ALLOWLIST:
        if (abook == book or abook == "*") and \
           (apage == page or apage == "*") and \
           (agt == gt or agt == "*"):
            return True, reason
    return False, ""


def filter_books(items: List[str], books: List[str]) -> List[str]:
    """过滤只保留指定书名的项。"""
    if not books:
        return items
    return [item for item in items
            if any(book_name in item.split(None, 1)[0] for book_name in books)]


def gate(baseline: str, candidate: str, books: Optional[List[str]] = None,
         repo_root: Optional[Path] = None) -> Tuple[bool, dict]:
    """执行门禁判定。

    Returns:
        (passed, report_dict)
    """
    if repo_root is None:
        repo_root = _resolve_repo_root()

    result = run_diff(baseline, candidate, repo_root)

    # 可选书名过滤
    red_categories = {
        "REGRESSED": result["REGRESSED"],
        "LOST": result["LOST"],
        "DEGRADED": result["DEGRADED"],
    }

    # 分离豁免与未豁免
    exempted = {}
    violations = {}
    for cat, items in red_categories.items():
        filtered = filter_books(items, books or [])
        exempted[cat] = []
        violations[cat] = []
        for item in filtered:
            ok, reason = check_allowlist(item)
            if ok:
                exempted[cat].append((item, reason))
            else:
                violations[cat].append(item)

    total_violations = sum(len(v) for v in violations.values())
    total_exempted = sum(len(e) for e in exempted.values())

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_tag": _get_tag(baseline, repo_root),
        "candidate_tag": _get_tag(candidate, repo_root),
        "total_red": result["total_red"],
        "total_green": result["total_green"],
        "violations": {k: v for k, v in violations.items() if v},
        "exempted": {k: v for k, v in exempted.items() if v},
        "total_violations": total_violations,
        "total_exempted": total_exempted,
        "passed": total_violations == 0,
        "all_items": result,
    }

    return total_violations == 0, report


def _get_tag(json_path: str, repo_root: Path) -> str:
    """从快照 JSON 中读取 tag。"""
    try:
        p = Path(json_path)
        if not p.is_absolute():
            p = repo_root / p
        with open(p) as f:
            data = json.load(f)
        return data.get("meta", {}).get("tag", "?")
    except Exception:
        return "?"


def format_report(report: dict) -> str:
    """格式化门禁报告为文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  基线: {report['baseline']} ({report['baseline_tag']})")
    lines.append(f"  候选: {report['candidate']} ({report['candidate_tag']})")
    lines.append("=" * 60)

    if report["passed"]:
        lines.append("\n✅ 门禁 PASS — 零不允许的红色项")
    else:
        lines.append(f"\n❌ 门禁 FAIL — {report['total_violations']} 条不允许的红色项")

    for cat in ("REGRESSED", "LOST", "DEGRADED"):
        violations = report["violations"].get(cat, [])
        exempted = report["exempted"].get(cat, [])
        if violations:
            lines.append(f"\n  ⚠ {cat} ({len(violations)} 条):")
            for item in violations[:20]:
                lines.append(f"    {item}")
            if len(violations) > 20:
                lines.append(f"    ... 及另外 {len(violations) - 20} 条")
        if exempted:
            lines.append(f"\n  ✓ {cat} 豁免 ({len(exempted)} 条):")
            for item, reason in exempted:
                lines.append(f"    {item} — {reason}")

    if report["total_green"]:
        lines.append(f"\n  修复 / 改善: {report['total_green']} 条")

    lines.append(f"\n  综合: 红色 {report['total_red']} (违规 {report['total_violations']}"
                 f" + 豁免 {report['total_exempted']}) |"
                 f" 绿色 {report['total_green']}")

    return "\n".join(lines)


# ── 自检 ──────────────────────────────────────────────────────

def self_test(repo_root: Optional[Path] = None) -> bool:
    """自检：r56 vs r56 必须 PASS、r56 vs r51 必须 FAIL。

    Returns:
        True if both self-tests pass (脚本正确).
    """
    if repo_root is None:
        repo_root = _resolve_repo_root()

    runs_dir = repo_root / "eval" / "runs"
    r56 = str(runs_dir / "r56_13books.json")
    r51 = str(runs_dir / "r51_13books.json")

    print("=" * 60)
    print("  回归门禁自检")
    print("=" * 60)

    all_ok = True

    # 测试 1: r56 vs r56 — 必须零红色
    print("\n[TEST 1] r56 vs r56 (预期: PASS)")
    passed, report = gate(r56, r56, repo_root=repo_root)
    print(format_report(report))
    if not passed:
        print("❌ 自检失败: r56 vs r56 居然报红 — 脚本可能有 bug")
        all_ok = False
    elif report["total_red"] != 0:
        print("❌ 自检失败: r56 vs r56 同文件 diff 应为 0 条红，"
              f"实为 {report['total_red']} 条")
        all_ok = False
    else:
        print("✅ 自检通过: r56 vs r56 零红色")

    # 测试 2: r56 vs r51 — 必须报红
    print("\n[TEST 2] r56 vs r51 (预期: FAIL)")
    passed2, report2 = gate(r56, r51, repo_root=repo_root)
    print(format_report(report2))
    if passed2:
        print("❌ 自检失败: r56 vs r51 应检测到差异（已知有 7 DEGRADED），"
              "但门禁判 PASS")
        all_ok = False
    elif report2["total_violations"] == 0:
        print("❌ 自检失败: r56 vs r51 未检测到任何违规项")
        all_ok = False
    else:
        print(f"✅ 自检通过: 正确检测到 {report2['total_violations']} 条违规")

    print("\n" + "=" * 60)
    if all_ok:
        print("  自检结论: 全部通过 — 门禁脚本工作正常")
    else:
        print("  自检结论: 存在问题 — 门禁脚本需排查")
    print("=" * 60)

    return all_ok


# ── CLI ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assay 回归门禁 — 三书 diff 零红色自动判定")
    parser.add_argument("baseline", nargs="?", help="基线快照 JSON 路径")
    parser.add_argument("candidate", nargs="?", help="候选快照 JSON 路径")
    parser.add_argument("--self-test", action="store_true",
                        help="运行自检（r56 vs r56 + r56 vs r51）")
    parser.add_argument("--books", nargs="+",
                        help="仅检查指定书名（默认全量）")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出报告")
    args = parser.parse_args()

    repo_root = _resolve_repo_root()

    if args.self_test:
        ok = self_test(repo_root)
        return 0 if ok else 1

    if not args.baseline or not args.candidate:
        parser.error("门禁模式需要 baseline 和 candidate 参数")

    passed, report = gate(args.baseline, args.candidate,
                          books=args.books, repo_root=repo_root)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False,
                         default=str))
    else:
        print(format_report(report))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
