"""
Output formatting — Markdown tables and Docling-compatible JSON.

Dual output format:
  - Markdown: human-readable, easy to diff and review
  - JSON: machine-readable, compatible with docling export structure

Coordinate system: all bbox values are in PDF points (native PDF units).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_CODE_REV: Optional[str] = None


def code_rev() -> str:
    """
    Fingerprint of the code that produced an output artifact.

    git short hash when available; otherwise the newest mtime among the
    assay package modules.  Embedded in JSON exports so an evaluation
    snapshot can always be traced back to (and checked against) the exact
    code state that generated it.
    """
    global _CODE_REV
    if _CODE_REV is not None:
        return _CODE_REV
    rev = ""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            rev = r.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "assay/"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                rev += "+dirty"
    except Exception:
        rev = ""
    if not rev:
        pkg = Path(__file__).resolve().parent
        rev = f"mt{int(max(f.stat().st_mtime for f in pkg.glob('*.py')))}"
    _CODE_REV = rev
    return rev


@dataclass
class CellOutput:
    """A single cell in a parsed table."""
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    row: int = 0
    col: int = 0
    row_span: int = 1
    col_span: int = 1
    column_header: bool = False
    row_header: bool = False


@dataclass
class TableOutput:
    """A single parsed table."""
    page_no: int
    num_rows: int
    num_cols: int
    cells: List[CellOutput] = field(default_factory=list)
    caption: str = ""
    table_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    """Overall table bbox in PDF points (YOLO + magnetic alignment result)."""
    completed: bool = False
    """True when the cell grid was geometrically rebuilt by
    assay.completion (SLANet under-segmentation rescue), not by SLANet."""

    def cell_grid(self) -> List[List[CellOutput]]:
        """Return cells as a 2D grid (row × col) using row/col attributes."""
        if not self.cells or self.num_rows == 0 or self.num_cols == 0:
            return []
        grid: List[List[CellOutput]] = [
            [CellOutput() for _ in range(self.num_cols)]
            for _ in range(self.num_rows)
        ]
        for cell in self.cells:
            r, c = cell.row, cell.col
            if 0 <= r < self.num_rows and 0 <= c < self.num_cols:
                grid[r][c] = cell
        return grid


@dataclass
class TextOutput:
    """A text segment (paragraph, heading, etc.)."""
    text: str
    page_no: int
    label: str = "text"  # 'text', 'section_heading', 'title'
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    """Block bbox in PDF points; (0,0,0,0) when unknown.  Lets the export
    drop table-internal text and interleave by reading order."""


@dataclass
class PageOutput:
    """Metadata about a parsed page."""
    page_no: int
    width: float = 0.0
    height: float = 0.0


@dataclass
class DocumentOutput:
    """Complete output for one PDF file."""
    file_name: str
    pages: List[PageOutput] = field(default_factory=list)
    tables: List[TableOutput] = field(default_factory=list)
    texts: List[TextOutput] = field(default_factory=list)
    status: str = "SUCCESS"
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Statement caption restoration & continuation marking (C2 报表页眉, C6 续表)
# ──────────────────────────────────────────────────────────────────────────────
#
# 报表页眉（"合并资产负债表(续) 2024年12月31日 人民币元"）在提取阶段的三种
# 失稳形态与对策：
#   丢失（平安 p122-p296）——运行页眉共识的"视觉行伙伴"连带删除把它从文本
#       流剥掉 → restore：按 detection 记录的真实 bbox 合成文本块补回，
#       上下文行（日期/单位）一并恢复；
#   泄漏 + 吸入（紫光 p96-p110）——页眉行幸存但重复出现、且被吸进表格首格
#       → dedupe：顶部带内同名/日期/单位重复块只留最上一个；strip：首行
#       格里的页眉名从单元格剔除；
#   未印（滨海 p85-p169）——续页本就不印页眉 → inherit：沿未断裂跨页链
#       （本页顶表 + 上页贴底末表 + x 重叠）回溯最近报表名会话锚点，生成
#       "报表名（续）"置于首表上方。
# 另对每条识别出的跨页续接首表标注 continued（C6：避免误判新表），无法
# 关联报表名的续表输出 <!-- 续表（接上页） --> 注释。
#
# 数据来源是 detection.calibrate_page_headers 全书扫描时记录的
# statement_caption_info（提取 Phase 1 同进程填充）。识别结果只以新增
# metadata 键（prov.statement_caption / prov.continued / prov.origin）与
# 新增文本块落地，不改任何既有键语义；无校准数据时 plan 为空，导出与
# 现状逐字节一致（fail-open）。

_CAP_TOP_BAND = 0.25   # 页眉识别带（×页高），对齐 detection.CAPTION_TOP_FRAC
_CONT_TOP_SANITY = 0.45  # 续页首表防呆界：网格顶须在页面上半部
_FOOTER_BAND = 0.92    # 页脚带：> 0.92×页高 的末表下文不阻断续接
_FOOTER_MINLEN = 6     # 末表下文 < 6 字（页码等）不阻断续接
_CONT_X_OV = 0.5       # 跨页两表 x 向重叠 ≥ 0.5×较窄者宽度

_DATE_ONLY_RE = re.compile(
    r"^[（(]?\s*\d{2,4}\s*年\s*(\d{1,2}\s*月\s*(\d{1,2}\s*日)?\s*)?[）)]?$"
    r"|^[（(]?\s*\d{4}\s*年度\s*[）)]?$")
_UNIT_ONLY_RE = re.compile(
    r"^[（(]?\s*(单位[：:]\s*)?(人民币|港币|美元|欧元)?\s*"
    r"(元|千元|万元|百万元|亿元)\s*[）)]?$")

try:
    from .detection import statement_caption_info as _stmt_info
except Exception:                    # 脱离 assay 包独立使用时静默空转
    def _stmt_info(_stem: str) -> dict:
        return {}


def _norm_txt(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _cap_match(norm: str, canon: str) -> bool:
    """规范化文本是否就是该报表名（容忍编号前缀与（续）后缀）。"""
    s = re.sub(r"^[0-9]+[、.．·]", "", norm)
    s = re.sub(r"^[、.．·]+", "", s)
    s = re.sub(r"[(（]续[)）]$", "", s)
    return bool(s) and s == canon


# 页眉族判定用（可无锚点局部剥离）：日期行 / 单位行 / 连接符
_DATE_PART_RE = re.compile(
    r"[（(]?\d{4}\s*年度[）)]?"
    r"|[（(]?\d{2,4}\s*年\s*(\d{1,2}\s*月\s*(\d{1,2}\s*日\s*)?)?[）)]?")
_UNIT_PART_RE = re.compile(
    r"[（(]?(除特别注明外[，,、]?)?(金额单位均为|金额单位为|单位[：:]\s*)?"
    r"(人民币|港币|美元|欧元)?"
    r"(元|千元|万元|百万元|亿元)[）)]?")
_CAP_FILLER_RE = re.compile(r"[\s（）()、，,：:;；·]+")
# 报表页固定收尾行（"后附财务报表附注为本财务报表的组成部分。"）——
# 等同页脚，不阻断跨页续接
_FOOTER_TEXT_RE = re.compile(r"^后附.{0,12}(附注|说明).{0,12}组成部分。?$")


def _cap_family(norm: str, names) -> bool:
    """该文本是否属于页眉族：报表名 / 日期 / 单位 或其组合。

    用于续接判定——表上方若只叠了页眉族行，则视为页首表（续表候选）；
    若混有正文或章节标题，剥离后必有残留 → 非页眉族 → 新表。
    """
    if not norm:
        return True
    if any(_cap_match(norm, n) for n in names):
        return True
    if re.search(r"[(（]续[)）]$", norm):
        return True    # 自带（续）后缀的行本身就是续接标记
    s = _DATE_PART_RE.sub("", norm)
    s = _UNIT_PART_RE.sub("", s)
    s = _CAP_FILLER_RE.sub("", s)
    return s == ""


def _grid_extent(table: TableOutput) -> Tuple[float, float, float, float]:
    """单元格并集 bbox（无有效格时退回检测框）。"""
    pts = [c.bbox for c in table.cells
           if c.bbox[2] > c.bbox[0] and c.bbox[3] > c.bbox[1]]
    if not pts:
        return table.table_bbox
    return (min(b[0] for b in pts), min(b[1] for b in pts),
            max(b[2] for b in pts), max(b[3] for b in pts))


def _emit(plan: dict, text: str, pno: int, bbox, origin: str) -> None:
    t = TextOutput(text=text, page_no=pno, label="text", bbox=bbox)
    plan["emit"].append(t)
    plan["emit_origin"][id(t)] = origin


def _caption_plan(doc: DocumentOutput) -> dict:
    """
    Build the caption restoration / continuation plan (pure recognition, no
    mutation — application lives in _apply_caption_plan).  Cached on the doc;
    an empty plan means zero behaviour change.
    """
    cached = getattr(doc, "_cap_plan", None)
    if cached is not None:
        return cached
    plan = {"emit": [], "emit_origin": {}, "drop_ids": set(),
            "strip_cells": [], "continued_ids": set(), "comment_ids": set(),
            "assoc": {}, "demote_pages": set(), "applied": False}
    info = _stmt_info(doc.file_name)
    pages_info = info.get("pages", {}) if info else {}
    sessions = info.get("sessions", {}) if info else {}
    heights = {p.page_no: p.height for p in doc.pages if p.height > 0}
    if (not pages_info and not sessions) or not heights:
        doc._cap_plan = plan
        return plan

    page_texts: Dict[int, List[TextOutput]] = {}
    for t in doc.texts:
        page_texts.setdefault(t.page_no, []).append(t)
    page_tables: Dict[int, List[TableOutput]] = {}
    for t in doc.tables:
        page_tables.setdefault(t.page_no, []).append(t)

    link_cache: Dict[int, object] = {}

    def link(pno: int):
        """
        跨页续接判定 (top_table, prev_table)：
          本页首表 —— 网格顶在页面上半部（防呆界 0.45×页高），且表上方
          所有文本都是页眉族（报表名 / 日期 / 单位及其组合）。续页上
          表方只会叠页眉行；混有正文或章节标题即新表。不以固定带为界：
          页眉块本身高矮不一（紫光 p206 表顶 216pt 已出 0.25 带）；
          上页末表 —— 全页网格底最大的表，其后只允许页脚（贴底带或
          <6 字的页码行）。报表常在页中截断续排（平安资产/负债分页），
          故不以贴底为条件，而以"末表即页尾内容"为条件；
          两表 x 向重叠 ≥ 0.5×较窄者宽度。
        """
        if pno in link_cache:
            return link_cache[pno]
        res = None
        H, Hp = heights.get(pno, 0.0), heights.get(pno - 1, 0.0)
        if H > 0 and Hp > 0:
            names = {e["name"] for e in pages_info.values()}
            names.update(e["name"] for e in sessions.values())
            tops = []
            for t in page_tables.get(pno, []):
                g = _grid_extent(t)
                if g[1] >= _CONT_TOP_SANITY * H:
                    continue
                above = [x for x in page_texts.get(pno, [])
                         if x.bbox[3] <= g[1] + 1.0]
                if all(_cap_family(_norm_txt(x.text), names)
                       for x in above):
                    tops.append(t)
            prevs = page_tables.get(pno - 1, [])
            bots = []
            if prevs:
                t_last = max(prevs, key=lambda t: _grid_extent(t)[3])
                gl = _grid_extent(t_last)
                clear = not any(
                    x.bbox[1] > gl[3] + 2.0
                    and x.bbox[1] < _FOOTER_BAND * Hp
                    and len(_norm_txt(x.text)) >= _FOOTER_MINLEN
                    and not _FOOTER_TEXT_RE.match(_norm_txt(x.text))
                    for x in page_texts.get(pno - 1, []))
                if clear:
                    bots.append(t_last)
            for t1 in tops:
                g1 = _grid_extent(t1)
                for t0 in bots:
                    g0 = _grid_extent(t0)
                    ov = min(g1[2], g0[2]) - max(g1[0], g0[0])
                    if ov >= _CONT_X_OV * min(g1[2] - g1[0],
                                              g0[2] - g0[0]):
                        res = (t1, t0)
                        break
                if res is not None:
                    break
        link_cache[pno] = res
        return res

    notes_pages = {p for p, e in pages_info.items()
                   if e.get("kind") == "notes"}

    def inherit_from(pno: int):
        """沿未断裂跨页链回溯最近的报表名会话锚点；途经附注段即终止。"""
        k = pno
        while k >= 1:
            if link(k) is None:
                return None
            k -= 1
            if k in notes_pages:
                return None            # 附注段开始，报表语境终结
            if k in sessions:
                return sessions[k]
        return None

    for pno in sorted(set(page_tables) | set(pages_info)):
        H = heights.get(pno, 0.0)
        if H <= 0:
            continue
        band = _CAP_TOP_BAND * H
        entry = pages_info.get(pno)
        lk = link(pno)
        texts = page_texts.get(pno, [])
        tables = page_tables.get(pno, [])
        grid_top = min((_grid_extent(t)[1] for t in tables), default=H)

        if entry is not None:
            canon = entry["name"]
            plan["demote_pages"].add(pno)
            # ── dedupe: 顶部带内页眉名/日期/单位的重复泄漏块只留最上一个 ──
            seen_dup = set()
            for t in sorted(texts, key=lambda t: (t.bbox[1], t.bbox[0])):
                if t.bbox[1] >= min(band, grid_top):
                    continue
                n = _norm_txt(t.text)
                if _cap_match(n, canon):
                    dup_key = ("name", n)
                elif _DATE_ONLY_RE.match(n):
                    dup_key = ("date", n)
                elif _UNIT_ONLY_RE.match(n):
                    dup_key = ("unit", n)
                else:
                    continue
                if dup_key in seen_dup:
                    plan["drop_ids"].add(id(t))
                else:
                    seen_dup.add(dup_key)
            # ── restore: 缺失的页眉行（含日期/单位上下文）补回 ──
            pieces = [(entry["text"], entry["bbox"], "name")]
            pieces += [(c["text"], c["bbox"], "ctx")
                       for c in entry.get("context", [])]
            missing = []
            for txt, bb, role in pieces:
                n = _norm_txt(txt)
                present = any(
                    t.bbox[1] < band and id(t) not in plan["drop_ids"]
                    and (_cap_match(_norm_txt(t.text), canon)
                         if role == "name" else _norm_txt(t.text) == n)
                    for t in texts)
                if not present:
                    missing.append((txt, bb))
            if missing:
                if all(bb[3] <= grid_top - 1.0 for _, bb in missing):
                    for txt, bb in missing:            # 真实位置可用（平安式）
                        _emit(plan, txt, pno, bb, "caption_restore")
                else:      # 页眉行被吸进网格（紫光式）→ 按原序堆到网格顶上方
                    cursor = grid_top - 1.0
                    for txt, bb in reversed(missing):
                        h = max(bb[3] - bb[1], 6.0)
                        y1, y0 = cursor, cursor - h
                        if y0 < 0:
                            break
                        _emit(plan, txt, pno, (bb[0], y0, bb[2], y1),
                              "caption_restore")
                        cursor = y0 - 0.2 * h
            # ── strip: 吸进首行格的页眉名从单元格剔除（紫光 r0c0）──
            for t in tables:
                if _grid_extent(t)[1] >= band:
                    continue
                for c in t.cells:
                    if c.row == 0 and c.text.strip() \
                            and _cap_match(_norm_txt(c.text), canon):
                        plan["strip_cells"].append(c)
            # ── 页内表格关联报表名（prov 新键）──
            for t in tables:
                plan["assoc"][id(t)] = canon
            # 印了（续）（statement 的 printed_continued，或 notes 名自带
            # （续）后缀）且跨页续接成立 → 首表标 continued（C6 显式续标识）
            cont_marked = entry.get("printed_continued") or \
                bool(re.search(r"[(（]续[)）]$", canon))
            if cont_marked and lk is not None:
                plan["continued_ids"].add(id(lk[0]))
        elif lk is not None:
            # ── 无印刷页眉的续页：C6 续表识别 + 滨海式继承 ──
            t1 = lk[0]
            plan["continued_ids"].add(id(t1))
            src = inherit_from(pno)
            if src is not None:
                canon = src["name"]
                g1 = _grid_extent(t1)
                hs = sorted(c.bbox[3] - c.bbox[1] for c in t1.cells
                            if 1.0 < c.bbox[3] - c.bbox[1] < 200)
                rh = hs[len(hs) // 2] if hs else 12.0
                y1 = g1[1] - 0.2 * rh
                y0 = y1 - 1.2 * rh
                if y0 >= 0:
                    _emit(plan, f"{canon}（续）", pno,
                          (g1[0], y0, g1[2], y1), "caption_inherit")
                    plan["assoc"][id(t1)] = canon
                else:
                    plan["comment_ids"].add(id(t1))
            else:
                plan["comment_ids"].add(id(t1))

    doc._cap_plan = plan
    return plan


def _apply_caption_plan(doc: DocumentOutput) -> dict:
    """Idempotently apply the plan: strip absorbed cells + add synthetic texts."""
    plan = _caption_plan(doc)
    if plan["applied"]:
        return plan
    for c in plan["strip_cells"]:
        c.text = ""
    if plan["emit"]:
        doc.texts.extend(plan["emit"])
    plan["applied"] = True
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# Markdown export
# ──────────────────────────────────────────────────────────────────────────────

def export_to_markdown(doc: DocumentOutput) -> str:
    """Export document as Markdown string with text and tables interleaved by page."""
    plan = _apply_caption_plan(doc)          # C2/C6: restore/dedupe/strip/mark
    page_heights = {p.page_no: p.height for p in doc.pages if p.height > 0}
    lines: List[str] = []

    # Group content by page — texts first, then tables
    page_texts: Dict[int, List[TextOutput]] = {}
    for t in doc.texts:
        page_texts.setdefault(t.page_no, []).append(t)

    page_tables: Dict[int, List[TableOutput]] = {}
    for t in doc.tables:
        page_tables.setdefault(t.page_no, []).append(t)

    # Use all page numbers from both texts and tables
    all_pages = sorted(set(list(page_texts.keys()) + list(page_tables.keys())))

    for page_no in all_pages:
        if len(lines) == 0:
            lines.append(f"<!-- Page {page_no + 1} -->\n")
        else:
            lines.append(f"\n<!-- Page {page_no + 1} -->\n")

        tables = page_tables.get(page_no, [])
        # Reading order: interleave texts and tables by their top y.
        # Table position = top of its cell grid (the grid, not the
        # detection box, is where the rendered table actually starts, so
        # a scope label absorbed above the grid keeps sorting before it).
        items = []
        for t in page_texts.get(page_no, []):
            if id(t) in plan["drop_ids"]:
                continue        # C2: 顶部带内重复的页眉泄漏块
            if _is_table_text(t, tables):
                continue        # table cell text — rendered in the table
            items.append((t.bbox[1] if t.bbox[3] > t.bbox[1] else 1e9,
                          len(items), ("text", t)))
        for tb in tables:
            y = _table_top(tb)
            items.append((y, len(items), ("table", tb)))
        items.sort(key=lambda it: (it[0], it[1]))

        for _, _, (kind, obj) in items:
            if kind == "text":
                label = obj.label
                # C2: 页眉块的日期/单位行被误判成 heading 时降级回正文
                # （仅页眉活动页 + 顶部带，避免误伤正文章节标题）
                if label in ("section_heading", "subsection_heading", "title") \
                        and obj.page_no in plan["demote_pages"] \
                        and obj.bbox[1] < _CAP_TOP_BAND * page_heights.get(
                            obj.page_no, 0.0) \
                        and (_DATE_ONLY_RE.match(_norm_txt(obj.text))
                             or _UNIT_ONLY_RE.match(_norm_txt(obj.text))):
                    label = "text"
                if label == "section_heading":
                    lines.append(f"\n## {obj.text}\n")
                elif label == "subsection_heading":
                    lines.append(f"\n### {obj.text}\n")
                elif label == "title":
                    lines.append(f"\n# {obj.text}\n")
                elif obj.text.strip():
                    lines.append(f"{obj.text}\n")
            else:
                if id(obj) in plan["comment_ids"]:
                    # C6: 识别为跨页续表但无法关联报表名 → 显式标注，避免
                    # 被下游误判为新表
                    lines.append("<!-- 续表（接上页） -->")
                lines.append(_table_to_md(obj))
                lines.append("")

    return "\n".join(lines)


def _table_top(table: TableOutput) -> float:
    """Top y of a table's rendered grid (falls back to the detection box)."""
    ys = [c.bbox[1] for c in table.cells if c.bbox[3] > c.bbox[1]]
    return min(ys) if ys else table.table_bbox[1]


def _is_table_text(text: TextOutput, tables: List[TableOutput]) -> bool:
    """
    True when a text block is table cell content.  Four tests:

    per-cell coverage (any table): cells of one table cover >=50% of the
    block.  Coverage accumulates over the table's cells because PyMuPDF
    often packs a whole table into ONE text block — per-cell tests alone
    would see only a small fraction each.

    grid-extent containment (rebuilt tables only): a completed grid tiles
    its whole region, so ANY block inside the grid extent is table
    content.  PyMuPDF groups section-label rows（"归属于："）into
    full-row-width blocks whose overlap with the single label cell is
    only ~25% — per-cell coverage lets them leak and they reappear after
    the table in reading order.  For rebuilt grids the leak is
    impossible: nothing inside the grid is anything but table content.

    detection-box containment (any table): a block almost fully inside
    the detection bbox is table content even when SLANet cells missed it
    —— SLANet 漏格不会让内容搬出表格区域，留在区域内的只会是重复泄漏。

    duplicate content (any table): 表后成段重复的行标签/整列数值
    （"预付款项/减：减值准备"、"H 股 21,099 …"）——块文本完整出现在
    该表单元格文本中、且紧贴表底（泄漏总在阅读顺序的表后）→ 过滤。
    """
    bx0, by0, bx1, by1 = text.bbox
    if bx1 <= bx0 or by1 <= by0:
        return False
    barea = (bx1 - bx0) * (by1 - by0)
    bn = re.sub(r"\s+", "", text.text)
    for table in tables:
        # 表格自身的布局尺度 = 中位格高（随排版密度缩放，不依赖字号）
        th = getattr(table, "_row_h", None)
        if th is None:
            hs = sorted(c.bbox[3] - c.bbox[1] for c in table.cells
                        if 1.0 < c.bbox[3] - c.bbox[1] < 200)
            th = hs[len(hs) // 2] if hs else 12.0
            table._row_h = th
        inter = 0.0
        gx0 = gy0 = gx1 = gy1 = None
        for c in table.cells:
            r = c.bbox
            if r[2] <= r[0] or r[3] <= r[1]:
                continue
            if gx0 is None:
                gx0, gy0, gx1, gy1 = r
            else:
                gx0, gy0 = min(gx0, r[0]), min(gy0, r[1])
                gx1, gy1 = max(gx1, r[2]), max(gy1, r[3])
            ix0, iy0 = max(bx0, r[0]), max(by0, r[1])
            ix1, iy1 = min(bx1, r[2]), min(by1, r[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter += (ix1 - ix0) * (iy1 - iy0)
        if inter >= 0.5 * barea:
            return True
        if table.completed and gx0 is not None:
            ix0, iy0 = max(bx0, gx0), max(by0, gy0)
            ix1, iy1 = min(bx1, gx1), min(by1, gy1)
            if ix1 > ix0 and iy1 > iy0 \
                    and (ix1 - ix0) * (iy1 - iy0) >= 0.5 * barea:
                return True
        # detection-box containment 收窄为 cell-union containment：
        # 检测框常比网格多包住表标题行（平安 "现金分红政策的专项说明"
        # 在框内、首行带外）——按框过滤会把这行吞掉，按格并集过滤则
        # 放行给正文。网格上沿贴上方的短行（表标题）同样放行：它压着
        # 首行格却不是表内容
        if gx0 is not None and gx1 > gx0 and gy1 > gy0:
            # 网格头两行带之间的孤立短行（表标题压着折行表头区）：
            # 高度 < 0.7×中位格高且不进入第三行带 → 放行为正文
            ys_sorted = sorted(c.bbox[1] for c in table.cells
                               if c.bbox[3] > c.bbox[1])
            if len(ys_sorted) >= 3:
                band2 = ys_sorted[len(ys_sorted) // 3]
                if by1 - by0 < 0.7 * th and (by0 + by1) / 2 < band2:
                    continue
            ix0, iy0 = max(bx0, gx0), max(by0, gy0)
            ix1, iy1 = min(bx1, gx1), min(by1, gy1)
            if ix1 > ix0 and iy1 > iy0 \
                    and (ix1 - ix0) * (iy1 - iy0) >= 0.9 * barea:
                return True
            if bn and len(bn) >= 2 and by0 >= gy1 - 0.3 * th \
                    and by0 <= gy1 + 3.0 * th:
                cached = getattr(table, "_norm_cells_cache", None)
                if cached is None:
                    cached = re.sub(r"\s+", "",
                                    "".join(c.text for c in table.cells))
                    table._norm_cells_cache = cached
                if bn in cached:
                    return True
    return False


def _table_to_md(table: TableOutput) -> str:
    """Convert a parsed table to Markdown table format."""
    grid = table.cell_grid()
    if not grid or not grid[0]:
        return ""

    num_cols = len(grid[0])
    num_rows = len(grid)
    md_lines: List[str] = []

    for ri, row in enumerate(grid):
        md_row: List[str] = []
        for cell in row:
            text = cell.text.replace("\n", " ").replace("|", "\\|")
            md_row.append(text)
        # Pad to full width
        while len(md_row) < num_cols:
            md_row.append("")
        md_lines.append("| " + " | ".join(md_row) + " |")

        # Add header separator after first row if:
        #   - The first row has explicit column_header flags, OR
        #   - The table has at least 2 rows (heuristic: first row = header)
        if ri == 0 and (any(c.column_header for c in row) or num_rows >= 2):
            sep = "| " + " | ".join(["---"] * num_cols) + " |"
            md_lines.append(sep)

    return "\n".join(md_lines)


# ──────────────────────────────────────────────────────────────────────────────
# JSON export (Docling-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def export_to_dict(doc: DocumentOutput) -> dict:
    """Export to Docling-compatible dict structure."""
    plan = _apply_caption_plan(doc)          # C2/C6: same plan as the md export
    return {
        "schema_name": "docling",
        "version": "0.1.0",
        "code_rev": code_rev(),
        "name": doc.file_name,
        "pages": {
            str(p.page_no): {
                "page_no": p.page_no,
                "size": {"width": p.width, "height": p.height},
            }
            for p in doc.pages
        },
        "texts": [
            {
                "text": t.text,
                "label": t.label,
                "prov": [{
                    "page_no": t.page_no,
                    "bbox": ([round(v, 1) for v in t.bbox]
                             if t.bbox[3] > t.bbox[1] else None),
                    # C2 新键：合成补回的页眉行标出来源，便于审计追溯
                    **({"origin": plan["emit_origin"][id(t)]}
                       if id(t) in plan["emit_origin"] else {}),
                }],
            }
            for t in doc.texts
            if t.text.strip() and id(t) not in plan["drop_ids"]
        ],
        "tables": [
            {
                "label": "table",
                "data": {
                    "num_rows": t.num_rows,
                    "num_cols": t.num_cols,
                    "grid": _build_grid_json(t),
                },
                "prov": [{
                    "page_no": t.page_no,
                    # Detection-level bbox (PDF points): the trusted region
                    # evidence — cell bboxes may cover less (SLANet under-
                    # segmentation); evaluators need this to score both.
                    "bbox": [round(v, 2) for v in t.table_bbox],
                    # True when the grid was geometrically rebuilt by
                    # assay.completion rather than read by SLANet.
                    "completed": t.completed,
                    # C2/C6 新键（不改既有键语义）：所属报表名 / 跨页续表标记
                    **({"statement_caption": plan["assoc"][id(t)]}
                       if id(t) in plan["assoc"] else {}),
                    **({"continued": True}
                       if id(t) in plan["continued_ids"] else {}),
                }],
            }
            for t in doc.tables
        ],
    }


def _build_grid_json(table: TableOutput) -> List[List[dict]]:
    """Build the grid JSON (list of rows, each row is list of cells)."""
    grid = table.cell_grid()
    result: List[List[dict]] = []
    for row in grid:
        json_row: List[dict] = []
        for cell in row:
            json_row.append({
                "bbox": {
                    "l": cell.bbox[0],
                    "t": cell.bbox[1],
                    "r": cell.bbox[2],
                    "b": cell.bbox[3],
                },
                "text": cell.text,
                "row": cell.row,
                "col": cell.col,
                "row_span": cell.row_span,
                "col_span": cell.col_span,
                "column_header": cell.column_header,
                "row_header": cell.row_header,
            })
        result.append(json_row)
    return result


def write_markdown(doc: DocumentOutput, path: str) -> None:
    """Write Markdown output to file."""
    content = export_to_markdown(doc)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_json(doc: DocumentOutput, path: str, indent: int = 0) -> None:
    """Write JSON output to file."""
    data = export_to_dict(doc)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────────
# Debug PDF: overlay table bboxes for manual verification
# ──────────────────────────────────────────────────────────────────────────────

def write_debug_pdf(
    doc: DocumentOutput,
    pdf_path: str,
    output_path: str,
) -> str:
    """
    Overlay detected table bboxes onto the original PDF for manual verification.

    Draws each table's magnet-aligned bbox as a green rectangle with label.
    Only page(s) that actually contain table detections are modified.

    Args:
        doc:         DocumentOutput with tables (each must have table_bbox set).
        pdf_path:    Path to the original PDF file.
        output_path: Path for the output debug PDF (e.g. ``{name}_layout.pdf``).

    Returns:
        The output file path on success.
    """
    import fitz

    # Group tables by page
    page_tables: Dict[int, List[TableOutput]] = {}
    for table in doc.tables:
        page_tables.setdefault(table.page_no, []).append(table)

    if not page_tables:
        _log_debug = __import__('logging').getLogger(__name__)
        _log_debug.warning("No tables to annotate — skipping debug PDF")
        return output_path

    src = fitz.open(pdf_path)
    try:
        for page_no, tables in page_tables.items():
            if page_no < 0 or page_no >= len(src):
                continue
            page = src[page_no]

            for ti, table in enumerate(tables):
                bbox = table.table_bbox
                if bbox == (0, 0, 0, 0):
                    continue  # no valid bbox stored

                rect = fitz.Rect(*bbox)

                # Green rectangle (stroke only)
                page.draw_rect(rect, color=(0, 0.69, 0.28), width=1.5)

                # Small label at top-left corner
                label = f"T{ti+1} | {table.num_rows}r×{table.num_cols}c"
                fontsize = 7
                # Place label just above the table bbox
                label_rect = fitz.Rect(
                    rect.x0,
                    max(0, rect.y0 - fontsize - 4),
                    rect.x0 + len(label) * 3.5,
                    max(0, rect.y0 - 2),
                )
                page.draw_rect(label_rect, color=(0, 0.69, 0.28),
                              fill=(0, 0.69, 0.28), width=0)
                page.insert_text(
                    label_rect.bl + (2, 0),
                    label,
                    fontname="Helv",
                    fontsize=fontsize,
                    color=(1, 1, 1),
                )

        # garbage=1 (basic compaction) NOT 3: the cross-object dedup pass of
        # garbage>=2 is pathologically slow on image-heavy PDFs (observed 167s
        # on a 4MB report vs 0.1s here) for no benefit on this throwaway debug
        # artifact.  deflate stays on so content streams are still compressed.
        src.save(output_path, garbage=1, deflate=True)
    finally:
        src.close()

    return output_path
