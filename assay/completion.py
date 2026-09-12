"""
Content-coverage completion for SLANet under-segmented tables.

SLANet under-segments certain table kinds — borderless all-text registers
(one giant 1x1 cell over a whole related-party list), framed prose
disclosures (关键审计事项 read as a stub grid over a page-tall frame) —
and the cell-clip text extraction then captures only a fraction of the
table's content.  The detection bbox is right (it is validated against
ground truth); only the internal grid is wrong.

This module is a pure CONTENT layer: it runs after all detection vetoes,
on tables that already survived on their own merits, and can neither add
nor remove tables.  Three gates, all evidence-driven:

  trigger (fail-closed)
      Text lines inside the trusted table bbox that no cell contains the
      centre of.  Only when such uncovered lines carry a material share
      of the region's characters does the rebuild fire — a well-formed
      SLANet grid tiles its table, so it can never trigger.

  rebuild (geometry only, no ML, no case-specific rules)
      Rows: assay.splitting._rows text-line bands, tiled to the full
      region (band-midpoint boundaries) so the grid covers the table the
      way a drawn grid would.
      Cols: the same gutter seed/confirm vote as looks_text_table,
      generalised from one dominant channel to every well-confirmed
      channel; _justified_prose vetoes the two-column-prose case.

  adoption (two self-checks — the whack-a-mole fuses)
      text: bag-of-chars recall against the region's native text must not
      regress.  structure: a rebuild that collapses columns is adopted
      only when the grid it replaces cuts through more text spans than it
      does — SLANet's "columns" on a prose frame slice full-width lines
      (bogus, lose), while its columns on a real grid land in gutters
      (genuine, stay).

Invariants:
  - never fires on a grid whose cells cover the region's text;
  - never emits a cell outside the trusted detection bbox;
  - never adopts a rebuild with lower text recall than the original;
  - never replaces a genuine multi-column grid with a poorer one.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

import fitz

from assay.output import CellOutput
from assay.rulings import extract_line_segments
from assay.splitting import (
    TEXT_TABLE_GUTTER,
    _colrow_ok,
    _justified_prose,
    _rows,
)

_log = logging.getLogger(__name__)

Rect = Tuple[float, float, float, float]

# ── Trigger ───────────────────────────────────────────────────────────────

MIN_UNCOVERED_FRAC = 0.10   # uncovered chars as a share of region chars
MIN_UNCOVERED_LINES = 2     # a single stray caption/adjacent line never fires

# ── Column gutters ────────────────────────────────────────────────────────

COL_CONFIRM_MIN = 3         # a column channel needs this many confirming rows
COL_CONFIRM_RATIO = 0.5     # ... and this share of the multi-span rows
                            # (looser than looks_text_table's 4/0.7: the region
                            # is already a confirmed table, only the column
                            # count is in question)


# ── Text recall (bag-of-chars, whitespace-normalised) ─────────────────────

_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub("", s or "")


def _recall(ref: Counter, cells: List[CellOutput]) -> float:
    """Share of reference characters present in the cells' text."""
    if not ref:
        return 1.0
    hyp = Counter(_norm("".join(c.text for c in cells)))
    return sum((ref & hyp).values()) / sum(ref.values())


# ── Trigger ───────────────────────────────────────────────────────────────


def _uncovered_text(page: fitz.Page, box: Rect,
                    cells: List[CellOutput]) -> Tuple[int, int, int]:
    """
    (total chars, uncovered chars, uncovered lines) inside ``box``.

    A line is uncovered when no valid cell bbox contains its centre — the
    robust under-segmentation signal, since a proper grid tiles its table.
    """
    rects = [c.bbox for c in cells
             if c.bbox[2] > c.bbox[0] and c.bbox[3] > c.bbox[1]]
    x0, y0, x1, y1 = box
    total = uncovered = n_unc_lines = 0
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
            if not (x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1):
                continue
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if not txt:
                continue
            total += len(txt)
            if not any(r[0] - 1 <= cx <= r[2] + 1
                       and r[1] - 1 <= cy <= r[3] + 1 for r in rects):
                uncovered += len(txt)
                n_unc_lines += 1
    return total, uncovered, n_unc_lines


def needs_completion(page: fitz.Page, table_bbox: Optional[Rect],
                     cells: List[CellOutput]) -> bool:
    """True when SLANet's grid provably misses region text."""
    if table_bbox is None or not cells:
        return False
    total, uncovered, n_lines = _uncovered_text(page, table_bbox, cells)
    return (n_lines >= MIN_UNCOVERED_LINES
            and total > 0
            and uncovered / total >= MIN_UNCOVERED_FRAC)


# ── Structure misalignment ────────────────────────────────────────────────

STRADDLE_MIN = 2       # fewer suspect cells than this is wrap noise
STRADDLE_RATIO = 0.15  # ... or under this share of the grid's cells
CUT_RATIO = 0.20       # share of cell edges allowed to slice a text row


def structure_suspect(page: fitz.Page, table_bbox: Optional[Rect],
                      cells: List[CellOutput]) -> bool:
    """
    True when SLANet's row edges have drifted off the text rows.

    Two drift morphologies, both centered on MULTI-SPAN text rows
    (prose lines are single-span and never count):

      straddling cells — a cell spans the centres of >=2 text rows
      （分部表 after a wrapped header: two rows' labels fused in one
      cell, two rows' values in the next);

      edge-cutting — a half-row offset keeps every cell holding one
      row centre while its edges slice THROUGH neighbouring text rows
      （中国石油 IFRS 摘要表： "所得税费用 净利润" fused, each cell
      containing exactly one centre).  A healthy grid puts its edges in
      the whitespace between rows, so the cut ratio is ~0.
    """
    if table_bbox is None or not cells:
        return False
    bands = [(y0, y1) for y0, y1, spans in _rows(page, table_bbox)
             if len(spans) >= 2 and y1 - y0 > 3.0]
    valid = [c for c in cells
             if c.bbox[2] > c.bbox[0] and c.bbox[3] > c.bbox[1]]
    if len(bands) < 3 or not valid:
        return False
    centres = [(y0 + y1) / 2 for y0, y1 in bands]
    straddle = sum(
        1 for c in valid
        if sum(1 for cy in centres if c.bbox[1] < cy < c.bbox[3]) >= 2)
    if straddle >= max(STRADDLE_MIN, STRADDLE_RATIO * len(valid)):
        return True
    cut = 0
    for c in valid:
        for e in (c.bbox[1], c.bbox[3]):
            if any(y0 + 1.0 < e < y1 - 1.0 for y0, y1 in bands):
                cut += 1
                break
    if cut >= max(6, CUT_RATIO * len(valid)):
        return True
    return _column_cut(page, table_bbox, valid) >= 1


def _column_cut(page: fitz.Page, table_bbox: Rect,
                cells: List[CellOutput]) -> int:
    """
    列向错位签名：网格内部列边界切进文字**片段**（片段 = 空白分隔的
    文字原子，FRAG_GAP 分割）。真列界两侧有 ≥4pt 的单元格留白，绝不
    会切进片段；切进片段的边界只可能是——

      · 混排字体的行内空格被当成列界（日期表头 "2024 年12 月31 日"
        被劈成 "2024" + "年12 月31 日"，并繁殖整列空列）；
      · SLANet 把通栏之外的窄间隙当成了界。

    通栏表头（≥1.5 倍平均列宽的片段）是合法跨越，不计。任一切割即
    触发重建。
    """
    if table_bbox is None or not cells:
        return 0
    xs = [v for c in cells if c.bbox[2] > c.bbox[0]
          for v in (c.bbox[0], c.bbox[2])]
    if len(xs) < 4:
        return 0
    edges = sorted(set(xs))
    interior = edges[1:-1]
    if not interior:
        return 0
    gx0, gx1 = edges[0], edges[-1]
    colw = (gx1 - gx0) / max(1, len(edges) - 1)
    frags = _line_fragments(page, table_bbox)
    n = 0
    for f in frags:
        if f["x1"] - f["x0"] >= SPAN_CELL_WIDTH * colw:
            continue
        if any(f["x0"] + 0.5 < e < f["x1"] - 0.5 for e in interior):
            n += 1
    return n


# ── Geometric rebuild ─────────────────────────────────────────────────────


def _column_edges(rows: list, x0: float, x1: float) -> List[float]:
    """
    Column x-boundaries from gutter evidence: every inter-span gap of a
    table-consistent row seeds a channel; channels confirmed by enough
    rows survive; overlapping survivors merge; their midpoints become
    column edges.  With no surviving channel the table is one column.
    """
    multi = [r[2] for r in rows if len(r[2]) >= 2 and _colrow_ok(r[2])]
    if len(multi) < COL_CONFIRM_MIN:
        return [x0, x1]
    seeds = [(a[1], b[0]) for sp in multi for a, b in zip(sp, sp[1:])
             if b[0] - a[1] >= TEXT_TABLE_GUTTER]
    chans = []
    for lo, hi in seeds:
        conf = sum(
            1 for sp in multi
            if any(b[0] - a[1] >= TEXT_TABLE_GUTTER
                   and min(hi, b[0]) - max(lo, a[1]) >= 0.75 * TEXT_TABLE_GUTTER
                   for a, b in zip(sp, sp[1:]))
        )
        chans.append((conf, lo, hi))
    need = max(COL_CONFIRM_MIN, COL_CONFIRM_RATIO * len(multi))
    chans = sorted((lo, hi) for conf, lo, hi in chans if conf >= need)
    merged: List[list] = []
    for lo, hi in chans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    # Two-column justified prose shows exactly one gutter with flush wide
    # columns — a text layout, not a column grid (关键审计事项 frames).
    if len(merged) == 1 and _justified_prose(multi, tuple(merged[0])):
        merged = []
    edges = [x0] + [(lo + hi) / 2 for lo, hi in merged] + [x1]
    return edges


def _split_violations(rows: list, edges: List[float]) -> int:
    """
    Text spans cut through by an interior column edge.  A genuine column
    boundary falls in a whitespace gutter; a bogus one slices text lines
    (SLANet reading a prose frame as columns does exactly that).
    """
    n = 0
    for _y0, _y1, spans in rows:
        for sx0, sx1, _t in spans:
            for e in edges[1:-1]:
                if sx0 + 2.0 < e < sx1 - 2.0:
                    n += 1
    return n


def _grid_col_edges(cells: List[CellOutput]) -> List[float]:
    """Column boundary x-values of an existing grid (cell x-edges, merged
    at 3 pt tolerance), sorted."""
    xs = sorted({v for c in cells if c.bbox[2] > c.bbox[0]
                 for v in (c.bbox[0], c.bbox[2])})
    merged: List[float] = []
    for x in xs:
        if merged and x - merged[-1] <= 3.0:
            continue
        merged.append(x)
    return merged


def _ruled_row_edges(page: fitz.Page, box: Rect) -> List[float]:
    """
    Horizontal row rules inside ``box`` — a framed table's row
    boundaries, exact regardless of wrapped cells.

    Row rules are often drawn in SEGMENTS per column block (4 segments
    per rule on the 中国石油 分部表), so lines are grouped by y and the
    union coverage of each y-band decides — no single segment needs to
    span the width.
    """
    H, _V = extract_line_segments(page)
    width = box[2] - box[0]
    by_y: dict = {}
    for h0, h1, hy in H:
        if box[1] - 2.0 <= hy <= box[3] + 2.0:
            by_y.setdefault(round(hy / 2.0), []).append(
                (max(h0, box[0]), min(h1, box[2])))
    rules: List[float] = []
    for key in sorted(by_y):
        segs = sorted(by_y[key])
        cov, cur_end = 0.0, None
        for a, b in segs:
            if cur_end is None or a > cur_end:
                cov += b - a
                cur_end = b
            elif b > cur_end:
                cov += b - cur_end
                cur_end = b
        if cov >= 0.6 * width:
            rules.append(key * 2.0)
    merged: List[float] = []
    for y in rules:
        if not merged or y - merged[-1] > 2.0:
            merged.append(y)
    if len(merged) < 3 or merged[0] > box[1] + 8.0 or merged[-1] < box[3] - 8.0:
        return []          # rules do not structure the whole region
    return merged


def _consolidate_col_edges(edges: List[float], src_cols: int) -> List[float]:
    """
    Consolidate a drifted grid's scattered column edges back to
    ``src_cols + 1`` boundaries.

    When row edges drift, SLANet's cell x-edges scatter around each true
    boundary (中国石油 分部表: 28 fragments over 8 boundaries).  The
    inter-column gaps are then the widest whitespace runs in the edge
    set, so repeatedly merging the closest pair until only
    ``src_cols + 1`` edges remain collapses each fragment cluster onto
    its boundary mean.  Only runs when fragments exceed the grid's own
    column count — a healthy grid's edges pass through untouched.
    """
    target = src_cols + 1
    es = sorted(edges)
    if len(es) <= target:
        return es
    while len(es) > target:
        gaps = [b - a for a, b in zip(es, es[1:])]
        j = gaps.index(min(gaps))
        es[j:j + 2] = [(es[j] + es[j + 1]) / 2]
    return es


def _span_crossing_rate(bands: list, edges: List[float]) -> float:
    """Share of text spans sliced by an interior candidate column edge —
    a sound column boundary falls in the whitespace between spans, so
    the lower the crossing rate, the better the edge set explains the
    text layout."""
    interior = edges[1:-1]
    if not interior:
        return 0.0
    total = bad = 0
    for _y0, _y1, spans in bands:
        for x0, x1, _t in spans:
            total += 1
            if any(x0 + 1.0 < e < x1 - 1.0 for e in interior):
                bad += 1
    return bad / total if total else 0.0


MIDPOINT_TOL = 10.0     # clustering tolerance for span-gap midpoints (pt)
MIDPOINT_ROWS = 0.35    # a boundary needs this share of multi-span rows
MIDPOINT_MIN_GAP = 4.0  # 混排字体行内空格（2-3pt）不是列界（pt）


def _span_midpoint_edges(rows: list, box: Rect) -> List[float]:
    """
    Column boundaries voted by the rows themselves: every multi-span row
    contributes the midpoints between its adjacent spans, and boundaries
    are the clusters confirmed by enough rows.  Numeric tables
    right-align their values, so span LEFT edges wander while the gaps
    between columns stay put — the gap midpoint is the stable feature
    （中国石油 IFRS 摘要表： boundaries vote at ~348/395/441/490 across
    26 rows where SLANet's drifted edges slice the values）.

    投票纪律：候选间隙必须 ≥ MIDPOINT_MIN_GAP——列内混排字体的空格
    （"2024 年12 月31 日" 的数字/汉字间距 2-3pt）绝不是列界，小表
    （n_multi 少）里单行空格投票曾把日期表头劈成 "2024"+"年12 月31 日"
    两格并繁殖整列空列。确认票数下限 2——单行投票不足以立界。
    """
    votes: List[float] = []
    n_multi = 0
    for _y0, _y1, spans in rows:
        if len(spans) < 2:
            continue
        n_multi += 1
        for a, b in zip(spans, spans[1:]):
            if b[0] - a[1] >= MIDPOINT_MIN_GAP:
                votes.append((a[1] + b[0]) / 2)
    if not votes or n_multi < 3:
        return []
    votes.sort()
    need = max(2, MIDPOINT_ROWS * n_multi)
    clusters: List[List[float]] = []
    for v in votes:
        if clusters and v - clusters[-1][-1] <= MIDPOINT_TOL:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    out = [sum(c) / len(c) for c in clusters if len(c) >= need]
    return [box[0]] + out + [box[2]] if out else []


def _line_fragments(page: fitz.Page, box: Rect) -> List[dict]:
    """
    Text lines inside ``box`` split into VALUE FRAGMENTS.

    A fragment is a run of glyphs separated by whitespace — the atom of
    table content.  Columns separate VALUES, and a value is never split
    by whitespace, so fragment centres are the reliable assignment key:
    a right-aligned "(1,586,319)" whose opening paren straddles the
    column edge keeps its whole fragment together instead of bleeding
    "(1,58" into the neighbour (clip and per-char assignment both fail
    here — the paren glyph's centre sits 1.5 pt on the wrong side).
    """
    import bisect
    # clip 四周外扩 4pt：检测框贴着边线，最后一列/最后一行的字形
    # （"…人民币" 的 "币"）可能压线——clip 会把压线字形整体丢弃，
    # 造成格尾截断。外扩只影响字形采集，行列归属仍由格心决定
    clip = fitz.Rect(box[0] - 4.0, box[1] - 4.0, box[2] + 4.0, box[3] + 4.0)
    raw = page.get_text("rawdict", clip=clip)
    fragments: List[dict] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            ly0, ly1 = line["bbox"][1], line["bbox"][3]
            runs: List[List[dict]] = []
            for span in line.get("spans", []):
                skey = id(span)
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if not c or c.isspace():
                        continue
                    bx = ch.get("bbox", (0, 0, 0, 0))
                    if runs and bx[0] - runs[-1][-1][2] <= FRAG_GAP:
                        runs[-1].append((bx[0], bx[1], bx[2], bx[3], c, skey))
                    else:
                        runs.append([(bx[0], bx[1], bx[2], bx[3], c, skey)])
            for run in runs:
                if not run:
                    continue
                fx0 = min(c[0] for c in run)
                fx1 = max(c[2] for c in run)
                # 片段文本保留词间空隙（>1.2pt）："Mangistau Investment
                # B.V." 不能连写成 "MangistauInvestmentB.V."
                text = ""
                prev_x1 = None
                for c in run:
                    if prev_x1 is not None and c[0] - prev_x1 > 1.2:
                        text += " "
                    text += c[4]
                    prev_x1 = c[2]
                skeys = {c[5] for c in run}
                fragments.append({
                    "y0": ly0, "y1": ly1, "x0": fx0, "x1": fx1,
                    "cx": (fx0 + fx1) / 2.0, "cy": (ly0 + ly1) / 2.0,
                    "text": text, "skeys": skeys,
                    # character-level geometry: lets the caller split a
                    # cross-column fragment at its internal gaps when a
                    # column edge falls there （窄列里 "2,759,237" 与
                    # "2,752,710" 间隙 <3pt 粘成一个片段的情形）
                    "chars": [( (c[0]+c[2])/2.0, c[0], c[2], c[4])
                              for c in run],
                })
    fragments.sort(key=lambda f: (f["y0"], f["x0"]))
    return fragments


FRAG_GAP = 3.0         # inter-glyph gap that starts a new fragment (pt)
GAPVOTE_MIN_GAP = 4.0  # 补边投票的最小片段间隙（pt）——行内空格不是列界
SPAN_CELL_WIDTH = 1.5  # a fragment this many columns wide is a spanning
                       # header cell, not column content


def _cells_from_fragments(page: fitz.Page, box: Rect,
                          row_edges: List[float], col_edges: List[float],
                          keep_rows: Optional[List[int]] = None
                          ) -> Tuple[List[CellOutput], List[float]]:
    """
    Build cells by assigning VALUE FRAGMENTS to the row/col interval
    containing each fragment centre.

    Header mode: a line carrying 1-2 fragments whose width blankets
    several columns is a spanning disclosure header
    （"截至或截止各年度12 月31 日" 通栏） — emitted as one cell with
    col_span instead of being scattered per column.

    Narrow-column split: a fragment may blanket several columns when the
    inter-value gap is under the fragment threshold
    （资产负债摘要表 "2,759,2372,752,710"）.  Such a fragment is split
    per character centre at each column edge it crosses — but only where
    a real inter-glyph gap (>= 1.2 pt) sits at the edge, so unbroken
    text never gets cut mid-glyph.
    """
    import bisect
    n_r, n_c = len(row_edges) - 1, len(col_edges) - 1
    colw = (col_edges[-1] - col_edges[0]) / max(1, n_c)
    keep = set(keep_rows) if keep_rows is not None else set(range(n_r))
    frags = _line_fragments(page, box)

    def _col_of(x: float) -> int:
        return min(max(bisect.bisect_right(col_edges, x) - 1, 0), n_c - 1)

    def _split_fragment(f: dict) -> List[tuple]:
        """Split a cross-column fragment at column edges.  切点 = 组边界
        附近（±10pt）最大的字符间隙——真实值界两侧有留白，而 SLANet 的
        边缘可能偏 1-2 个字（"183,020,977,818100.00" 的真界在 8.2pt
        间隙处，网格边却在 "100.00" 内部）；找不到间隙的墨迹连续段
        （CJK 词被斜切）并入前段，绝不硬切。子段按自身中心归属列。"""
        chars = f["chars"]
        n = len(chars)
        if n == 0:
            return []
        # 按字符中心分箱 → 组；组间即候选切点
        groups: List[tuple] = []
        cur = [chars[0]]
        cur_ci = _col_of(chars[0][0])
        for ch in chars[1:]:
            ci = _col_of(ch[0])
            if ci == cur_ci:
                cur.append(ch)
            else:
                groups.append((cur_ci, cur))
                cur = [ch]
                cur_ci = ci
        groups.append((cur_ci, cur))
        if len(groups) == 1:
            return [(cur_ci, f["text"], f["x0"], f["x1"])]
        cuts: List[int] = []
        start = 0
        for gi in range(len(groups) - 1):
            edge = col_edges[groups[gi][0] + 1]
            best_w, best_i = 0.35, None
            lo_j = max(1, start + 1)
            hi_j = min(n - 1, start + len(groups[gi][1]) + 6)
            for j in range(lo_j, hi_j + 1):
                g = chars[j][1] - chars[j - 1][2]
                mid = (chars[j - 1][2] + chars[j][1]) / 2.0
                if abs(mid - edge) <= 10.0 and g > best_w:
                    best_w, best_i = g, j
            if best_i is not None:
                cuts.append(best_i)
                start = best_i
            # 无间隙：不设切点，该组合并进前段（CJK 斜切保护）
        pieces: List[tuple] = []
        start = 0
        for cut in cuts + [n]:
            if cut <= start:
                continue
            cs = chars[start:cut]
            px0 = min(c[1] for c in cs)
            px1 = max(c[2] for c in cs)
            pieces.append((_col_of((px0 + px1) / 2.0),
                           "".join(c[3] for c in cs), px0, px1))
            start = cut
        # 相邻同列子段合并
        merged: List[tuple] = []
        for ci, text, px0, px1 in pieces:
            if merged and merged[-1][0] == ci:
                p = merged[-1]
                merged[-1] = (ci, p[1] + text, p[2], px1)
            else:
                merged.append((ci, text, px0, px1))
        return merged

    # group fragments per row interval (by fragment vertical centre)
    by_row: dict = {}
    for f in frags:
        ri = min(max(bisect.bisect_right(row_edges, f["cy"]) - 1,
                     0), n_r - 1)
        by_row.setdefault(ri, []).append(f)

    # Gap completion: a wide inter-fragment gap (>= GAPVOTE_MIN_GAP)
    # confirmed by several rows but lacking a column edge within ±3pt means
    # the edge set is UNDER-FIT — the candidate competition tied on crossing
    # rates and an under-fit candidate won （资产负债摘要表: 追溯后/
    # 追溯前 boundary missing, "2,759,2372,752,710" fused）.  Insert
    # boundaries at the confirmed gap midpoints — whitespace-only
    # insertion, monotone safe.  阈值 4pt：片段间隙本就 ≥FRAG_GAP(3pt)，
    # 3-4pt 的窄缝多为行内混排空格/窄标签间隙，不是列界。
    gap_votes: dict = {}
    n_multi = 0
    all_rows = [sorted(rfs, key=lambda f: f["x0"])
                for rfs in by_row.values()]
    for rfs in all_rows:
        if len(rfs) < 2:
            continue
        n_multi += 1
        for a, b in zip(rfs, rfs[1:]):
            gap = b["x0"] - a["x1"]
            if gap >= GAPVOTE_MIN_GAP:
                mid = (a["x1"] + b["x0"]) / 2.0
                # ±10pt: 右对齐数值列的参差空白（gap 中点每行漂移
                # 十余 pt）不算缺失边界——附近已有边界即认账；真正的
                # 缺失边界（p8 追溯后/追溯前，缺口 20pt+）仍会投票
                if not any(abs(mid - e) <= 10.0 for e in col_edges[1:-1]):
                    gap_votes.setdefault(round(mid / 5.0), []).append(mid)
    need_votes = max(2, round(0.3 * n_multi))
    for key, votes in list(gap_votes.items()):
        if len(votes) >= need_votes:
            import bisect as _b
            mid = sum(votes) / len(votes)
            # 交叉否决：候选边界上有片段横穿（如间隔姓名"周 松"的空隙，
            # "李战明" 无空格整名横穿）→ 不是列界。真列界两侧的文字
            # 永远分属不同片段
            crossed = sum(1 for rfs in all_rows
                          for f in rfs
                          if f["x0"] + 1.0 < mid < f["x1"] - 1.0)
            if crossed == 0:
                _b.insort(col_edges, mid)
    n_c = len(col_edges) - 1

    cells: List[CellOutput] = []
    for ri in sorted(keep):
        row_frags = sorted(by_row.get(ri, []), key=lambda f: (f["y0"], f["x0"]))
        # merge same-column fragments from multiple lines (wrapped cells)
        placed: dict = {}
        order: List[tuple] = []
        for f in row_frags:
            # narrow-column split for cross-edge fragments
            covered = _col_of(f["x0"]) != _col_of(f["x1"]) or \
                any(col_edges[k] > f["x0"] and col_edges[k] < f["x1"]
                    for k in range(1, len(col_edges) - 1))
            pieces = _split_fragment(f) if covered else \
                [(_col_of(f["cx"]), f["text"], f["x0"], f["x1"])]
            for ci, text, px0, px1 in pieces:
                if not text.strip():
                    continue
                if ci not in placed:
                    placed[ci] = []
                    order.append((ci, f["y0"]))
                placed[ci].append({"text": text, "y0": f["y0"],
                                   "x0": px0, "x1": px1})
        if not placed:
            continue
        # header mode: the whole row is 1-2 wide fragments
        wide = [f for f in row_frags
                if f["x1"] - f["x0"] >= SPAN_CELL_WIDTH * colw]
        if len(row_frags) <= 2 and len(wide) == len(row_frags) and n_c >= 3:
            f = row_frags[0]
            c0 = min(max(bisect.bisect_right(col_edges, f["x0"]) - 1,
                         0), n_c - 1)
            c1 = min(max(bisect.bisect_right(col_edges, f["x1"]) - 1,
                         0), n_c - 1)
            if c1 > c0:
                cells.append(CellOutput(
                    text=f["text"], row=ri, col=c0,
                    col_span=c1 - c0 + 1,
                    bbox=(col_edges[c0], row_edges[ri],
                          col_edges[c1 + 1], row_edges[ri + 1])))
                continue
        for ci, _ly in sorted(order):
            fs = sorted(placed[ci], key=lambda f: (f["y0"], f["x0"]))
            parts: List[str] = []
            last_y = None
            last_x1 = None
            for f in fs:
                if last_y is not None and f["y0"] - last_y > 3.0:
                    parts.append("\n")
                elif last_x1 is not None and f["x0"] - last_x1 > 1.2:
                    # 同行拆出的子段间隙 >1.2pt → 词间空格
                    parts.append(" ")
                parts.append(f["text"])
                last_y = f["y0"]
                last_x1 = f["x1"]
            text = "".join(parts).strip()
            if text:
                cells.append(CellOutput(
                    text=text,
                    bbox=(col_edges[ci], row_edges[ri],
                          col_edges[ci + 1], row_edges[ri + 1]),
                    row=ri, col=ci,
                ))
    return cells, col_edges


# ── Wrapped-label band merge（三线表折行标签） ─────────────────────────────

WRAP_FILL = 0.62       # 折行首行顶满标签列宽度的比例阈值
WRAP_LEFT_TOL = 15.0   # 续行起始 x 与标签列左缘的容差（pt）
_WRAP_COMPLETE_END = re.compile(r'[：:；;。．）)％%]$|\d$|\d\s*日$')
_WRAP_ENUM_PREFIX = (
    re.compile(r'^[（(]?\d{1,3}[.．](?!\d)\s*\S'),          # 1. / 3．内容
    re.compile(r'^[（(]?\d{1,3}[、)）]\s*\S'),               # (3) / 1、
    re.compile(r'^[（(]?[一二三四五六七八九十]{1,3}[)）、.．]\s*\S'),  # 一、/（一）
    re.compile(r'^\(?[ivxIVXa-zA-Z]{1,4}[.、)）]\s*\S'),   # (a) / iv.
    re.compile(r'^[•·●▪◦‣■○–─-]\s*\S'),                   # bullets
    re.compile(r'^[①-⑳㉑-㉟]\s*\S'),                        # circled
)


_HEADER_NUM_RE = re.compile(r'\d[\d,.，]*\d|\d')


def _merge_header_bands(rows: list, edges: List[float]) -> list:
    """
    表头折行合并：无行线表格的表头单元格文字折成 2-3 行时，文本带把
    一个表头拆成多行——薪酬表"2024 年在本公司领取的税前报酬总额
    （人民币千元）"跨三行，每行各自成带成行，表头碎裂串格。

    判据：第一个含数字片段的带之前（数据行的签名是数字）全是不含数字
    的纯文字带，且这些带的片段铺开的列宽 < 70% 网格宽（表头折行只
    占自己那一列，数据行横跨全宽）→ 合并成一个表头带。
    数据行首行（含数字）绝不被并入；纯文字表格（无任何数字带）不动。
    """
    if len(rows) < 3 or len(edges) < 3:
        return rows
    grid_w = edges[-1] - edges[0]

    def has_num(band) -> bool:
        return any(_HEADER_NUM_RE.search(t) for _x0, _x1, t in band[2])

    def span_frac(band) -> float:
        xs = [x for x0, x1, _t in band[2] for x in (x0, x1)]
        return (max(xs) - min(xs)) / grid_w if xs else 0.0

    first_num = next((i for i, r in enumerate(rows) if has_num(r)), None)
    if first_num is None or first_num < 2:
        return rows
    head = rows[:first_num]
    if any(span_frac(b) >= 0.70 for b in head):
        return rows
    merged = [min(b[0] for b in head), max(b[1] for b in head),
              sorted((s for b in head for s in b[2]),
                     key=lambda s: (s[0], s[1]))]
    return [merged] + rows[first_num:]


def _wrap_veto(t1: str, t2: str) -> bool:
    """
    结构性标签否决：几何四条件无法区分"折行标签"与"独立结构行 +
    数据行"时，靠文字收尾特征裁决。

    首行以冒号/句号/右括号/百分号/数字/日期（"…12 月31 日"）收尾，
    首行或次行以枚举头（"一、"/"1."/"(a)"/"①"）起始 → 首行是完整
    的结构性标签（"2024 年12 月31 日"、"其中："、"一、合营企业"），
    独立成行，否决合并。折行标签（"…扣除非经"）不带任何收尾特征，
    不触发否决。
    """
    t1 = (t1 or '').strip()
    t2 = (t2 or '').strip()
    if not t1 or not t2:
        return True
    if _WRAP_COMPLETE_END.search(t1):
        return True
    if any(rx.match(t1) or rx.match(t2) for rx in _WRAP_ENUM_PREFIX):
        return True
    if re.match(r'^[（(]?\d', t2):
        return True
    return False


def _merge_wrapped_label_bands(rows: list, edges: List[float]) -> list:
    """
    三线表（无行线）的折行标签修复：一个逻辑行的标签太长被折成两行、
    数值与标签末行底部对齐时，文本带把一行拆成两行——首行只有标签
    前半（值列全空）、次行是标签后半 + 全部数值。

    合并判据 = 几何四条件 AND 无结构否决：

    几何（同时成立）：
      1. 首行所有片段都在标签列 x 范围内（值列无墨迹）；
      2. 首行最右墨迹顶到标签列宽度的 85% 以上——折行的原因就是
         顶边；
      3. 次行带数值片段（数值与标签末行底部对齐的签名）；
      4. 次行最左片段从标签列左缘接排（续行接排）。

    结构否决（_wrap_veto）：首行以冒号/括号/百分号/数字/日期收尾、
    或次行以枚举头起始 → 首行是完整的结构性标签，独立成行，不并。
    这是对"首行顶满但真是独立行"这一反例的正面回答：此类行几乎
    总带收尾特征或枚举头；两者皆无的长平文本行，统计上属于折行。

    连锁吸收支持三行折行。规则路径不受影响（有行线的表由横线定行）。
    """
    if len(edges) < 3 or len(rows) < 2:
        return rows
    lx0, lx1 = edges[0], edges[1]
    lw = lx1 - lx0

    def label_only_fills(band) -> bool:
        spans = band[2]
        if not spans:
            return False
        if any(x1 > lx1 + 4.0 for x0, x1, _t in spans):
            return False          # 有墨迹越出标签列 = 带了值
        return max(x1 for x0, x1, _t in spans) >= lx0 + WRAP_FILL * lw

    def has_value_frags(band) -> bool:
        return any(x0 > lx1 + 2.0 for x0, x1, _t in band[2])

    out: list = []
    i = 0
    while i < len(rows):
        cur = rows[i]
        j = i + 1
        while (j < len(rows)
               and label_only_fills(cur)
               and has_value_frags(rows[j])
               and min(x0 for x0, x1, _t in rows[j][2]) <= lx0 + WRAP_LEFT_TOL
               and not _wrap_veto(
                   ''.join(s[2] for s in sorted(cur[2])),
                   min(rows[j][2], key=lambda s: s[0])[2])):
            cur = [min(cur[0], rows[j][0]), max(cur[1], rows[j][1]),
                   sorted(cur[2] + rows[j][2], key=lambda s: (s[0], s[1]))]
            j += 1
        out.append(cur)
        i = j
    return out


def rebuild_grid(page: fitz.Page, box: Rect,
                 col_edges: Optional[List[float]] = None,
                 src_cols: Optional[int] = None,
                 ) -> Tuple[List[CellOutput], int, int, List[float], list]:
    """
    Re-segment the table region into a row x column grid from geometry
    alone (drawn rules + text lines + gutter evidence).

    Row boundaries: drawn full-width rules when they structure the whole
    region — a ruled table's rules are the exact row edges, immune to the
    wrapped 2-line cells that split text bands.  Otherwise text-band
    midpoints.

    Columns: two candidates compete — the existing grid's own x-edges
    (consolidated to the grid's column count) and span-gap gutters — and
    the span-crossing rate picks the winner: a sound column boundary
    falls in the whitespace between text spans, so drifted SLANet edges
    that slice through spans lose to the gutters, and vice versa.

    Returns (cells, num_rows, num_cols, col_edges, rows).
    """
    rows = _rows(page, box)
    if not rows:
        return [], 0, 0, [box[0], box[2]], []

    def _clamp(es: List[float]) -> List[float]:
        # Clamp (never drop) edges into the box: SLANet cells legitimately
        # overhang the detection box by a few pt, and dropping the outermost
        # edge amputates the whole label column (中国石油 分部表).
        es = sorted({min(max(e, box[0]), box[2]) for e in es})
        return es if len(es) >= 2 else [box[0], box[2]]

    cand_hint = cand_gutter = cand_mid = None
    if col_edges and len(col_edges) >= 2:
        cand_hint = _clamp(col_edges)
        if src_cols is not None:
            cand_hint = _consolidate_col_edges(cand_hint, src_cols)
    cand_gutter = _clamp(_column_edges(rows, box[0], box[2]))
    cand_mid = _clamp(_span_midpoint_edges(rows, box))

    cands = [("hint", cand_hint), ("gutter", cand_gutter),
             ("midpoint", cand_mid)]
    cands = [(n, c) for n, c in cands if c]
    # Structure sufficiency: a candidate too poor to separate the columns
    # of a typical row (2-column degenerate gutter fits) trivially scores
    # zero crossings and must not compete — require interior edges >= the
    # median spans per multi-span row minus one.
    span_counts = sorted(len(spans) for _y, _y2, spans in rows
                         if len(spans) >= 2)
    min_interior = max(1, (span_counts[len(span_counts) // 2] - 1)) \
        if span_counts else 1
    cands = [(n, c) for n, c in cands
             if len(c) - 2 >= min_interior]
    if not cands:
        cands = [("gutter", cand_gutter)]
    best_name, edges = cands[0]
    best_rate = _span_crossing_rate(rows, edges)
    for name, cand in cands[1:]:
        rate = _span_crossing_rate(rows, cand)
        # 平票取列数更少的候选：跨列率同为 0 时，多出来的边只可能是
        # 行内空格上的假边界（幽灵空列的来源）
        if rate < best_rate - 1e-9 or \
                (abs(rate - best_rate) <= 1e-9 and len(cand) < len(edges)):
            best_name, best_rate, edges = name, rate, cand
    n_cols = len(edges) - 1
    rules = _ruled_row_edges(page, box)
    row_edges = None
    keep_rows = None
    if rules:
        cand = [(a, b) for a, b in zip(rules, rules[1:]) if b - a >= 4.0]
        # Density guard: hybrid tables (ruled head, wireless tail held
        # together by subtotal double-rules) yield sparse rule intervals
        # that would fuse many text rows per cell.  Rules win only when
        # they structure at least 80% of the text rows — 按带覆盖计：
        # 折行单元格（股东名、薪酬表头）把文本带撑到比行线区间多，
        # 按"区间数 ≥ 0.8×带数"计会错杀行线，按"带被某个行线区间
        # 完整包含"计则不受折行影响
        covered = sum(
            1 for y0, y1, _s in rows
            if any(a - 1.0 <= y0 and y1 <= b + 1.0 for a, b in cand))
        if rows and covered >= 0.8 * len(rows):
            # 三线格数据区：上下边线之间没有行线，多条数据带落在同一个
            # 行线区间里（p53 关联交易表 3 行数据 47,026/123,253/170,279
            # 被压成一行）。相邻带细分判据：只要前带带值列墨迹就切——
            # 唯一例外是折行标签（前带仅标签列有墨迹且不是结构性收尾
            # 行），它并入后带；"归属于："这类冒号收尾的结构行独立成行
            def _has_value_ink(band) -> bool:
                return any(x1 > edges[1] + 2.0 for x0, x1, _t in band[2])

            def _structural_end(band) -> bool:
                t = "".join(s[2] for s in sorted(band[2])).rstrip()
                return bool(t) and (t[-1] in "：:；;。" or
                                    any(rx.match(t) for rx in _WRAP_ENUM_PREFIX))

            def _label_only(band) -> bool:
                return not _has_value_ink(band)

            intervals = []
            for a, b in cand:
                inner = [r for r in rows
                         if a - 1.0 <= r[0] and r[1] <= b + 1.0]
                if len(inner) >= 2:
                    bounds = [a]
                    for k in range(len(inner) - 1):
                        b1, b2 = inner[k], inner[k + 1]
                        if _label_only(b1) and not _structural_end(b1):
                            continue        # 折行标签并入后带
                        bounds.append((b1[1] + b2[0]) / 2)
                    bounds.append(b)
                    intervals.extend(zip(bounds, bounds[1:]))
                else:
                    intervals.append((a, b))
            n_rows = len(intervals)
            # assignment runs over the FULL rule set; only kept intervals
            # emit cells (double-rule slivers hold no text)
            row_edges = sorted({v for ab in intervals for v in ab})
            keep_rows = [row_edges.index(a) for a, _b in intervals]
    if row_edges is None:
        rows = _merge_header_bands(rows, edges)
        # y boundaries: bbox top, then band midpoints, then bbox bottom
        yb = [box[1]]
        for a, b in zip(rows, rows[1:]):
            yb.append((a[1] + b[0]) / 2)
        yb.append(box[3])
        intervals = list(zip(yb, yb[1:]))
        n_rows = len(rows)
        row_edges = yb
        rows = _merge_wrapped_label_bands(rows, edges)
        if len(rows) != len(intervals):
            # recompute boundaries from the merged band set
            yb = [box[1]]
            for a, b in zip(rows, rows[1:]):
                yb.append((a[1] + b[0]) / 2)
            yb.append(box[3])
            intervals = list(zip(yb, yb[1:]))
            n_rows = len(rows)
            row_edges = yb
    cells, edges = _cells_from_fragments(page, box, row_edges, edges,
                                        keep_rows=keep_rows)
    n_cols = len(edges) - 1   # 间隙补边可能新增列，以最终列边为准
    if keep_rows is not None:
        # re-index rows from the full rule set down to the kept intervals
        remap = {old: new for new, old in enumerate(sorted(keep_rows))}
        for c in cells:
            c.row = remap.get(c.row, c.row)
    return cells, n_rows, n_cols, edges, rows


# ── Numeric coverage ──────────────────────────────────────────────────────

NUM_TOKEN = re.compile(r'\d[\d,，.]*\d|\d')


def _numeric_gap(page: fitz.Page, table_bbox: Rect,
                 cells: List[CellOutput]) -> int:
    """
    Count numeric tokens of the region text that are absent from the
    cells' text (comma/space-insensitive substring test).  SLANet cell
    bboxes sometimes amputate the last glyph of a right-aligned value
    ('25,688' → '25,68') — one or two chars below the uncovered-text
    ratio trigger, but fatal for a financial corpus.
    """
    text = "".join(c.text for c in cells).replace(",", "").replace(" ", "")
    clip = fitz.Rect(table_bbox[0] - 4.0, table_bbox[1] - 4.0,
                     table_bbox[2] + 4.0, table_bbox[3] + 4.0)
    region = page.get_text("text", clip=clip)
    missing = 0
    for tok in NUM_TOKEN.findall(region):
        v = tok.replace(",", "").replace(" ", "")
        if len(v) < 3 or re.fullmatch(r"(19|20)\d{2}", v):
            continue
        if v not in text:
            missing += 1
    return missing


# ── Public entry ──────────────────────────────────────────────────────────


def complete_table(page: fitz.Page,
                   table_bbox: Optional[Rect],
                   cells: List[CellOutput],
                   num_rows: int,
                   num_cols: int,
                   ) -> Tuple[List[CellOutput], int, int, bool]:
    """
    Replace an under-segmented / misaligned / incomplete SLANet grid with
    a geometric rebuild when a trigger fires and the rebuild's text
    recall is no worse.  Triggers: uncovered region text
    (under-segmentation), structure misalignment (cells fusing two text
    rows or slicing through them), numeric gap (region numeric tokens
    absent from the cells — SLANet cell-edge truncation).

    Returns (cells, num_rows, num_cols, replaced).
    """
    if not (needs_completion(page, table_bbox, cells)
            or structure_suspect(page, table_bbox, cells)
            or (table_bbox is not None and cells
                and _numeric_gap(page, table_bbox, cells) > 0)):
        return cells, num_rows, num_cols, False

    new_cells, new_rows, new_cols, new_edges, rows = rebuild_grid(
        page, table_bbox, col_edges=_grid_col_edges(cells),
        src_cols=num_cols)
    if not new_cells:
        return cells, num_rows, num_cols, False

    ref = Counter(_norm(page.get_text(
        "text", clip=fitz.Rect(*table_bbox))))
    before = _recall(ref, cells)
    after = _recall(ref, new_cells)
    if after + 1e-9 < before:
        _log.debug(f"completion declined (recall {before:.3f}→{after:.3f})")
        return cells, num_rows, num_cols, False

    # Structure guard: a rebuild that collapses columns must prove the
    # grid it replaces was bogus — its column edges cutting through more
    # text spans than the rebuild's.  A genuine SLANet grid (edges in
    # gutters, 0 violations) is never replaced by a poorer-columned one.
    if new_cols < num_cols:
        old_viol = _split_violations(rows, _grid_col_edges(cells))
        new_viol = _split_violations(rows, new_edges)
        if old_viol <= new_viol:
            _log.debug(f"completion declined (cols {num_cols}→{new_cols}, "
                       f"violations {old_viol}→{new_viol})")
            return cells, num_rows, num_cols, False

    _log.info(f"completion: rebuilt {num_rows}x{num_cols} SLANet grid as "
              f"{new_rows}x{new_cols} (text recall {before:.3f}→{after:.3f})")
    return new_cells, new_rows, new_cols, True
