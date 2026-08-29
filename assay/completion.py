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


def rebuild_grid(page: fitz.Page, box: Rect
                 ) -> Tuple[List[CellOutput], int, int, List[float], list]:
    """
    Re-segment the table region into a row x column grid from text-line
    and gutter geometry alone.

    Row cells tile the full region vertically (boundaries at inter-band
    midpoints, outer edges at the bbox) the way a drawn grid would — text
    assignment is unchanged (boundaries fall in whitespace) and the grid
    honestly covers the whole table.

    Returns (cells, num_rows, num_cols, col_edges, rows).
    """
    rows = _rows(page, box)
    if not rows:
        return [], 0, 0, [box[0], box[2]], []
    edges = _column_edges(rows, box[0], box[2])
    n_cols = len(edges) - 1
    # y boundaries: bbox top, then band midpoints, then bbox bottom
    yb = [box[1]]
    for a, b in zip(rows, rows[1:]):
        yb.append((a[1] + b[0]) / 2)
    yb.append(box[3])
    cells: List[CellOutput] = []
    for ri in range(len(rows)):
        for ci in range(n_cols):
            rect = fitz.Rect(edges[ci], yb[ri], edges[ci + 1], yb[ri + 1])
            txt = page.get_text("text", clip=rect).strip()
            if txt:
                cells.append(CellOutput(
                    text=txt, bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    row=ri, col=ci,
                ))
    return cells, len(rows), n_cols, edges, rows


# ── Public entry ──────────────────────────────────────────────────────────


def complete_table(page: fitz.Page,
                   table_bbox: Optional[Rect],
                   cells: List[CellOutput],
                   num_rows: int,
                   num_cols: int,
                   ) -> Tuple[List[CellOutput], int, int, bool]:
    """
    Replace an under-segmented SLANet grid with a geometric rebuild when
    the trigger fires and the rebuild's text recall is no worse.

    Returns (cells, num_rows, num_cols, replaced).
    """
    if not needs_completion(page, table_bbox, cells):
        return cells, num_rows, num_cols, False

    new_cells, new_rows, new_cols, new_edges, rows = rebuild_grid(
        page, table_bbox)
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
