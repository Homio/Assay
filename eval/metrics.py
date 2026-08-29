#!/usr/bin/env python3
"""
Shared evaluation metrics for the Assay pipeline.

Three quality dimensions, all computed from ground_truth.json plus the
pipeline's own JSON export (assay/output.py) — no GPU, no model imports:

  box      — detection: greedy IoU matching of table_bbox against GT boxes
             (same algorithm as eval.py; ported here so both stay in sync)
  coverage — cell-level area coverage: how much of the reference box is
             actually covered by extracted cells.  SLANet under-segmentation
             (the 1x1 giant-cell failure) shows up here even when the
             detection box itself is perfect.
  text     — content completeness: bag-of-chars recall/precision of the
             concatenated cell text against the native PDF text inside the
             reference box (whitespace-normalised; robust for CJK text).

Reference box priority: GT box when annotated, else the detection
table_bbox (self-coverage — the only GT-free content signal).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

Rect = Tuple[float, float, float, float]

IOU_THRESHOLD = 0.7      # TP threshold, identical to eval.py
WEAK_THRESHOLD = 0.9     # TP but localisation weaker than this = weak match
COVERAGE_FLOOR = 0.90    # a TP table below this coverage silently loses content
TEXT_FLOOR = 0.95        # a TP table below this text recall silently loses text
OVERLAP_FRAC = 0.05      # pred/pred area overlap that is physically impossible


# ── Geometry ─────────────────────────────────────────────────────────────


def area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def inter_area(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def iou(a, b) -> float:
    inter = inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes) -> Optional[List[float]]:
    """Bounding box of a set of boxes; None when empty."""
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def boxes_cover(boxes, ref) -> float:
    """Approximate share of ``ref`` area covered by ``boxes``.

    Uses the bounding-union (not exact multi-box union); cell grids tile
    their table without overlap, so bounding-union is exact for full grids
    and only mildly pessimistic for gapped ones.
    """
    u = union_box(boxes)
    if u is None or area(ref) <= 0:
        return 0.0
    return inter_area(u, ref) / area(ref)


# ── Box-level matching (ported from eval.py — keep semantics identical) ────


def match_page(preds, gts, thr: float = IOU_THRESHOLD):
    """
    Greedy best-IoU matching (highest IoU pair first).

    Returns (matches, fp, miss) where matches = [(gi, pi, iou)],
    fp = [(pi, box)], miss = [(gi, box)].
    """
    pairs = sorted(
        ((iou(p, g), gi, pi)
         for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, gi, pi in pairs:
        if v < thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((gi, pi, v))
    fp = [(pi, p) for pi, p in enumerate(preds) if pi not in used_p]
    miss = [(gi, g) for gi, g in enumerate(gts) if gi not in used_g]
    return matches, fp, miss


def overlap_pairs(boxes, frac: float = OVERLAP_FRAC) -> int:
    """Count box pairs on one page sharing more than ``frac`` of the smaller.

    Two genuine tables never share page area (eval.py pred_overlaps).
    """
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter = inter_area(boxes[i], boxes[j])
            if inter <= 0:
                continue
            lo = min(area(boxes[i]), area(boxes[j]))
            if lo > 0 and inter / lo >= frac:
                n += 1
    return n


# ── Output-JSON accessors ────────────────────────────────────────────────


def table_box(t: Dict) -> Optional[List[float]]:
    """Detection bbox from the JSON export; None for pre-table_bbox exports."""
    try:
        b = t["prov"][0].get("bbox")
    except (KeyError, IndexError, TypeError):
        return None
    return list(b) if b else None


def table_page(t: Dict) -> int:
    """0-based page number from the JSON export."""
    return int(t["prov"][0]["page_no"])


def cell_boxes(t: Dict) -> List[List[float]]:
    """Valid (non-degenerate) cell rects [x0, y0, x1, y1] of one table."""
    out = []
    for row in t["data"]["grid"]:
        for c in row:
            b = c.get("bbox") or {}
            r = [b.get("l", 0), b.get("t", 0), b.get("r", 0), b.get("b", 0)]
            if r[2] > r[0] and r[3] > r[1]:
                out.append(r)
    return out


def cell_text(t: Dict) -> str:
    """Concatenated text of all cells (reading order)."""
    return "".join(
        c.get("text", "") for row in t["data"]["grid"] for c in row)


def cell_stats(t: Dict) -> Tuple[int, int]:
    """(total grid slots, slots with empty text) — empty-cell diagnostics."""
    total, empty = 0, 0
    for row in t["data"]["grid"]:
        for c in row:
            total += 1
            if not c.get("text", "").strip():
                empty += 1
    return total, empty


# ── Text completeness ────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Whitespace-free text; CJK financial content carries no spaces, so
    whitespace is pure layout noise for accuracy measurement."""
    return _WS_RE.sub("", s or "")


def text_scores(ref: str, hyp: str) -> Tuple[float, float]:
    """
    Bag-of-chars (recall, precision) of ``hyp`` against ``ref``.

    Order-insensitive by design: cell reading order differs between
    pipelines, but the *characters present* must match the native text.
    """
    ref_bag = Counter(norm_text(ref))
    hyp_bag = Counter(norm_text(hyp))
    if not ref_bag and not hyp_bag:
        return 1.0, 1.0
    hit = sum((ref_bag & hyp_bag).values())
    recall = hit / sum(ref_bag.values()) if ref_bag else 0.0
    precision = hit / sum(hyp_bag.values()) if hyp_bag else 0.0
    return recall, precision


# ── Geometric row count (under-segmentation proxy) ───────────────────────


def count_text_rows(page, box) -> int:
    """
    Text-row bands inside ``box``: PyMuPDF lines clustered by the same
    2 pt vertical-overlap merge as assay.splitting._rows.  A SLANet grid
    with far fewer rows than this count under-segmented the table.
    """
    x0, y0, x1, y1 = box
    ys = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
            if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
                if "".join(s["text"] for s in line["spans"]).strip():
                    ys.append((ly0, ly1))
    ys.sort()
    rows = 0
    cur_y1 = None
    for ly0, ly1 in ys:
        if cur_y1 is not None and ly0 <= cur_y1 - 2.0:
            cur_y1 = max(cur_y1, ly1)
        else:
            rows += 1
            cur_y1 = ly1
    return rows
