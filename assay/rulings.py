"""
Vector ruling-line table geometry.

Financial-report tables are (>90%) drawn with real vector ruling lines.
This module recovers table rectangles directly from ``page.get_drawings()``
and provides *shrink-only* bbox refinement.  A table boundary is defined by
its own ruling lines / text spans, so a refined bbox can never absorb the
page header or a body paragraph (which carry no lines) — this replaces the
old magnetic-alignment heuristics that grew bboxes into neighbouring text.

Three tiers of evidence (best → weakest):
  A. Full grid   (H + V lines)  -> exact connected-component rectangle
  B. Ruled rows  (H lines only) -> y & x from horizontal-line extents
  C. Wireless    (no lines)     -> shrink the YOLO box to contained text spans
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import fitz

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HEADER_ZONE = 72.0      # ignore recurring page-header decoration lines (pt from top)
TOL = 2.5               # collinear / touch tolerance (pt)
MIN_SEG = 8.0           # ignore ruling segments shorter than this (pt)
MIN_RECT_W = 120.0      # discard ruling rectangles narrower than this (pt)
MIN_RECT_H = 14.0       # discard ruling rectangles shorter than this (pt).
                        # One drawn header row is ~16 pt tall; a header-only
                        # disclosure-template table must still clear the floor.
MIN_H_LINES = 2         # a real table needs at least this many horizontal lines
MIN_V_LINES = 1         # a full grid needs at least this many vertical lines
HBAND_MIN_LINES = 3     # an H-only ruled table needs at least this many rows
HBAND_ROW_GAP_MAX = 34.0  # max vertical gap between H-lines of the same band (pt)
HBAND_X_OVERLAP = 0.5   # min x-overlap ratio to keep H-lines in the same band
HEADER_BAND_MAX_H = 34.0  # a detached header band is at most this tall (1-2 rows)
HEADER_BAND_GAP = 18.0    # max gap between a header band and its body grid (pt);
                          # separate stacked tables sit further apart

# Double-underline recovery (accounting grand-total terminator) ─────────────
# The single/double separation span is NOT universal across typesetters (a
# 236-report audit found the valley anywhere from 0.25 to 4.5 pt), so it is
# calibrated per document from the whole-book span histogram; when no clean
# bimodal valley exists the recovery is disabled for that document.
UG_CALIB_MIN_CLUSTERS = 40   # need this many y-clusters book-wide to calibrate
UG_CALIB_MIN_DOUBLES = 3     # need this many double-side clusters to trust it
UG_CALIB_MIN_VALLEY = 0.5    # min empty span gap (pt) between the two modes
UG_CALIB_NOISE = 0.01        # ignore histogram buckets under this fraction
UG_COL_MIN_LINES = 3    # a value column needs >= this many short underlines
UG_VALUE_MAX_W = 260.0  # a value-cell underline is at most this wide (pt)
UG_HEADER_GAP = 16.0    # max gap between contiguous column-header text rows (pt)
UG_ROW_GAP = 20.0       # max gap between contiguous table rows in the top climb (pt)
UG_HEADER_MAX = 40.0    # a column-header block is at most this tall (pt)
UG_LEFT_REACH = 300.0   # max pt a left item-label may sit left of the value band
                        # (华夏-style small tables right-shift the value columns,
                        # leaving the row label ~275 pt out at the body margin)
UG_MIN_HEIGHT = 45.0    # discard recovered boxes shorter than this: a genuine
                        # underline table is taller, and this floor removes the
                        # stray subtotal/footnote double-rule pairs that would
                        # otherwise pose as one-row phantom tables.

Rect = Tuple[float, float, float, float]


# ──────────────────────────────────────────────────────────────────────────────
# Line-segment extraction
# ──────────────────────────────────────────────────────────────────────────────


def _merge_1d(intervals, tol=TOL):
    """Merge overlapping/touching 1D intervals [(a, b), ...]."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def extract_line_segments(page) -> Tuple[list, list]:
    """
    Extract horizontal / vertical ruling segments from vector drawings.

    Returns:
        H = [(x0, x1, y), ...]  horizontal segments (collinear-merged)
        V = [(y0, y1, x), ...]  vertical segments (collinear-merged)

    Segments living entirely in the top ``HEADER_ZONE`` are dropped so that
    recurring page-header decoration rules do not create phantom tables.
    """
    raw_h, raw_v = {}, {}

    def add_h(x0, x1, y):
        if x1 - x0 >= MIN_SEG:
            raw_h.setdefault(round(y, 1), []).append((min(x0, x1), max(x0, x1)))

    def add_v(y0, y1, x):
        if y1 - y0 >= MIN_SEG:
            raw_v.setdefault(round(x, 1), []).append((min(y0, y1), max(y0, y1)))

    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) <= 1.0:
                    add_h(p1.x, p2.x, (p1.y + p2.y) / 2)
                elif abs(p1.x - p2.x) <= 1.0:
                    add_v(p1.y, p2.y, (p1.x + p2.x) / 2)
            elif it[0] == "re":
                r = it[1]
                add_h(r.x0, r.x1, r.y0)
                add_h(r.x0, r.x1, r.y1)
                add_v(r.y0, r.y1, r.x0)
                add_v(r.y0, r.y1, r.x1)

    H, V = [], []
    for y, ivs in raw_h.items():
        if y <= HEADER_ZONE:
            continue
        for a, b in _merge_1d(ivs, TOL):
            H.append((a, b, y))
    for x, ivs in raw_v.items():
        for a, b in _merge_1d(ivs, TOL):
            if b > HEADER_ZONE:  # keep verticals that reach below the header
                V.append((a, b, x))
    return H, V


# ──────────────────────────────────────────────────────────────────────────────
# Tier A: full-grid rectangles (connected components of H+V)
# ──────────────────────────────────────────────────────────────────────────────


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _grid_rects(H, V) -> Tuple[List[Rect], set]:
    """
    Connected-components over touching H/V segments -> full-grid rectangles.

    Returns (rects, used_h_indices).  ``used_h_indices`` lets the caller feed
    the *leftover* horizontal lines to the H-only band detector.
    """
    nH, nV = len(H), len(V)
    if nH == 0:
        return [], set()
    uf = _UF(nH + nV)

    def touch(hi, vi):
        hx0, hx1, hy = H[hi]
        vy0, vy1, vx = V[vi]
        return (hx0 - TOL <= vx <= hx1 + TOL) and (vy0 - TOL <= hy <= vy1 + TOL)

    for hi in range(nH):
        for vi in range(nV):
            if touch(hi, vi):
                uf.union(hi, nH + vi)

    comps = {}
    for i in range(nH + nV):
        comps.setdefault(uf.find(i), []).append(i)

    rects: List[Rect] = []
    used_h: set = set()
    for members in comps.values():
        h_idx = [i for i in members if i < nH]
        v_idx = [i - nH for i in members if i >= nH]
        if len(h_idx) < MIN_H_LINES or len(v_idx) < MIN_V_LINES:
            continue
        hs = [H[i] for i in h_idx]
        vs = [V[i] for i in v_idx]
        x0 = min(min(h[0] for h in hs), min(v[2] for v in vs))
        x1 = max(max(h[1] for h in hs), max(v[2] for v in vs))
        y0 = min(min(h[2] for h in hs), min(v[0] for v in vs))
        y1 = max(max(h[2] for h in hs), max(v[1] for v in vs))
        if x1 - x0 < MIN_RECT_W or y1 - y0 < MIN_RECT_H:
            continue
        rects.append((x0, y0, x1, y1))
        used_h.update(h_idx)
    return rects, used_h


# ──────────────────────────────────────────────────────────────────────────────
# Tier B: horizontal-ruled bands (H lines only, no verticals)
# ──────────────────────────────────────────────────────────────────────────────


def _hband_rects(H) -> List[Rect]:
    """
    Cluster horizontal lines with no verticals into ruled-row table bands.

    Financial statements (balance sheet, income statement) are commonly drawn
    with full-width row rules and no column rules.  Group H-lines by vertical
    proximity + horizontal overlap; each band with >= HBAND_MIN_LINES rules
    becomes a rectangle.
    """
    if len(H) < HBAND_MIN_LINES:
        return []
    lines = sorted(H, key=lambda h: h[2])
    groups: List[list] = []
    cur = [lines[0]]
    for h in lines[1:]:
        gap = h[2] - cur[-1][2]
        gx0 = min(s[0] for s in cur)
        gx1 = max(s[1] for s in cur)
        ox0 = max(gx0, h[0])
        ox1 = min(gx1, h[1])
        denom = max(1.0, min(gx1 - gx0, h[1] - h[0]))
        overlap = (ox1 - ox0) / denom
        if gap <= HBAND_ROW_GAP_MAX and overlap >= HBAND_X_OVERLAP:
            cur.append(h)
        else:
            groups.append(cur)
            cur = [h]
    groups.append(cur)

    rects: List[Rect] = []
    for g in groups:
        if len(g) < HBAND_MIN_LINES:
            continue
        x0 = min(s[0] for s in g)
        x1 = max(s[1] for s in g)
        y0 = min(s[2] for s in g)
        y1 = max(s[2] for s in g)
        if x1 - x0 >= MIN_RECT_W and y1 - y0 >= MIN_RECT_H:
            rects.append((x0, y0, x1, y1))
    return rects


# ──────────────────────────────────────────────────────────────────────────────
# Public: ruling rectangles + shrink-only refinement
# ──────────────────────────────────────────────────────────────────────────────


PROSE_FRAME_MIN_LINES = 8      # ruled-prose frames stack this many rules
PROSE_FRAME_GAP_SPREAD = 3.0   # max spread of inter-rule gaps to count as
                               # metronomic (a real table's rows vary more)
PROSE_FRAME_PAGE_FRAC = 0.5    # a full-page ruled frame fills >= this much of
                               # the page height


def _is_ruled_prose_frame(rect: Rect, H: list, V: list, page) -> bool:
    """
    True when ``rect`` is a page of underlined prose masquerading as a grid,
    not a table.  Ruled-prose stationery draws one full-width rule under every
    text line and only an outer border frame, so it presents as: many
    metronomically-spaced horizontal rules, NO internal vertical divider, and
    a block filling most of the page.  A genuine table breaks at least one of
    these — it has an internal column rule, non-uniform row heights, or does
    not span the whole page.  All three signals must agree, so a real grid or
    horizontal-ruled financial band is never dropped.
    """
    x0, y0, x1, y1 = rect
    if (y1 - y0) < PROSE_FRAME_PAGE_FRAC * page.rect.height:
        return False
    ys = sorted(h[2] for h in H
                if h[2] >= y0 - TOL and h[2] <= y1 + TOL
                and (min(h[1], x1) - max(h[0], x0)) >= 0.8 * (x1 - x0))
    if len(ys) < PROSE_FRAME_MIN_LINES:
        return False
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    if not gaps or (max(gaps) - min(gaps)) > PROSE_FRAME_GAP_SPREAD:
        return False                       # rows vary -> a real table
    # An internal vertical divider (strictly between the edges and tall enough
    # to define a column) proves a real column grid.
    for vy0, vy1, vx in V:
        if x0 + 3 * TOL < vx < x1 - 3 * TOL and (vy1 - vy0) >= 2 * (gaps[0]):
            return False
    return True


def _absorb_header_bands(rects: List[Rect], V: list) -> List[Rect]:
    """
    Merge a detached header band into the body grid right below it.

    Some typesetters draw the column-header row as its own closed frame a few
    points above the table body (中国石油 style).  Left separate, the short
    header rect becomes a phantom one-row table and the body rect loses its
    header.  A short rect (<= HEADER_BAND_MAX_H) whose x-extent matches a
    rect starting within HEADER_BAND_GAP below it is such a band -- union the
    two.  The band must carry an internal vertical divider (a real header row
    separates its columns); a framed prose sentence has only its outer border
    and must stay out (华夏 p32).  A genuinely standalone header-only table
    keeps its rect because no grid follows within the gap.
    """

    def _has_inner_v(r: Rect) -> bool:
        return any(
            r[0] + 3 * TOL < vx < r[2] - 3 * TOL
            and vy0 <= r[1] + TOL and vy1 >= r[3] - TOL
            for vy0, vy1, vx in V
        )

    rects = sorted(rects, key=lambda r: r[1])
    changed = True
    while changed:
        changed = False
        for i, top in enumerate(rects):
            if top[3] - top[1] > HEADER_BAND_MAX_H or not _has_inner_v(top):
                continue
            for j, bot in enumerate(rects):
                if j == i:
                    continue
                gap = bot[1] - top[3]
                if not (-TOL <= gap <= HEADER_BAND_GAP):
                    continue
                ov = min(top[2], bot[2]) - max(top[0], bot[0])
                narrow = min(top[2] - top[0], bot[2] - bot[0])
                if narrow <= 0 or ov / narrow < 0.85:
                    continue
                rects[j] = (min(top[0], bot[0]), top[1],
                            max(top[2], bot[2]), bot[3])
                rects.pop(i)
                changed = True
                break
            if changed:
                break
    return rects


def detect_ruling_rects(page) -> Tuple[List[Rect], list, list]:
    """
    Return (ruling_rects, H, V) for a page.

    ``ruling_rects`` combines full grids (Tier A) and horizontal-ruled bands
    (Tier B).  These are independent, high-precision table candidates that do
    not depend on the visual detector at all.
    """
    H, V = extract_line_segments(page)
    grids, used_h = _grid_rects(H, V)
    remaining_h = [h for i, h in enumerate(H) if i not in used_h]
    hbands = _hband_rects(remaining_h)
    rects = grids + hbands
    rects = [
        r for r in rects
        if r[2] - r[0] >= MIN_RECT_W and r[3] - r[1] >= MIN_RECT_H
        and not _is_ruled_prose_frame(r, H, V, page)
    ]
    rects = _absorb_header_bands(rects, V)
    return rects, H, V


def _value_header_lines(page, vx0, vx1, y_lo, y_hi):
    """Text lines confined to the value band (``lx0 >= vx0 - 15``) within
    ``[y_lo, y_hi]``: these are column-header cells (本集团 / 2024年度 /
    拥有权益比例), never left-side item labels or full-width prose (which start
    at the left margin).  Returns their (y0, y1) extents, ordered top-down.
    Reads no text content."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx = (lx0 + lx1) / 2
            if lx0 >= vx0 - 15 and cx <= vx1 + 5 and ly0 >= y_lo and ly1 <= y_hi:
                out.append((ly0, ly1))
    out.sort()
    return out


def _value_band_clusters(H: list, V: list):
    """Locate the value-column underline band on a page and y-cluster its rules.

    Returns ``(vx0, vx1, clusters)`` — the band's x-extent and the tol-3pt
    y-clusters of the short underlines inside it (each cluster a list of
    ``(y, x0, x1)`` strokes) — or ``None`` when the page does not carry the
    underline-grid style (< 2 value columns of >= ``UG_COL_MIN_LINES`` short
    rules).  Shared by the per-document span calibration and the actual box
    recovery so both see identical signals.

    A genuine cell underline FLOATS: it touches no vertical line.  Short
    segments attached to a vertical are bordered-grid cell edges (shaded-cell
    borders fragment into short pieces) and would otherwise fake the value
    columns on ordinary bordered tables — those are dropped here.
    """
    def floats(y, x0, x1):
        for vy0, vy1, vx in V:
            if vy0 - TOL <= y <= vy1 + TOL and x0 - TOL <= vx <= x1 + TOL:
                return False
        return True

    shorts = [(y, x0, x1) for x0, x1, y in H
              if 8 <= x1 - x0 <= UG_VALUE_MAX_W and floats(y, x0, x1)]
    if len(shorts) < 4:
        return None

    # Value columns: cluster short-underline centres by x (tol 25 pt).
    cols: List[dict] = []
    for y, x0, x1 in sorted(shorts, key=lambda s: (s[1] + s[2]) / 2):
        cx = (x0 + x1) / 2
        if cols and cx - cols[-1]["cx"] <= 25:
            c = cols[-1]
            c["x0"] = min(c["x0"], x0); c["x1"] = max(c["x1"], x1)
            c["n"] += 1; c["cx"] = (c["x0"] + c["x1"]) / 2
        else:
            cols.append({"x0": x0, "x1": x1, "cx": cx, "n": 1})
    val_cols = [c for c in cols if c["n"] >= UG_COL_MIN_LINES]
    if len(val_cols) < 2:
        return None
    vx0 = min(c["x0"] for c in val_cols)
    vx1 = max(c["x1"] for c in val_cols)

    # y-clusters of the short rules lying in the value band (tol 3 pt).
    band = sorted(s for s in shorts
                  if vx0 - 6 <= (s[1] + s[2]) / 2 <= vx1 + 6)
    clusters: List[list] = []
    for s in band:
        if clusters and s[0] - clusters[-1][-1][0] <= 3.0:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    return vx0, vx1, clusters


def _double_span(cluster) -> float:
    """Widest y-separation between two x-OVERLAPPING strokes of a cluster.

    A true double rule is one rule drawn twice, so its strokes overlap in x.
    Underlines of *different* columns that merely land within the y-cluster
    tolerance (staggered header underlines: 行业 vs 金额 rows a point apart)
    never overlap in x and must not read as a double — they used to fabricate
    phantom header-strip tables above genuine ones.
    """
    best = 0.0
    for i in range(len(cluster)):
        yi, a0, a1 = cluster[i]
        for j in range(i + 1, len(cluster)):
            yj, b0, b1 = cluster[j]
            ov = min(a1, b1) - max(a0, b0)
            if ov >= 0.5 * min(a1 - a0, b1 - b0):
                best = max(best, abs(yj - yi))
    return best


_ug_span_cache: Dict[tuple, Optional[float]] = {}


def calibrate_double_span(doc) -> Optional[float]:
    """
    Learn this document's single/double underline separation threshold.

    Scans the whole book once (vector lines only, ~0.5 ms/page) and builds the
    histogram of value-band y-cluster spans.  A document that uses the
    double-underline convention shows a clean bimodal histogram: a large
    single-line mode (span ~ rule thickness) and a double-line mode above it,
    separated by an empty valley.  The threshold is the valley midpoint.

    Returns ``None`` — disabling the recovery for the whole document — when
    the convention cannot be learned with confidence: too few clusters, no
    empty valley >= ``UG_CALIB_MIN_VALLEY``, too few doubles, or doubles
    outnumbering singles (a physically implausible reading).  Missing a
    recovery only loses a bonus box; a wrong threshold would fabricate them.
    """
    key = (doc.name, len(doc))
    if key in _ug_span_cache:
        return _ug_span_cache[key]

    spans: List[float] = []
    for page in doc:
        spans.extend(double_span_scan_page(page))
    thr = double_span_threshold(spans)
    _ug_span_cache[key] = thr
    return thr


def double_span_scan_page(page) -> List[float]:
    """Per-page body of calibrate_double_span: this page's cluster spans."""
    H, V = extract_line_segments(page)
    vb = _value_band_clusters(H, V)
    return [_double_span(c) for c in vb[2]] if vb is not None else []


def double_span_threshold(spans: List[float]) -> Optional[float]:
    """Book-level merge of calibrate_double_span (spans in page order)."""
    thr: Optional[float] = None
    if len(spans) >= UG_CALIB_MIN_CLUSTERS:
        hist: Dict[float, int] = {}
        for s in spans:
            b = round(s * 4) / 4                      # 0.25 pt buckets
            hist[b] = hist.get(b, 0) + 1
        floor = max(3, UG_CALIB_NOISE * len(spans))
        pop = sorted(b for b, n in hist.items() if n >= floor)
        if len(pop) >= 2:
            gap, lo, hi = max(
                (b - a, a, b) for a, b in zip(pop, pop[1:]))
            if gap >= UG_CALIB_MIN_VALLEY:
                singles = sum(1 for s in spans if s <= lo + 0.125)
                doubles = sum(1 for s in spans if s >= hi - 0.125)
                if doubles >= UG_CALIB_MIN_DOUBLES and singles >= doubles:
                    thr = (lo + hi) / 2
    return thr


def recover_underline_grids(page, H: list, V: list) -> List[Rect]:
    """
    Recover per-table rectangles anchored on the accounting DOUBLE underline.

    Some financial tables are drawn *wireless* except for a short rule under
    each numeric cell and a DOUBLE rule (two lines ~1.4pt apart) beneath the
    grand-total row.  The visual detector sees the value columns but drops the
    left label column, merges vertically-stacked tables into one box, or scores
    too low to survive — so these tables are lost entirely.  The double
    underline is a universal accounting convention terminating a table, giving a
    clean per-table anchor.

    Purely geometric (reads no text content):
      * value columns = short underlines (<= ``UG_VALUE_MAX_W``) clustered by x,
        each column needing >= ``UG_COL_MIN_LINES`` rules; >= 2 columns required.
      * double underline = an H-line y-cluster whose sub-lines span >= the
        per-document calibrated threshold (see ``calibrate_double_span``);
        documents without a clean single/double bimodality are skipped.
      * table i = ``(previous_double, this_double]``; the top is anchored on the
        value-confined column-header block just above the first underline (never
        breaking on interior row gaps), and the left edge is extended to the
        item-label column.

    NOTE: interior subtotal rows also carry double rules, so this can over-split
    a table or emit a stray subtotal fragment.  It is therefore **not** a
    standalone detector: callers must add a box only where the visual detector
    found nothing overlapping (see ``detect_tables_on_page``), and the
    ``UG_MIN_HEIGHT`` floor drops one-row subtotal fragments.  Returns a list of
    Rects (possibly empty).
    """
    span_thr = calibrate_double_span(page.parent)
    if span_thr is None:        # no learnable single/double convention -> off
        return []
    vb = _value_band_clusters(H, V)
    if vb is None:
        return []
    vx0, vx1, clusters = vb
    urows = [sum(s[0] for s in c) / len(c) for c in clusters]
    doubles = [sum(s[0] for s in c) / len(c) for c in clusters
               if _double_span(c) >= span_thr]

    boxes: List[Rect] = []
    prev = 0.0
    for db in doubles:
        region = sorted(r for r in urows if prev < r <= db + 1.5)
        if not region:
            prev = db
            continue
        first_u = region[0]
        # Anchor the top on the column-header block above the first underline:
        # contiguous value-confined header rows only (small gaps), so a section
        # title / prose line at the left margin is never pulled in.
        vh = _value_header_lines(page, vx0, vx1, prev + 1, first_u + 1)
        body_top = first_u
        cur = first_u
        for ly0, ly1 in reversed(vh):
            if ly1 <= cur + 2 and (cur - ly1) <= UG_HEADER_GAP and ly0 > prev:
                body_top = ly0
                cur = ly0
            elif ly1 < cur - 1:
                break
        body_top = max(body_top, first_u - UG_HEADER_MAX)
        # The header walk stops at the first bare row, but wireless tables
        # often rule only the totals block near the double line while the
        # upper data rows carry no cell underlines at all — the recovered top
        # then amputates those rows (aging tables, disposal-of-subsidiary
        # notes).  Climb further through contiguous TABLE rows: data rows
        # whose numeric cells sit in the value band, period-header rows
        # (2025年 | 2024年) or a lone value-band cell (金额), and wrapped-cell
        # label heads (a lone left-margin line reaching into the value band)
        # once a data row has been climbed.  Prose, section headings and any
        # long drawn rule stop the walk.
        from assay.splitting import _rows as _srows, _row_kind as _skind, \
            _YEAR_RE, _PERIOD_CHAR_RE, _is_numeric_token

        def _data_row(spans):
            return (len(spans) >= 2 and _skind(spans) == "data"
                    and any(s[0] >= vx0 - 15 and _is_numeric_token(s[2])
                            for s in spans))

        region = _srows(page, (max(0.0, vx0 - UG_LEFT_REACH),
                               prev + 1, vx1 + 2, db))
        srows = [r for r in region if r[0] > prev and r[1] <= body_top + 2]
        cur = body_top
        # Wrap labels only ride on data: seed from the row right below the
        # anchored top (the ug body's first row), so a wrap head directly
        # above a data body is climbed but one above a header block is not.
        below = next((r for r in region if r[0] >= body_top - 2), None)
        prev_kind = "data" if below is not None and _data_row(below[2]) \
            else "hdr"
        hdr_top = None
        for ry0, ry1, spans in reversed(srows):
            if cur - ry1 > UG_ROW_GAP or not spans:
                break
            if any(hx1 - hx0 >= 0.5 * (vx1 - vx0) and ry1 < hy < cur
                   for hx0, hx1, hy in H):
                break
            if _data_row(spans):
                body_top = cur = ry0
                prev_kind = "data"
                hdr_top = None
                continue
            if all(s[0] >= vx0 - 15 for s in spans) and (
                    len(spans) == 1 or
                    any(_YEAR_RE.search(t) or _PERIOD_CHAR_RE.search(t)
                        for _, _, t in spans)):
                if hdr_top is None:
                    hdr_top = ry1
                if hdr_top - ry0 > UG_HEADER_MAX:
                    break
                body_top = cur = ry0
                prev_kind = "hdr"
                continue
            if (len(spans) == 1 and prev_kind == "data"
                    and spans[0][0] < vx0 - 15
                    and vx0 < spans[0][1] <= vx1 - 60):
                body_top = cur = ry0
                prev_kind = "wrap"
                continue
            break
        body_top -= 4.0     # a small pad above the header row, as annotated
        # Extend the left edge to the item-label column (labels sit left of the
        # value band; bounded by UG_LEFT_REACH so a marginal note cannot leak).
        # Span-level, not line-level: PDFs often emit a whole row (label +
        # values) as ONE text line whose centre falls inside the value band,
        # which would hide the label from a line-centre test.
        left = vx0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                ly0, ly1 = line["bbox"][1], line["bbox"][3]
                if not (ly1 >= body_top - 1 and ly0 <= db + 2):
                    continue
                for sp in line["spans"]:
                    if not sp["text"].strip():
                        continue
                    sx0, sx1 = sp["bbox"][0], sp["bbox"][2]
                    cx = (sx0 + sx1) / 2
                    if vx0 - UG_LEFT_REACH <= cx <= vx0 - 2:
                        left = min(left, sx0)
        box = (left, body_top, vx1, db + 5)
        prev = db
        if box[3] - box[1] >= UG_MIN_HEIGHT:
            boxes.append(box)
    return boxes


def is_bordered_cell_table(page, bbox: Rect, min_ratio: float = 1.8) -> bool:
    """
    True when ``bbox`` is a closed drawn grid whose full-width interior dividers
    partition it into cells of clearly **non-uniform** height.

    This is the geometric signature that separates a genuine single-column
    bordered table (e.g. a "本次现金分红情况" dividend disclosure: short centred
    header cells interleaved with tall multi-line prose cells) from a bordered
    *prose* box.  Some reports draw management-discussion callouts and
    director-bio highlights with a horizontal rule under every text line, so
    every row equals one line height (uniform).  A real table has cells that
    span differing numbers of lines, so its row heights vary.  Reads no text
    content; both the closed-grid requirement and the height-variation test are
    purely geometric, which also rejects horizontal-band-only regions (e.g. a
    period-label strip) that are not vertically bordered cells at all.
    """
    H, V = extract_line_segments(page)
    grids, _ = _grid_rects(H, V)

    def _cover(a, b):
        ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        ba = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / min(aa, ba)

    host = next((g for g in grids if _cover(bbox, g) >= 0.7), None)
    if host is None:
        return False

    gx0, gy0, gx1, gy1 = host
    w = gx1 - gx0
    ys = [gy0, gy1]
    for hx0, hx1, hy in H:
        if gy0 + 3 < hy < gy1 - 3 and (min(hx1, gx1) - max(hx0, gx0)) >= 0.9 * w:
            ys.append(hy)
    ys.sort()
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1.0]
    if len(gaps) < 2:                      # need >=1 interior divider (>=2 cells)
        return False
    return (max(gaps) / max(1.0, min(gaps))) >= min_ratio


def is_partitioned_grid(page, bbox: Rect) -> bool:
    """
    True when ``bbox`` is substantially covered by a closed drawn grid that
    carries at least one interior VERTICAL rule (>= MIN_SEG long, strictly
    inside the grid's x-range).

    The single-column veto's second geometric exception: a sparse ruled
    table (其中： label rows with values only in the right column) reads as
    one column to SLANet, and its partial-width cell dividers can never
    satisfy is_bordered_cell_table's full-width-divider + height-variation
    requirements.  Interior vertical ink is the discriminator: a ruled
    *prose* block underlines every text line but never draws column
    dividers, while any real multi-cell grid — however sparse — does.
    """
    H, V = extract_line_segments(page)
    grids, _ = _grid_rects(H, V)
    for gx0, gy0, gx1, gy1 in grids:
        ix0, iy0 = max(bbox[0], gx0), max(bbox[1], gy0)
        ix1, iy1 = min(bbox[2], gx1), min(bbox[3], gy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        lo = min(max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                 max(1.0, (gx1 - gx0) * (gy1 - gy0)))
        if inter < 0.7 * lo:
            continue
        for vy0, vy1, vx in V:
            if gx0 + 3 < vx < gx1 - 3 \
                    and min(vy1, gy1) - max(vy0, gy0) >= MIN_SEG:
                return True
    return False


def _overlap_area(a: Rect, b: Rect) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _iou(a: Rect, b: Rect) -> float:
    inter = _overlap_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(a: Rect, b: Rect) -> float:
    """Fraction of a's area lying inside b."""
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a <= 0:
        return 0.0
    return _overlap_area(a, b) / area_a


def _rowband_extent(page, rect: Rect):
    """
    Min/max x of text spans whose vertical centre falls inside the ruling
    rectangle's row band.  Used to recover a left label column (or a
    right-hand unit note) that sticks out past the ruling lines — common in
    bank statements where rules are drawn only under the numeric columns.
    Only spans within a bounded horizontal reach of the rect are considered,
    so an unrelated marginal note cannot drag the box across the page.
    """
    x0, y0, x1, y1 = rect
    reach = 340.0  # max pt a label may sit left/right of the ruled area
    ext_x0 = x0
    ext_x1 = x1
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span["bbox"]
                my = (sy0 + sy1) / 2
                if not (y0 <= my <= y1):
                    continue
                if x0 - reach <= sx0 < x0:
                    ext_x0 = min(ext_x0, sx0)
                if x1 < sx1 <= x1 + reach:
                    ext_x1 = max(ext_x1, sx1)
    return ext_x0, ext_x1


def _content_bbox(page, bbox: Rect) -> Optional[Rect]:
    """Union of text-span bboxes whose centre falls inside ``bbox``."""
    x0, y0, x1, y1 = bbox
    cx0 = cy0 = float("inf")
    cx1 = cy1 = float("-inf")
    found = False
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span["bbox"]
                mx = (sx0 + sx1) / 2
                my = (sy0 + sy1) / 2
                if x0 - 1 <= mx <= x1 + 1 and y0 - 1 <= my <= y1 + 1:
                    cx0 = min(cx0, sx0)
                    cy0 = min(cy0, sy0)
                    cx1 = max(cx1, sx1)
                    cy1 = max(cy1, sy1)
                    found = True
    if not found:
        return None
    return (cx0, cy0, cx1, cy1)


def _value_grid_bottom(
    page,
    box: Rect,
    tol: float = 4.0,
    min_rows: int = 3,
    max_row_gap: float = 55.0,
) -> Optional[float]:
    """
    Bottom ``y`` of the table's aligned value-column grid inside ``box``.

    A data table is a grid of aligned columns; in financial reports the most
    reliable, content-agnostic table signature is a **right-aligned value
    column** (accounting always right-aligns numbers).  Prose lines and the
    signature block that follow a table never right-align to those columns, so
    the lowest cell of the value-column grid marks the true table bottom.

    Purely geometric: clusters narrow spans by their right edge and walks the
    largest vertically-contiguous run of column cells.  Reads no text, so it
    works for any language / any cell content.  Returns ``None`` when no value
    column is found (the caller then keeps its own bbox unchanged).
    """
    x0, y0, x1, y1 = box
    boxw = max(1.0, x1 - x0)
    narrow = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars") or []
                if not chars:
                    continue
                sx0, sy0, sx1, sy1 = span["bbox"]
                cx = (sx0 + sx1) / 2
                cy = (sy0 + sy1) / 2
                if not (x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1):
                    continue
                # Value cells are narrow; justified prose spans are wide and are
                # excluded here so a right-margin paragraph cannot pose as a
                # column.  Left-edge clustering is deliberately avoided because
                # every left-margin label/prose line shares x0 (the page
                # margin) and would drag the grid into the signature block.
                if not 1.0 < (sx1 - sx0) < 0.35 * boxw:
                    continue
                # Accounting negatives wrap in parentheses ("(23,685.97)"),
                # and the closing paren hangs RIGHT of the digit grid the
                # column right-aligns on.  Measured only at the raw span edge
                # those rows can fall out of the column cluster and the walk
                # stops above them, amputating the table's trailing rows.
                # The raw edge entry is always kept (whitespace-padding spans
                # included — they chain-bridge adjacent edges and real
                # clusters depend on them); a paren-closed span contributes
                # an EXTRA entry measured at its last non-paren glyph, so it
                # can join the digit-aligned column as well.  Adding entries
                # only ever grows clusters — the prior behaviour stays a
                # subset and no existing bottom can shrink.
                narrow.append((sx0, sy0, sx1, sy1))
                k = len(chars) - 1
                while k >= 0 and chars[k]["c"] == " ":
                    k -= 1
                if k >= 0 and chars[k]["c"] in ")）":
                    while k >= 0 and chars[k]["c"] in ") ） ":
                        k -= 1
                    if k >= 0:
                        narrow.append((sx0, sy0, chars[k]["bbox"][2], sy1))
    if len(narrow) < min_rows:
        return None

    # Single-linkage cluster by right edge -> right-aligned columns.
    buckets: List[list] = []
    last_key = None
    for s in sorted(narrow, key=lambda s: s[2]):
        if last_key is not None and s[2] - last_key <= tol:
            buckets[-1].append(s)
        else:
            buckets.append([s])
        last_key = s[2]

    members: List[tuple] = []
    for b in buckets:
        if len({round(s[1]) for s in b}) >= min_rows:
            members.extend(b)
    if not members:
        return None

    # Keep the largest vertically-contiguous group of column cells (the table
    # body); this ignores a stray aligned cell far above or below the table.
    members.sort(key=lambda s: s[1])
    groups: List[list] = [[members[0]]]
    for s in members[1:]:
        if s[1] - groups[-1][-1][1] > max_row_gap:
            groups.append([s])
        else:
            groups[-1].append(s)
    body = max(groups, key=len)
    return max(s[3] for s in body)


def _caption_top(page, box: Rect, ruling_rects: List[Rect]) -> Optional[float]:
    """
    Top ``y`` of the table body, excluding a centred title/caption block.

    Financial statements print a centred title block above the table
    (company name / statement name / period / unit note), separated from the
    table by the header's top ruling line.  A wireless (Tier C) box shrinks to
    text and swallows that title because it is just more text with no line to
    stop it.

    Purely geometric fix: find the topmost table-width ruling edge below the
    box top; if **every** text line above that edge is horizontally centred on
    the table (a caption never aligns to the label column or the value
    columns), the block is a title and the true table top is the ruling edge.
    Reads no text content, so it is language-agnostic.  Returns ``None`` when
    there is no such ruling edge or the rows above it are not all centred (the
    caller then keeps its own top unchanged).
    """
    x0, y0, x1, y1 = box
    boxw = max(1.0, x1 - x0)
    cx_box = (x0 + x1) / 2

    y_rule: Optional[float] = None
    for r in ruling_rects:
        if (r[2] - r[0]) >= 0.5 * boxw and r[1] > y0 + 2:
            if y_rule is None or r[1] < y_rule:
                y_rule = r[1]
    if y_rule is None:
        return None

    # Group text lines above the ruling edge by their vertical position.
    rows: Dict[int, list] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            lcx = (lx0 + lx1) / 2
            lcy = (ly0 + ly1) / 2
            if y0 - 1 <= lcy < y_rule and x0 - 1 <= lcx <= x1 + 1:
                r = rows.setdefault(round(ly0), [lx0, lx1])
                r[0] = min(r[0], lx0)
                r[1] = max(r[1], lx1)
    if not rows:
        return None

    # A caption is centred on the table; a genuine header/label row is not.
    for lx0, lx1 in rows.values():
        if abs((lx0 + lx1) / 2 - cx_box) > 0.15 * boxw:
            return None
    return y_rule


def _text_rows(
    page, x0: float, x1: float, y_lo: float, y_hi: float
) -> List[Tuple[float, float]]:
    """
    Vertical (y0, y1) extents of the text *rows* whose horizontal centre lies in
    the column span ``[x0, x1]`` and that fall within ``[y_lo, y_hi]``.

    Spans that overlap vertically (e.g. every cell of one column-header row) are
    merged into a single row, so the result is the ordered sequence of text
    baselines the table is built from.  Used to quantise a box edge to the
    whitespace between rows.  Reads no text content.
    """
    spans: List[Tuple[float, float]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx = (lx0 + lx1) / 2
            if x0 - 1 <= cx <= x1 + 1 and ly1 >= y_lo and ly0 <= y_hi:
                spans.append((ly0, ly1))
    if not spans:
        return []
    spans.sort()
    rows: List[list] = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= rows[-1][1] - 2.0:          # real vertical overlap => same row
            rows[-1][0] = min(rows[-1][0], a)
            rows[-1][1] = max(rows[-1][1], b)
        else:
            rows.append([a, b])
    return [(r[0], r[1]) for r in rows]


def _snap_edges_to_line_gaps(page, box: Rect) -> Tuple[float, float]:
    """
    Quantise the top/bottom edges of ``box`` to the whitespace between text rows.

    The visual detector places box edges to sub-line precision, so an edge can
    land in the *interior* of a text line -- e.g. a spanning column-group header
    ("本集团" / "本行") printed above the column labels, which YOLO bisects
    horizontally at ~0.94 confidence.  A table boundary must never cut through a
    line: a line is atomic and belongs wholly inside or wholly outside.

    Purely geometric resolution, reading no text: when an edge bisects row *i*,
    that row joins whichever neighbouring block it is more tightly spaced with
    (the smaller inter-row gap is normal line spacing within a block; the larger
    gap is the real block boundary).  The edge is then moved into that larger
    gap, so the bisected line is either fully kept (it clings to the table body)
    or fully dropped (it clings to the title / prose above).  The move is bounded
    by one row height plus the adjacent gap, so it can never reach unrelated
    text.  Returns the (possibly unchanged) ``(top_y, bottom_y)``.
    """
    x0, y0, x1, y1 = box
    rows = _text_rows(page, x0, x1, y0 - 40.0, y1 + 40.0)
    if len(rows) < 2:
        return y0, y1
    ny0, ny1 = y0, y1

    for i, (a, b) in enumerate(rows):
        if a + 1.0 < y0 < b - 1.0:              # top edge cuts through row i
            gap_above = (a - rows[i - 1][1]) if i > 0 else float("inf")
            gap_below = (rows[i + 1][0] - b) if i + 1 < len(rows) else float("inf")
            if gap_above >= gap_below:          # row clings to table below -> keep
                ny0 = (rows[i - 1][1] + a) / 2 if i > 0 else a - 1.0
            else:                               # row clings to block above -> drop
                ny0 = (b + rows[i + 1][0]) / 2 if i + 1 < len(rows) else b + 1.0
            break

    for i, (a, b) in enumerate(rows):
        if a + 1.0 < y1 < b - 1.0:              # bottom edge cuts through row i
            gap_above = (a - rows[i - 1][1]) if i > 0 else float("inf")
            gap_below = (rows[i + 1][0] - b) if i + 1 < len(rows) else float("inf")
            if gap_below >= gap_above:          # row clings to table above -> keep
                ny1 = (b + rows[i + 1][0]) / 2 if i + 1 < len(rows) else b + 1.0
            else:                               # row clings to block below -> drop
                ny1 = (rows[i - 1][1] + a) / 2 if i > 0 else a - 1.0
            break

    return ny0, ny1


def refine_bbox(
    page,
    bbox: Rect,
    ruling_rects: List[Rect],
    H: list,
    V: list,
    pad: float = 2.0,
    snap_iou: float = 0.45,
) -> Tuple[Rect, str]:
    """
    Shrink-only refinement of a candidate bbox.

    - If a ruling rectangle mutually overlaps ``bbox`` well (IoU >= snap_iou),
      the lines truly delineate this table: snap to the ruling rectangle, then
      extend along its row band to reclaim a left label column / right unit
      note that sits outside the ruled area (Tier A/B).  A small header strip
      inside a much larger box has low IoU and is therefore NOT snapped to,
      which prevents collapsing a big table onto one of its header bands.
    - Otherwise treat as wireless: shrink ``bbox`` to its contained text spans
      (Tier C).  The result never grows beyond the input bbox.

    Returns (refined_bbox, source) where source is "line" | "text" | "yolo".
    """
    x0, y0, x1, y1 = bbox
    best: Optional[Rect] = None
    best_iou = 0.0
    for r in ruling_rects:
        iou = _iou(bbox, r)
        if iou > best_iou:
            best_iou = iou
            best = r
    # Chain sibling ruling bands into one snap target (中石油 style): the
    # header band and the value band are ruled separately with a 15-30 pt
    # unruled seam — often holding the first data row — between them.
    # Snapping to a single band amputates the other half, which then
    # resurfaces as a spurious recovered box.  The chain is only taken
    # when its union tracks the detector box BETTER than the single best
    # band, so pages with one band per table keep their behaviour, and
    # sibling stacked tables (>= 40 pt apart) never chain.
    if best is not None:
        inside = sorted((r for r in ruling_rects
                         if _containment(r, bbox) >= 0.8),
                        key=lambda r: r[1])
        if best in inside and len(inside) > 1:
            from assay.splitting import _rows as _srows, _row_kind

            def _seam_bridged(a: Rect, b: Rect) -> bool:
                # A seam <= 32 pt is normal intra-table spacing.  Up to
                # 40 pt it may still be interior when it HOLDS a data row
                # (a tall wrapped-cell row between two ruled bands, 中石油
                # p11 shareholder register); a between-tables seam of that
                # size only ever holds captions / titles, never data.
                gap = b[1] - a[3]
                if gap <= 32.0:
                    return True
                if gap > 40.0:
                    return False
                sx0, sx1 = max(a[0], b[0]), min(a[2], b[2])
                return any(len(r[2]) >= 2 and _row_kind(r[2]) == "data"
                           for r in _srows(page, (sx0, a[3] + 1.0,
                                                  sx1, b[1] - 1.0))
                           if r[0] >= a[3] - 1.0 and r[1] <= b[1] + 1.0)

            i = inside.index(best)
            lo, hi = i, i
            while lo > 0 and _seam_bridged(inside[lo - 1], inside[lo]):
                lo -= 1
            while hi < len(inside) - 1 \
                    and _seam_bridged(inside[hi], inside[hi + 1]):
                hi += 1
            chain = inside[lo:hi + 1]
            if len(chain) > 1:
                u = (min(r[0] for r in chain), chain[0][1],
                     max(r[2] for r in chain), chain[-1][3])
                u_iou = _iou(bbox, u)
                if u_iou > best_iou:
                    best, best_iou = u, u_iou
    if best is not None and best_iou >= snap_iou:
        rx0, ry0, rx1, ry1 = best
        # Ruled-header style (中石油): the drawn band covers ONLY the header
        # while every data row sits unruled below — snapping amputates the
        # whole body.  Reclaim contiguous data rows hanging right under the
        # band (tight spacing, still inside the detector box); the first
        # label row — a note line or the next table's caption — stops the
        # walk.
        if ry1 < y1 - 10.0:
            from assay.splitting import _rows as _srows, _row_kind
            rows_below = [row for row in _srows(page, (rx0, ry1 - 2.0, rx1, y1))
                          if row[0] >= ry1 - 2.0]
            x_mid = rx0 + 0.55 * (rx1 - rx0)
            prev_bot = ry1
            k = 0
            while k < len(rows_below):
                row = rows_below[k]
                if row[0] - prev_bot > 15.0:
                    break
                if len(row[2]) >= 2 and _row_kind(row[2]) == "data":
                    prev_bot = max(prev_bot, row[1])
                    k += 1
                    continue
                # Wrapped-cell label line: a lone span confined to the left
                # label column whose continuation row right below is data
                # (中石油 p155 归属于母公司股东的综合收/益总额).  A note line
                # or the next table's caption reaches past mid-box and
                # still stops the walk.
                if (len(row[2]) == 1 and row[2][0][1] < x_mid
                        and k + 1 < len(rows_below)
                        and rows_below[k + 1][0] - row[1] <= 15.0
                        and len(rows_below[k + 1][2]) >= 2
                        and _row_kind(rows_below[k + 1][2]) == "data"):
                    prev_bot = max(prev_bot, rows_below[k + 1][1])
                    k += 2
                    continue
                break
            ry1 = prev_bot
        ex0, ex1 = _rowband_extent(page, (rx0, ry0, rx1, ry1))
        rx0 = min(rx0, ex0)
        rx1 = max(rx1, ex1)
        return (rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad), "line"

    shr = _content_bbox(page, bbox)
    if shr is None:
        return bbox, "yolo"
    sx0, sy0, sx1, sy1 = shr
    bx0 = max(x0, sx0)
    by0 = max(y0, sy0)
    bx1 = min(x1, sx1)
    by1 = min(y1, sy1)
    # Wireless numeric tables (bank notes) often have the visual detector fire
    # on the number columns only, leaving the left label column outside the
    # box.  Reclaim same-row labels/unit-notes via the row band, mirroring the
    # line-snap branch — bounded by ``reach`` so unrelated text cannot leak in.
    ex0, ex1 = _rowband_extent(page, (bx0, by0, bx1, by1))
    bx0 = min(bx0, ex0)
    bx1 = max(bx1, ex1)
    # Drop a centred title/caption block above the header's top ruling line:
    # the true table top is that line, not the statement title printed above it.
    ct = _caption_top(page, (bx0, by0, bx1, by1), ruling_rects)
    if ct is not None and ct > by0:
        by0 = ct
    # Trim any trailing prose / signature block: the true table bottom is the
    # last row of the aligned value-column grid, not the last text span in the
    # (often over-tall) detector box.
    gb = _value_grid_bottom(page, (bx0, by0, bx1, by1))
    if gb is not None and gb < by1:
        by1 = gb
    # Finally, quantise the top/bottom edges to inter-row whitespace so a
    # boundary never bisects a text line: the detector can place an edge through
    # a spanning column-group header ("本集团"/"本行"), and that header must be
    # kept whole.  The bisected line joins the block it is tighter with; the edge
    # moves into the wider adjacent gap.
    by0, by1 = _snap_edges_to_line_gaps(page, (bx0, by0, bx1, by1))
    return (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad), "text"
