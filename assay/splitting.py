"""
Stacked-table fission: split one refined detection box that spans several
vertically stacked tables into per-table boxes.

Financial reports stack sibling tables (本集团 vs 本行, 2024年度 vs 2023年度,
quarterly repeats) with little whitespace between them, so the visual
detector + shrink refinement often yields ONE box covering them all.  The
ground-truth rule is "a new table header starts a new table"; this module
implements exactly that rule geometrically via a **header-mirror test**:

  A box is split at a *header restart* -- a text row that structurally
  mirrors an earlier header row of the same box (similar column count, same
  x-grid, similar label tokens).  Data rows can never mirror a header (they
  are numeric), prose/section lines never reproduce the column grid, so a
  genuine single table is never cut.  No hard-coded vocabulary: the box's
  own first header is the template, which generalises across industries.

Safety invariants:
  - vertical cuts only, placed on a full-width ruling line or in the
    whitespace gap between two text rows -- never through a text line;
  - grid-continuity veto: a cut crossed by a drawn vertical ruling line is
    slicing through ONE continuous bordered grid (a repeated in-table header
    row is a section switch, not a new table) and is rejected;
  - all-or-nothing: if ANY resulting segment fails validation (each must
    contain a multi-span header row and a data row), the whole split is
    abandoned and the original box is returned unchanged.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

Rect = Tuple[float, float, float, float]

# ── Tunables ────────────────────────────────────────────────────────────────
MIN_SPLIT_HEIGHT = 60.0   # don't even scan boxes shorter than this (pt)
MIN_HEADER_SPANS = 2      # a header/restart row needs at least this many cells
X_ALIGN_TOL = 12.0        # max center distance for two spans to share a column
ALIGN_RATIO = 0.6         # min fraction of columns that must x-align
TOKEN_RATIO = 0.5         # min fraction of aligned columns with similar labels
CHAR_JACCARD = 0.4        # per-column label similarity (character Jaccard)
CUT_LINE_MIN_W = 0.6      # a cut ruling line must span >= this frac of box width
HEADER_ROW_MAX_W = 0.75   # header-block walk-up rows must be narrower than
                          # this frac of box width (excludes full-width prose)
EDGE_PAD = 2.0            # padding applied to tightened segment bboxes (pt)

# Scope-label absorption: a short centred label just above the box top
# (本集团 / 本行 / 本集团及本行 ...) belongs to the table but is often left
# outside the detector box, and shrink-only refinement can never recover it.
SCOPE_GAP_MAX = 14.0      # max gap between label bottom and box top (pt)
SCOPE_ISOLATION = 6.0     # min whitespace required above the label row (pt)
SCOPE_MAX_W = 0.30        # max label row width as fraction of box width
SCOPE_CENTER_LO = 0.40    # label row centre must sit over the data columns:
SCOPE_CENTER_HI = 0.92    # centre ratio within the box, excludes left-aligned
                          # section headings and right-edge artifacts

# A span is a numeric value cell: digits with grouping/sign/percent only.
# The charset includes 、 (ideographic comma): Chinese financial tables
# list multiple rates/amounts in one cell as "1%、3%、5%".
_NUMERIC_RE = re.compile(r'^[\d,，、.．\-−–—%％()（）\s/:：]*\d'
                         r'[\d,，、.．\-−–—%％()（）\s/:：]*$')
_DASH_RE = re.compile(r'^[-−–—]+$')
_ENUM_RE = re.compile(r'^[（(]?\d{1,3}[.、)）]$')  # "24." / "(3)" / "（2）"
_YEAR_RE = re.compile(r'(19|20)\d{2}')
_PERIOD_CHAR_RE = re.compile(r'[年月日]')


def _is_numeric_token(t: str) -> bool:
    t = t.strip()
    if not t or _ENUM_RE.match(t):
        return False
    return bool(_NUMERIC_RE.match(t)) or bool(_DASH_RE.match(t))


# ── Row extraction (same overlap-merge as rulings._text_rows, keeps spans) ──


def _rows(page, box: Rect) -> List[list]:
    """Text rows inside ``box`` as [y0, y1, [(x0, x1, text), ...]], sorted."""
    x0, y0, x1, y1 = box
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            cx = (lx0 + lx1) / 2
            cy = (ly0 + ly1) / 2
            if not (x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1):
                continue
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if not txt:
                continue
            lines.append((ly0, ly1, lx0, lx1, txt))
    lines.sort()
    rows: List[list] = []
    for ly0, ly1, lx0, lx1, txt in lines:
        # vertical overlap => same row.  重叠量按行高相对化：绝对 2pt
        # 是 12pt 字排版下的 0.2 行高，平安 10pt 字的行距更紧——基线
        # 错位仅 0.9-2.1pt 的相邻数据行（标签行与数值行）会被它并成
        # 一个带，三个逻辑行吸进两个带（一格两数的根源）
        if rows:
            overlap = rows[-1][1] - ly0
            avg_h = 0.5 * ((rows[-1][1] - rows[-1][0]) + (ly1 - ly0))
            if overlap >= 0.2 * avg_h:
                rows[-1][0] = min(rows[-1][0], ly0)
                rows[-1][1] = max(rows[-1][1], ly1)
                rows[-1][2].append((lx0, lx1, txt))
                continue
        rows.append([ly0, ly1, [(lx0, lx1, txt)]])
    for r in rows:
        r[2].sort()
    return rows


# ── Borderless text-table test ────────────────────────────────────────────────────────

TEXT_TABLE_MIN_ROWS = 4    # a text-column table shows at least this many rows
TEXT_TABLE_GUTTER = 8.0    # min vertical whitespace channel width (pt)
TEXT_TABLE_RATIO = 0.7     # min fraction of multi-span rows sharing the gutter

# Row-level vetoes: structures that share the column-gutter geometry but are
# not tables.  A numbered clause list opens with an enumerator ("1." / "(i)" /
# "（一）") or a bullet glyph, a filing-info panel keys every row with a
# full-width-colon label ("公司注册中文名称："), a table of contents ends every
# row with a bare page number or page range ("6 - 7"), and a bilingual
# letterhead (auditor address block) opens every row with pure-Latin text.
# Financial value cells always carry grouping commas or decimals, so the
# bare-number test never hits a real one; Chinese report tables key their
# rows with CJK labels, so the Latin-head test never hits a real one either.
_ENUM_HEAD_RES = (
    _ENUM_RE,                                       # "24." / "(3)" / "（2）"
    re.compile(r'^[（(]?\d{1,3}[．]$'),              # "1．" fullwidth stop
    re.compile(r'^\(?[ivx]{1,4}[.、)）]$'),          # "(i)" / "iv."
    re.compile(r'^[（(]?[一二三四五六七八九十]{1,3}[)）、.．]$'),  # "（一）" / "二、"
    re.compile(r'^[•·●▪◦‣■○–─-]$'),                 # bullet glyphs
    re.compile(r'^[①-⑳㉑-㉟]$'),               # "①" circled enumerators
)
_PAGENO_RE = re.compile(r'^\d{1,3}(\s*[-–—]\s*\d{1,3})?$')
# Fullwidth enumerators glue to their caption with no space ("（2）收到…"),
# so a whole-token match can never see them; this prefix form is used only
# by the section-fission heading test, which is further guarded by the
# no-numeric / left-confined / whitespace-float requirements.
_ENUM_PREFIX_RE = re.compile(r'^([（(]\d{1,3}[)）]|[①-⑳㉑-㉟])\s*\S')
_DOTLEAD_RE = re.compile(r'[.．·…]{4,}')   # TOC dot leader "公司治理......"
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


def _colrow_ok(spans, allow_numeric: bool = True) -> bool:
    """True when a multi-span row is believable as a text-table row."""
    head = spans[0][2].strip()
    if any(rx.match(head) for rx in _ENUM_HEAD_RES):
        return False
    if head.endswith('：') or head.endswith(':'):
        return False
    if not _CJK_RE.search(head) and not _is_numeric_token(head):
        return False
    if _PAGENO_RE.match(spans[-1][2].strip()):
        return False
    if any(_DOTLEAD_RE.search(t) for _, _, t in spans):
        return False
    if not allow_numeric and any(_is_numeric_token(t) for _, _, t in spans):
        return False
    return True


def _justified_prose(rows_spans, chan) -> bool:
    """
    True when the channel-confirming rows read as justified two-column
    prose (关键审计事项 double-column layout): justified text fills its
    column to a flush right margin, so the ink edge left of the channel is
    constant and the column is wide.  A table label cell ends where its
    text ends, so across rows the edge wanders and stays narrow.
    """
    lo, hi = chan
    edges = []
    for sp in rows_spans:
        for a, b in zip(sp, sp[1:]):
            if b[0] - a[1] >= TEXT_TABLE_GUTTER \
                    and min(hi, b[0]) - max(lo, a[1]) >= 0.75 * TEXT_TABLE_GUTTER:
                edges.append((a[1], a[1] - a[0]))
                break
    if len(edges) < TEXT_TABLE_MIN_ROWS:
        return False
    xs = sorted(e[0] for e in edges)
    med = xs[len(xs) // 2]
    flush = [e for e in edges if abs(e[0] - med) <= 2.5]
    wide = sum(1 for e in flush if e[1] >= 100.0)
    # 0.6 not 0.7: paragraph-final lines fall short of the flush margin, so
    # justified prose never reaches full agreement.  Genuine tables are safe
    # at 0.6 because their label cells stay narrow (< 100 pt) — the width
    # test, not the flush count, is what clears them.
    return len(flush) >= 0.6 * len(edges) and wide >= 0.7 * len(flush)


def looks_text_table(page, box) -> bool:
    """
    True when ``box`` holds a borderless all-text table: many text rows whose
    spans share one vertical whitespace channel (a column gutter).

    Related-party registers, importance-criteria and entity-list disclosures
    are set as label/label tables with no rulings and no digits, so every
    numeric heuristic and every ruling detector is blind to them.  What no
    prose page reproduces is their geometry: prose lines run full-width as a
    single span, while a text table breaks each row at the same x-band.  The
    test needs TEXT_TABLE_MIN_ROWS such rows AND the channel shared by most
    multi-span rows, so an indented list or a chance line-wrap never passes.
    Rows vetoed by _colrow_ok (clause enumerators, colon-keyed info panels,
    page-number contents lines) share the geometry but not the nature, and
    are excluded up front.
    """
    rows = [r[2] for r in _rows(page, box)
            if len(r[2]) >= 2 and _colrow_ok(r[2])]
    if len(rows) < TEXT_TABLE_MIN_ROWS:
        return False
    seeds = [(a[1], b[0]) for sp in rows for a, b in zip(sp, sp[1:])
             if b[0] - a[1] >= TEXT_TABLE_GUTTER]
    best, best_chan = 0, None
    for lo, hi in seeds:
        n = sum(
            1 for sp in rows
            if any(b[0] - a[1] >= TEXT_TABLE_GUTTER
                   and min(hi, b[0]) - max(lo, a[1]) >= 0.75 * TEXT_TABLE_GUTTER
                   for a, b in zip(sp, sp[1:]))
        )
        if n > best:
            best, best_chan = n, (lo, hi)
    if best < TEXT_TABLE_MIN_ROWS or best < TEXT_TABLE_RATIO * len(rows):
        return False
    return not _justified_prose(rows, best_chan)


TEXT_TABLE_ROW_GAP = 30.0  # max vertical gap between rows of one text table


def recover_text_tables(page, allow_numeric: bool = True) -> List[Rect]:
    """
    Page-level recovery of borderless all-text tables the visual detector
    never boxed: maximal runs of text rows sharing one column-gutter channel.

    Walks the page's text rows top-down, growing a run while the vertical
    whitespace channel stays clean.  Rows with a matching inter-span gap
    confirm the channel; a row without one may still belong to the table
    (an in-table section label — 合营企业 — or a lone column header —
    处置日) as long as none of its ink crosses the channel, so it is held
    pending and committed when a later row confirms the channel again.  A
    row whose ink crosses the channel (full-width prose) breaks the run.
    A run of at least TEXT_TABLE_MIN_ROWS channel-confirming rows becomes a
    candidate rectangle.  When a run starts, the row just above joins it if
    it sits entirely right of the new channel (a lone column header — 处置日
    — hangs over the value columns); a left-margin section title never
    qualifies.  Prose cannot sustain a clean shared channel, so runs only
    form over genuine columnar text.  Downstream gates (size filter, dedup
    against visual / ruling candidates, PP-LCNet) still apply to every
    recovered rectangle.

    ``allow_numeric=False`` restricts channel-confirming rows to all-text
    ones (the pure label/label table hunt on digit-free pages);
    ``allow_numeric=True`` also accepts numeric value rows (rescuing a
    digit table on a page where every other detector came up empty).  Rows
    failing _colrow_ok never confirm a channel in either mode.
    """

    def _ivals(sp):
        return [(a[1], b[0]) for a, b in zip(sp, sp[1:])
                if b[0] - a[1] >= TEXT_TABLE_GUTTER]

    MIN_W = 0.75 * TEXT_TABLE_GUTTER

    rows = _rows(page, (page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1))
    out: List[Rect] = []
    run: List[list] = []    # committed rows (channel-confirming + interior)
    n_conf = 0              # rows in ``run`` that confirmed the channel
    pend: List[list] = []   # channel-neutral rows awaiting confirmation
    chans: List[Tuple[float, float]] = []

    def _flush():
        nonlocal run, n_conf, pend
        if n_conf >= TEXT_TABLE_MIN_ROWS and not (
                chans and _justified_prose([r[2] for r in run], chans[0])):
            x0 = min(s[0] for r in run for s in r[2])
            x1 = max(s[1] for r in run for s in r[2])
            out.append((x0, run[0][0], x1, run[-1][1]))
        run, n_conf, pend = [], 0, []

    TOL_ROW = 2.0
    prev = None
    for r in rows:
        ivals = _ivals(r[2]) if _colrow_ok(r[2], allow_numeric) else []
        prev_bottom = (pend[-1][1] if pend else run[-1][1]) if run else None
        in_gap = run and r[0] - prev_bottom <= TEXT_TABLE_ROW_GAP
        if in_gap:
            if ivals:
                nxt = [(max(c[0], lo), min(c[1], hi))
                       for c in chans for lo, hi in ivals
                       if min(c[1], hi) - max(c[0], lo) >= MIN_W]
                if nxt:
                    chans = sorted(set(nxt))[:6]
                    run.extend(pend)
                    pend = []
                    run.append(r)
                    n_conf += 1
                    prev = r
                    continue
            else:
                # channel-neutral only when no ink crosses ANY live channel
                clean = [c for c in chans
                         if not any(s[0] < c[1] - MIN_W and s[1] > c[0] + MIN_W
                                    for s in r[2])]
                if clean:
                    chans = clean
                    pend.append(r)
                    prev = r
                    continue
        _flush()
        if ivals:
            # a continuation line of a colon-keyed info panel ("职工代表
            # 监事： 李战明 金彦江" wrapping extra names onto the next line)
            # shares the panel's value columns — never seed a run from it
            if prev is not None and len(prev[2]) >= 2 \
                    and r[0] - prev[1] <= TEXT_TABLE_ROW_GAP \
                    and (prev[2][0][2].strip().endswith('：')
                         or prev[2][0][2].strip().endswith(':')):
                run, n_conf, chans = [], 0, []
                prev = r
                continue
            run, n_conf, chans = [r], 1, sorted(ivals)[:6]
            # a lone column header hanging over the value columns just
            # above the first body row belongs to the table
            if prev is not None and r[0] - prev[1] <= TEXT_TABLE_ROW_GAP \
                    and not _ivals(prev[2]) \
                    and all(s[0] >= chans[0][0] - TOL_ROW for s in prev[2]):
                run.insert(0, prev)
        else:
            run, n_conf, chans = [], 0, []
        prev = r
    _flush()
    return out


def _row_kind(spans) -> str:
    """'data' if the row carries numeric value cells, else 'label'."""
    n_num = sum(1 for _, _, t in spans if _is_numeric_token(t))
    if n_num >= 2 or (n_num == len(spans) and n_num >= 1):
        return "data"
    if n_num == 0:
        return "label"
    return "data"   # mixed label+single number rows count as data


# ── Header-mirror test ──────────────────────────────────────────────────────


def _mirrors(header_spans, cand_spans) -> bool:
    """
    True when ``cand_spans`` structurally reproduces ``header_spans``:
    similar span count, columns x-aligned, and aligned columns carry
    similar label text (character Jaccard) or are both period tokens
    (e.g. "2024年12月31日" vs "2023年").  Reads structure, not vocabulary.
    """
    a, b = header_spans, cand_spans
    if abs(len(a) - len(b)) > 1:
        return False
    matched = []
    used = set()
    for bx0, bx1, bt in b:
        for ai, (ax0, ax1, at) in enumerate(a):
            if ai in used:
                continue
            ov = min(ax1, bx1) - max(ax0, bx0)
            close = abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) <= X_ALIGN_TOL
            if ov > 0.5 * min(ax1 - ax0, bx1 - bx0) or close:
                matched.append((at, bt))
                used.add(ai)
                break
    if len(matched) < max(2, ALIGN_RATIO * max(len(a), len(b))):
        return False
    ok = 0
    for at, bt in matched:
        sa = set(at.replace(" ", ""))
        sb = set(bt.replace(" ", ""))
        jac = len(sa & sb) / max(1, len(sa | sb))
        if jac >= CHAR_JACCARD or (_YEAR_RE.search(at) and _YEAR_RE.search(bt)):
            ok += 1
    return ok >= max(2, TOKEN_RATIO * len(matched))


# ── Cut placement ───────────────────────────────────────────────────────────


def _cut_y(gap_lo: float, gap_hi: float, box: Rect, H: list) -> Optional[float]:
    """
    Cut coordinate inside the whitespace gap (gap_lo, gap_hi): prefer a
    full-width ruling line (shared-border grids draw one there), else the
    gap midpoint.  None when the rows touch (no legal cut exists).
    """
    if gap_hi <= gap_lo + 1.0:
        return None
    x0, _, x1, _ = box
    boxw = max(1.0, x1 - x0)
    lines = [hy for hx0, hx1, hy in H
             if gap_lo + 0.3 < hy < gap_hi - 0.3
             and (min(hx1, x1) - max(hx0, x0)) >= CUT_LINE_MIN_W * boxw]
    if lines:
        mid = (gap_lo + gap_hi) / 2
        return min(lines, key=lambda y: abs(y - mid))
    return (gap_lo + gap_hi) / 2


# ── Scope-label absorption ───────────────────────────────────────────────


def absorb_scope_label(page, box: Rect) -> Rect:
    """
    Extend ``box`` upward to include a scope label (本集团 / 本集团及本行 …)
    sitting just above its top edge.  Requirements: single short text-only
    row, centred over the data-column band, isolated by whitespace above.
    Left-aligned section headings and full-width prose never qualify.
    """
    x0, y0, x1, y1 = box
    boxw = max(1.0, x1 - x0)
    band: List[Tuple[float, float, float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            txt = "".join(s["text"] for s in line["spans"]).strip()
            # Membership by the row's vertical CENTRE, not its bottom: refine's
            # line-snap can drop the top edge onto a ruling that bisects the
            # group-header row (本集团 / 2023年12月31日), leaving its bottom a
            # point or two below the edge.  Such a straddling header still
            # belongs above the table; a "ly1 <= top" test would miss it while a
            # centre test still excludes the real column-label row that sits
            # wholly inside the box just below.
            lcy = (ly0 + ly1) / 2
            if txt and lcy <= y0 + 1.0 and lcy >= y0 - SCOPE_GAP_MAX \
                    and lx1 > x0 and lx0 < x1:
                band.append((ly0, ly1, lx0, lx1, txt))
    if len(band) != 1:            # nothing there, or a multi-line block
        return box
    ly0, ly1, lx0, lx1, txt = band[0]
    center = ((lx0 + lx1) / 2 - x0) / boxw
    # A fully parenthesised line is a typographic note (（货币单位：…）),
    # not part of the table per the annotation standard — never absorb.
    parenthesised = txt[0] in "（(" and txt[-1] in "）)"
    # No digit veto: a period header (2023年12月31日) is a legitimate
    # value-column group header and carries digits.  A genuine DATA row is
    # excluded structurally instead -- it spans the label column and so is
    # wider than SCOPE_MAX_W; only a narrow line floating over the value band
    # (center gate) and isolated by whitespace above survives here.
    if (lx1 - lx0) > SCOPE_MAX_W * boxw \
            or not (SCOPE_CENTER_LO <= center <= SCOPE_CENTER_HI) \
            or parenthesised:
        return box
    # isolation: no *inked* text line ends within SCOPE_ISOLATION above the
    # label.  Empty-text lines ('' spans -- leader-line / cell-border artifacts
    # PyMuPDF emits as zero-ink lines, typically a ~3pt sliver in the label
    # column) carry no content and must not count as an isolating line, exactly
    # as the band collection above already ignores them.
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            bx0, by0, bx1, by1 = line["bbox"]
            btxt = "".join(s["text"] for s in line["spans"]).strip()
            if btxt and by1 <= ly0 + 0.5 and by1 > ly0 - SCOPE_ISOLATION \
                    and bx1 > x0 and bx0 < x1:
                return box
    return (x0, ly0 - 1.0, x1, y1)


# ── Period-band fission (lone period rows sharing one column grid) ───────────


def _is_lone_period(spans) -> bool:
    """A single-span row that is only a period label (e.g. "2024年度",
    "2024年12月31日") — a sibling table that borrows its neighbour's columns
    rather than repeating the multi-span column header."""
    if len(spans) != 1:
        return False
    t = spans[0][2]
    return bool(_YEAR_RE.search(t)) and bool(_PERIOD_CHAR_RE.search(t))


def split_by_period_bands(
    page, box: Rect, H: list, ruling_rects: list
) -> List[Rect]:
    """
    Split a box that stacks sibling statements sharing ONE column grid — an
    income statement ("2024年度") above a balance sheet ("2024年12月31日"),
    each introduced by a *lone period row* (single span, no column header of
    its own) and each backed by its own drawn ruling band.

    ``split_stacked_tables`` cannot handle this: the restart row is single-span
    (fails the multi-span header test) and the lower segment has no multi-span
    header to validate.  This is a deliberately narrow, double-corroborated
    rule that never fires on a single table:
      * requires >= 2 lone period rows with data between them (a new period
        header = a new table, per the annotation standard);
      * every cut must fall in the whitespace gap between TWO disjoint ruling
        bands (each resulting table is backed by its own drawn grid), so a
        single table with a header underline + a total underline — one period
        row — can never be cut.
    Returns [box] unchanged unless every resulting segment validates.
    """
    x0, y0, x1, y1 = box
    if y1 - y0 < MIN_SPLIT_HEIGHT:
        return [box]
    boxw = max(1.0, x1 - x0)
    # Disjoint ruling bands overlapping the box in x (y-sorted).  A band may
    # stick out past a YOLO box that under-reached and got shrunk mid-row, so
    # match on y-overlap, not strict containment.
    bands = sorted(
        (r for r in ruling_rects
         if r[3] > y0 and r[1] < y1
         and (min(r[2], x1) - max(r[0], x0)) >= 0.3 * boxw),
        key=lambda r: r[1])
    if len(bands) < 2:
        return [box]
    gaps = [(bands[k][3], bands[k + 1][1]) for k in range(len(bands) - 1)
            if bands[k + 1][1] - bands[k][3] > 1.0]
    if not gaps:
        return [box]

    # Extend the working box to span the bands so a row the YOLO shrink clipped
    # (a value row whose grid line the box bottom cut through) is recovered.
    wy0, wy1 = min(y0, bands[0][1]), max(y1, bands[-1][3])
    rows = _rows(page, (x0, wy0, x1, wy1))
    if len(rows) < 4:
        return [box]
    kinds = [_row_kind(r[2]) for r in rows]
    period_at = [i for i, r in enumerate(rows) if _is_lone_period(r[2])]
    if len(period_at) < 2:
        return [box]

    cuts: List[float] = []
    for pi in period_at[1:]:                     # never cut above the first
        if pi == 0 or not any(kinds[j] == "data" for j in range(pi)):
            continue                              # need data above the restart
        cut = _cut_y(rows[pi - 1][1], rows[pi][0], box, H)
        if cut is None:
            continue
        if not any(lo < cut < hi for lo, hi in gaps):  # must sit in a band gap
            continue
        cuts.append(cut)
    if not cuts:
        return [box]

    bounds = [wy0] + cuts + [wy1]
    segments: List[Rect] = []
    for lo, hi in zip(bounds, bounds[1:]):
        seg_rows = [r for r in rows if r[0] >= lo - 0.5 and r[1] <= hi + 0.5]
        if len(seg_rows) < 2:
            return [box]
        if not any(_row_kind(r[2]) == "data" for r in seg_rows):
            return [box]
        sy0 = min(r[0] for r in seg_rows)
        sy1 = max(r[1] for r in seg_rows)
        sx0 = min(s[0] for r in seg_rows for s in r[2])
        sx1 = max(s[1] for r in seg_rows for s in r[2])
        segments.append((sx0 - EDGE_PAD, sy0 - EDGE_PAD,
                         sx1 + EDGE_PAD, sy1 + EDGE_PAD))
    return segments


# ── Public API ──────────────────────────────────────────────────────────────


# ── Section-heading fission ─────────────────────────────────────────────────

SECTION_GAP = 8.0    # min whitespace above AND below a section-heading row (pt)


def _section_fission(page, box: Rect, H: list, V: list = ()) -> List[Rect]:
    """
    Split ``box`` at enumerated SECTION headings ("3. 套期" / "64.每股收益" /
    "五、…") that a blanket detection swallowed together with the tables of
    several note sections.  A qualifying heading row is left-confined (never
    reaches the right half — table headers always do), opens with a list
    enumerator, carries no numeric cell, and floats in whitespace
    (>= ``SECTION_GAP`` above and below); interior table rows never combine
    all of these.  Segments are tightened to their rows; a segment with no
    table evidence (no data row and < 3 multi-span rows) is a section
    preamble / stray continuation band and is dropped, but a failing segment
    that carries data aborts the whole fission (never deletes table
    content).  Returns ``[box]`` unchanged when nothing qualifies.
    """
    x0, y0, x1, y1 = box
    boxw = max(1.0, x1 - x0)
    x_mid = x0 + 0.55 * boxw
    rows = _rows(page, box)
    if len(rows) < 3:
        return [box]

    def _grid_crossed(cut: float) -> bool:
        return any(vy0 < cut - 2.0 and vy1 > cut + 2.0
                   and x0 - 1.0 <= vx <= x1 + 1.0
                   for vy0, vy1, vx in V)

    def _is_heading(r) -> bool:
        spans = r[2]
        if not spans or len(spans) > 3:
            return False
        if max(s[1] for s in spans) > x_mid:
            return False
        if any(_is_numeric_token(t) for _, _, t in spans):
            return False
        head = spans[0][2].strip().split()[0] if spans[0][2].strip() else ""
        if any(rx.match(head) for rx in _ENUM_HEAD_RES):
            return True
        return bool(_ENUM_PREFIX_RE.match(spans[0][2].strip()))

    cuts: List[float] = []
    for i, r in enumerate(rows):
        if i == 0 or not _is_heading(r):
            continue
        gap_above = r[0] - rows[i - 1][1]
        gap_below = rows[i + 1][0] - r[1] if i + 1 < len(rows) else 0.0
        if gap_above < SECTION_GAP or gap_below < SECTION_GAP:
            continue
        # Table evidence below: a multi-span row reaching the right half.
        if not any(len(rr[2]) >= MIN_HEADER_SPANS
                   and max(s[1] for s in rr[2]) > x_mid
                   for rr in rows[i + 1:]):
            continue
        cut = _cut_y(rows[i - 1][1], r[0], box, H)
        if cut is None or _grid_crossed(cut):
            continue
        cuts.append(cut)
    if not cuts:
        return [box]

    def _left_only(row) -> bool:
        return all((s[0] + s[1]) / 2 < x_mid for s in row[2])

    keepers: List[Rect] = []
    bounds = [y0] + cuts + [y1]
    for lo, hi in zip(bounds, bounds[1:]):
        seg_rows = [r for r in rows if r[0] >= lo - 0.5 and r[1] <= hi + 0.5]
        while seg_rows and _row_kind(seg_rows[0][2]) == "label" \
                and _left_only(seg_rows[0]):
            seg_rows.pop(0)
        while seg_rows and _row_kind(seg_rows[-1][2]) == "label" \
                and _left_only(seg_rows[-1]):
            seg_rows.pop()
        multi = [r for r in seg_rows if len(r[2]) >= MIN_HEADER_SPANS]
        has_data = any(_row_kind(r[2]) == "data" for r in seg_rows)
        has_header = any(_row_kind(r[2]) == "label" for r in multi)
        reaches = any(max(s[1] for s in r[2]) > x_mid for r in multi)
        if not (len(seg_rows) >= 2 and has_header and reaches
                and (has_data or len(multi) >= 3)):
            if has_data:            # would delete table content -> abort
                return [box]
            continue                # preamble / stray band -> drop
        sy0 = min(r[0] for r in seg_rows)
        sy1 = max(r[1] for r in seg_rows)
        sx0 = min(s[0] for r in seg_rows for s in r[2])
        sx1 = max(s[1] for r in seg_rows for s in r[2])
        keepers.append((sx0 - EDGE_PAD, sy0 - EDGE_PAD,
                        sx1 + EDGE_PAD, sy1 + EDGE_PAD))
    return keepers if keepers else [box]


def split_stacked_tables(page, box: Rect, H: list, V: list = ()) -> List[Rect]:
    """
    Split ``box`` at header restarts.  Returns [box] unchanged when no valid
    restart is found or any resulting segment fails validation (all-or-
    nothing).  ``H`` / ``V`` are the page's horizontal / vertical ruling
    segments.
    """
    x0, y0, x1, y1 = box
    if y1 - y0 < MIN_SPLIT_HEIGHT:
        return [box]
    # Section-heading fission first: a blanket over several note sections is
    # cut apart (and its preamble discarded) before the header-restart walk
    # runs on each piece independently.
    pieces = _section_fission(page, box, H, V)
    if pieces != [box]:
        out: List[Rect] = []
        for p in pieces:
            out.extend(split_stacked_tables(page, p, H, V))
        return out
    boxw = max(1.0, x1 - x0)
    rows = _rows(page, box)
    if len(rows) < 4:
        return [box]
    kinds = [_row_kind(r[2]) for r in rows]

    def _grid_crossed(cut: float) -> bool:
        """A drawn vertical line running through the cut = one continuous
        bordered grid; cutting it would shred a genuine single table."""
        return any(vy0 < cut - 2.0 and vy1 > cut + 2.0
                   and x0 - 1.0 <= vx <= x1 + 1.0
                   for vy0, vy1, vx in V)

    # Walk rows tracking the current segment's header template.  Two restart
    # signatures, both requiring data rows above:
    #   1. header mirror -- the row structurally repeats an earlier header of
    #      this box (sibling tables with identical layout);
    #   2. period-header row -- every span is a period token (contains a
    #      year or 年/月/日, e.g. "2024年1月1日 | 本年增加额 | 本年支付额 |
    #      2024年12月31日"): a *different* table's period header.  Guarded by
    #      a grid-continuity veto: when the data rows below it keep exactly
    #      the column grid of the data rows above, the row is a period
    #      *section switch inside one table* (损益段 -> 规模段) and must not
    #      cut; a genuine new table brings its own, different grid.
    #      No interior table row (section label, subtotal) looks like either.
    cuts: List[float] = []
    header_rows: List[list] = []   # header-template span lists of current segment
    seen_data = False
    hdr_top = None                 # top y of the first template row
    hdr_bottom = None              # bottom y of the last template row
    x_mid = x0 + 0.55 * boxw       # a real header reaches past this; section
                                   # headings / note lines live left of it

    def _full_rule(y_a: float, y_b: float) -> bool:
        return any(y_a < hy < y_b
                   and min(hx1, x1) - max(hx0, x0) >= 0.8 * boxw
                   for hx0, hx1, hy in H)

    def _band_sealed(y_a: float, y_b: float) -> bool:
        """The header template lives inside a CLOSED ruled band (full-width
        rules above its first row and below its last row) and a rule
        separates it from the row at ``y_b``.  A label row below the seal
        is body ink (a multi-line cell of the first entry), NOT header
        template — letting it in makes every later entry "mirror" it and
        shreds the table into per-row segments (中石油 bond register).  A
        page-header underline alone must not seal: it has no rule above
        it, so the band is open and the template keeps growing."""
        if hdr_top is None or not _full_rule(hdr_top - 8.0, hdr_top + 2.0):
            return False
        return _full_rule(y_a - 1.0, y_b + 1.0)

    def _same_grid(i: int) -> bool:
        """Data rows straddling row ``i`` share one column grid (count + x)."""
        above = next((rows[j][2] for j in range(i - 1, -1, -1)
                      if kinds[j] == "data"), None)
        below = next((rows[j][2] for j in range(i + 1, len(rows))
                      if kinds[j] == "data"), None)
        if above is None or below is None or len(above) != len(below):
            return False
        aligned = sum(
            1 for (ax0, ax1, _), (bx0, bx1, _) in zip(above, below)
            if abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) <= 15.0
            or min(ax1, bx1) - max(ax0, bx0) > 0)
        return aligned >= 0.8 * len(above)

    i = 0
    while i < len(rows):
        spans = rows[i][2]
        reaches_right = any((s[0] + s[1]) / 2 >= x_mid for s in spans)
        if kinds[i] == "label" and len(spans) >= MIN_HEADER_SPANS \
                and reaches_right:
            all_period = (
                any(_YEAR_RE.search(t) for _, _, t in spans) and
                all(_YEAR_RE.search(t) or _PERIOD_CHAR_RE.search(t)
                    for _, _, t in spans))
            is_restart = seen_data and (
                any(_mirrors(h, spans) for h in header_rows) or
                (all_period and not _same_grid(i)))
            if is_restart:
                # Restart: walk up over the new table's contiguous header
                # block (narrow super-headers like "本集团及本行"); full-width
                # prose lines stay with the segment above.
                hb = i
                while (hb - 1 > 0 and kinds[hb - 1] == "label" and
                       (rows[hb - 1][2][-1][1] - rows[hb - 1][2][0][0])
                       <= HEADER_ROW_MAX_W * boxw):
                    hb -= 1
                cut = _cut_y(rows[hb - 1][1], rows[hb][0], box, H)
                if cut is not None and not _grid_crossed(cut):
                    cuts.append(cut)
                    header_rows = [spans]
                    hdr_top = rows[i][0]
                    hdr_bottom = rows[i][1]
                    seen_data = False
                    i += 1
                    continue
            if not seen_data and not (
                    header_rows and hdr_bottom is not None
                    and _band_sealed(hdr_bottom, rows[i][0])):
                header_rows.append(spans)
                if hdr_top is None:
                    hdr_top = rows[i][0]
                hdr_bottom = rows[i][1]
        elif kinds[i] == "data":
            seen_data = True
        i += 1

    if not cuts:
        return [box]

    # Build segments and validate every one: a real table has a multi-span
    # header row AND a data row.  Any failure aborts the whole split.
    # Section/enumeration headings and note lines that ride along at a
    # segment's edge live entirely in the left half of the box (a table
    # header always has cells reaching the right side), so they are trimmed
    # from the segment's leading/trailing label rows before tightening.
    def _left_only(row) -> bool:
        return all((s[0] + s[1]) / 2 < x_mid for s in row[2])

    bounds = [y0] + cuts + [y1]
    segments: List[Rect] = []
    for lo, hi in zip(bounds, bounds[1:]):
        seg_rows = [r for r in rows if r[0] >= lo - 0.5 and r[1] <= hi + 0.5]
        while seg_rows and _row_kind(seg_rows[0][2]) == "label" \
                and _left_only(seg_rows[0]):
            seg_rows.pop(0)
        while seg_rows and _row_kind(seg_rows[-1][2]) == "label" \
                and _left_only(seg_rows[-1]):
            seg_rows.pop()
        if len(seg_rows) < 2:
            return [box]
        has_header = any(
            _row_kind(r[2]) == "label" and len(r[2]) >= MIN_HEADER_SPANS
            for r in seg_rows)
        has_data = any(_row_kind(r[2]) == "data" for r in seg_rows)
        if not (has_header and has_data):
            return [box]
        sy0 = min(r[0] for r in seg_rows)
        sy1 = max(r[1] for r in seg_rows)
        sx0 = min(s[0] for r in seg_rows for s in r[2])
        sx1 = max(s[1] for r in seg_rows for s in r[2])
        segments.append((sx0 - EDGE_PAD, sy0 - EDGE_PAD,
                         sx1 + EDGE_PAD, sy1 + EDGE_PAD))
    return segments


# ── Grid-band fission ─────────────────────────────────────────────────────

GB_MIN_V = 2          # a grid band needs at least this many vertical rules
GB_MIN_VLEN = 6.0     # min vertical-rule overlap with the box (pt); shorter is dust
GB_Y_TOUCH = 3.0      # y-interval touch tolerance when clustering bands (pt)
GB_X_TOL = 2.5        # column-skeleton x-match tolerance across a gap (pt)
GB_SAME = 0.7         # skeleton match ratio (both directions) = same grid
GB_MIN_GAP = 4.0      # a hard gap is at least this much whitespace (pt);
                      # adjoining header rows (touching rule clusters) never qualify
GB_HDR_MAX_H = 34.0   # a label-only upper band at most this tall can be a
                      # floating header of the lower band, not a sibling table
GB_PAD = 2.0          # segment padding around band extents (pt)


def _v_bands(box: Rect, V: list) -> List[tuple]:
    """
    Vertical-rule y-extent clusters ("grid bands") overlapping ``box``.

    A vertical rule crossing a y-gap is proof of grid continuity, so
    clustering by y-interval overlap (with a small touch tolerance) keeps
    one continuous ruled table in a single band even when its verticals
    are drawn as per-segment fragments; only a genuine break between two
    closed grids produces two bands.  Returns [(y0, y1, [x, ...]), ...].
    """
    vs = [(vy0, vy1, vx) for vy0, vy1, vx in V
          if box[0] - 3.0 <= vx <= box[2] + 3.0
          and min(vy1, box[3]) - max(vy0, box[1]) >= GB_MIN_VLEN]
    ivs = sorted((vy0, vy1) for vy0, vy1, _ in vs)
    bands: List[list] = []
    for y0, y1 in ivs:
        if bands and y0 <= bands[-1][1] + GB_Y_TOUCH:
            bands[-1][1] = max(bands[-1][1], y1)
        else:
            bands.append([y0, y1])
    out = []
    for y0, y1 in bands:
        xs = sorted({vx for vy0, vy1, vx in vs
                     if vy0 < y1 - 2.0 and vy1 > y0 + 2.0})
        if len(xs) >= GB_MIN_V:
            out.append((y0, y1, xs))
    return out


def _skeletons_same(xs_a: list, xs_b: list) -> bool:
    """Same column skeleton on both sides of a gap (x-match both ways)."""
    ma = sum(1 for x in xs_a if any(abs(x - xb) <= GB_X_TOL for xb in xs_b))
    mb = sum(1 for x in xs_b if any(abs(x - xa) <= GB_X_TOL for xa in xs_a))
    return ma >= GB_SAME * len(xs_a) and mb >= GB_SAME * len(xs_b)


def split_by_grid_bands(page, box: Rect, H: list, V: list) -> List[Rect]:
    """
    Split one detection box that spans several vertically stacked CLOSED
    grids into per-grid boxes, from ruling evidence alone.

    The header-mirror splitter (split_stacked_tables) handles wireless
    stacks via text structure; it cannot fire on short boxes and cannot
    tell two bordered grids apart when their headers differ.  Two closed
    grids stacked in one box are unambiguous geometrically: no vertical
    rule crosses the gap between them.  Per-gap decision:

      - both bands header-only (no numeric data row)  -> CUT: sibling
        template grids (one-row disclosure headers share the column
        skeleton, so geometry alone must decide and each closed grid is
        its own table);
      - same column skeleton across the gap           -> NO CUT: one
        table continuing past a section row (its columns resume below);
      - upper band label-only and short, lower has data -> NO CUT: a
        floating header grid of ONE table;
      - lower band a lone header row (<2 rows, no data) after a real
        table -> NO CUT: an orphaned bare-header grid is folded into
        the region above (annotation convention);
      - otherwise (different skeletons)               -> CUT.

    Segments tighten to their band extents, which also recovers a grid
    whose bottom was clipped by earlier shrink-only refinement.
    """
    bands = _v_bands(box, V)
    if len(bands) < 2:
        return [box]

    def _band_rows(y0: float, y1: float) -> list:
        return _rows(page, (box[0], y0, box[2], y1))

    cuts: List[float] = []
    for (ay0, ay1, axs), (by0, by1, bxs) in zip(bands, bands[1:]):
        if by0 - ay1 < GB_MIN_GAP:
            continue
        rows_a = _band_rows(ay0, ay1)
        rows_b = _band_rows(by0, by1)
        data_a = any(_row_kind(r[2]) == "data" for r in rows_a)
        data_b = any(_row_kind(r[2]) == "data" for r in rows_b)
        if not data_a and not data_b:
            cuts.append((ay1 + by0) / 2)                     # siblings
        elif _skeletons_same(axs, bxs):
            continue                                         # same table
        elif not data_a and (ay1 - ay0) <= GB_HDR_MAX_H and data_b:
            continue                                         # floating header
        elif not data_b and len(rows_b) < 2:
            # An orphaned bare-header grid below a real table: a lone
            # header row with no body of its own is folded into the
            # region above (annotation convention), not counted as a
            # standalone table.  Two header-only bands never reach here —
            # the sibling rule above already cut them.
            continue
        else:
            cuts.append((ay1 + by0) / 2)                     # different grids
    if not cuts:
        return [box]

    bounds = [box[1]] + cuts + [box[3]]
    segments: List[Rect] = []
    for lo, hi in zip(bounds, bounds[1:]):
        my = [(y0, y1) for y0, y1, _ in bands if y0 < hi and y1 > lo]
        if not my:
            # a piece holding no grid at all (a caption strip between two
            # grids) belongs to neither table — ground truth excludes it.
            continue
        segments.append((box[0], min(y0 for y0, _ in my) - GB_PAD,
                         box[2], max(y1 for _, y1 in my) + GB_PAD))
    return segments if len(segments) > 1 else [box]


# ── Per-page markdown splitting & verification (C8) ─────────────────────────
#
# Root cause of the systematic -1 offset: export_to_markdown numbers its
# "<!-- Page N -->" markers with the fitz 0-based page index, while every
# downstream consumer (per-page md files, audit tooling, dashboards, RAG
# citations) numbers physical pages from 1.  Splitting a full-book md on the
# raw markers therefore yields page_0NNN.md holding physical page NNN+1's
# content (verified full-book on 平安银行 296p / 滨海能源 169p:
# marker N segment matches PDF physical page N+1 with ~0.75-1.0 signature
# coverage vs ~0.0 on both neighbours; evidence in
# analysis/fix_c1c8/trackE/).  The segment CONTENT is correctly assigned to
# its marker — page headers/footers are NOT bled into the next page — so the
# entire offset is the marker numbering base.
#
# These helpers are the single canonical per-page entry point:
#   split_markdown_pages  — split + re-base segments to 1-based physical
#                           page numbers regardless of on-disk marker base;
#   tag_pdf_page          — stamp an explicit "<!-- PDF page: N -->" header
#                           (fix_recommendations C8 item 3);
#   write_per_page_md     — emit page_0NNN.md files keyed by physical page;
#   verify_markdown_pages — re-match every segment against the source PDF
#                           and flag any page whose content sits on a
#                           neighbouring physical page (C8 item 2), so an
#                           offset can never ship silently again.

_MD_PAGE_RE = re.compile(r'<!-- Page (\d+) -->')
_PDF_PAGE_META_RE = re.compile(r'<!-- PDF page:\s*(\d+) -->')
# Signature alphabet for ownership matching: CJK ideographs + digits only.
# Markdown syntax (###, |, ---), punctuation and whitespace differ between
# md and the PDF text layer; neither carries page identity.
_SIG_STRIP_RE = re.compile(r'[^一-鿿0-9]')

# Signature window length (significant chars).  A text-identity parameter,
# not a geometric threshold: long enough that a window is near-unique within
# a page, short enough to survive reading-order reflow inside table rows.
# Calibrated full-book on 平安银行/滨海能源: w=12 keeps every page's
# expected-page coverage strictly above both neighbours (w=20/40 start
# false-flagging digit-fragmented table pages).
VERIFY_WINDOW = 12


def _sig(text: str) -> str:
    return _SIG_STRIP_RE.sub('', text)


def split_markdown_pages(md_text: str,
                         page_count: Optional[int] = None) -> Dict[int, str]:
    """Split full-book markdown into ``{physical_page: segment}``.

    Keys are 1-based physical page numbers regardless of the on-disk marker
    base.  Base detection: a ``Page 0`` marker proves legacy 0-based
    numbering; without it, ``page_count`` (physical page count of the source
    PDF) disambiguates — max marker == page_count - 1 means 0-based, max
    marker == page_count means 1-based.  With neither signal the markers are
    trusted as already 1-based (the forward format once the generator is
    fixed).
    """
    parts = _MD_PAGE_RE.split(md_text)
    markers = [int(parts[i]) for i in range(1, len(parts), 2)]
    if not markers:
        return {}
    if min(markers) == 0:
        base = 1                       # legacy 0-based markers
    elif page_count is not None and max(markers) == page_count - 1:
        base = 1                       # page 0 empty/absent, still 0-based
    else:
        base = 0                       # already 1-based
    return {int(parts[i]) + base: parts[i + 1]
            for i in range(1, len(parts), 2)}


def tag_pdf_page(segment: str, physical_page: int) -> str:
    """Prepend the explicit physical-page metadata header (idempotent)."""
    if _PDF_PAGE_META_RE.search(segment):
        return segment
    return f'<!-- PDF page: {physical_page} -->\n' + segment.lstrip('\n')


def write_per_page_md(md_text: str, out_dir: str, *,
                      page_count: Optional[int] = None,
                      with_meta: bool = True) -> List[int]:
    """Write one ``page_NNNN.md`` per physical page; returns pages written."""
    import os
    pages = split_markdown_pages(md_text, page_count)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for phys in sorted(pages):
        seg = pages[phys]
        if with_meta:
            seg = tag_pdf_page(seg, phys)
        with open(os.path.join(out_dir, f'page_{phys:04d}.md'),
                  'w', encoding='utf-8') as f:
            f.write(seg)
        written.append(phys)
    return written


def verify_markdown_pages(md_text: str, pdf, *,
                          window: int = VERIFY_WINDOW) -> dict:
    """Verify every segment sits on its physical page (C8 split校验).

    ``pdf`` is a filesystem path or an open fitz.Document.  For each segment
    the signature (CJK+digit) windows are matched against the expected
    physical page and its two neighbours.  Verdicts per page:
      ok              — expected page strictly beats both neighbours;
      OFFSET_SUSPECT  — a neighbour beats the expected page (the C8 defect);
      ambiguous       — expected ties a neighbour (near-empty/cover pages
                        whose little text recurs across pages);
      low_signal      — no window matched anywhere (heavy reading-order
                        reflow, e.g. TOC row/page-number re-pairing, or
                        image-only pages) — unverifiable by signature.
    Returns ``{'pages': {phys: {...}}, 'summary': {...}}``.
    """
    import fitz
    doc = fitz.open(pdf) if isinstance(pdf, str) else pdf
    pages = split_markdown_pages(md_text, len(doc))
    ptxt = [_sig(doc[p].get_text()) for p in range(len(doc))]

    def cover(seg_sig: str, p: int) -> float:
        wins = [seg_sig[i:i + window]
                for i in range(0, max(1, len(seg_sig) - window + 1), window)
                if len(seg_sig[i:i + window]) == window]
        if not wins:
            return -1.0                # no signature material at all
        return sum(1 for w in wins if w in ptxt[p]) / len(wins)

    report: Dict[int, dict] = {}
    for phys in sorted(pages):
        s = _sig(pages[phys])
        idx = phys - 1
        c_exp = cover(s, idx) if 0 <= idx < len(doc) else -1.0
        c_prev = cover(s, idx - 1) if idx - 1 >= 0 else -1.0
        c_next = cover(s, idx + 1) if idx + 1 < len(doc) else -1.0
        c_near = max(c_prev, c_next)
        if c_exp < 0:
            verdict = 'empty'
        elif c_exp == 0 and c_near <= 0:
            verdict = 'low_signal'
        elif c_near > c_exp:
            verdict = 'OFFSET_SUSPECT'
        elif c_near == c_exp:
            verdict = 'ambiguous'
        else:
            verdict = 'ok'
        report[phys] = {'verdict': verdict,
                        'coverage': round(c_exp, 3),
                        'prev': round(c_prev, 3),
                        'next': round(c_next, 3)}
    summary: Dict[str, int] = {}
    for r in report.values():
        summary[r['verdict']] = summary.get(r['verdict'], 0) + 1
    if isinstance(pdf, str):
        doc.close()
    return {'pages': report, 'summary': summary,
            'offset_suspects': [p for p, r in report.items()
                                if r['verdict'] == 'OFFSET_SUSPECT']}
