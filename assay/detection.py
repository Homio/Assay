"""
YOLO-based table detection with vector ruling-line refinement.

Table candidates come from two independent sources and are then refined by
*shrink-only* geometry (see :mod:`assay.rulings`):

Architecture (detect_tables_on_page pipeline):
  1. Render page → YOLO → table bboxes (dual-pass for wireless tables)
  2. Map image coords → PDF points, containment/IoU merge (NMS)
  3. Extract vector ruling rectangles (full grids + horizontal-ruled bands)
  4. Refine each YOLO bbox: snap to its ruling rectangle when one is covered,
     otherwise shrink to contained text spans — never grow outward.  This is
     what keeps page headers / body paragraphs out of the box.
  5. Recover ruling rectangles that YOLO missed entirely (rescues tables that
     the visual detector dropped or collapsed into a phantom title box).
  6. Size filter + containment dedup.
"""

from __future__ import annotations

import logging
import re as _re
from pathlib import Path
from typing import List, Tuple

import fitz

from assay.rulings import (
    detect_ruling_rects,
    extract_line_segments,
    recover_underline_grids,
    refine_bbox,
)
from assay.splitting import (
    absorb_scope_label,
    looks_text_table,
    recover_text_tables,
    split_by_grid_bands,
    split_by_period_bands,
    split_stacked_tables,
    _colrow_ok,
    _ENUM_HEAD_RES,
    _rows,
    _row_kind,
)

_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

YOLO_MODEL_PATH = "/home/homio/sda/1111/Assay/model/doclayout_yolo_ft.pt"

YOLO_CONFIDENCE = 0.3       # YOLO detection confidence threshold
YOLO_CONFIDENCE_FALLBACK = 0.05  # fallback for wireless/borderless tables
                              # when page_has_table heuristic says a table is
                              # likely but YOLO found nothing at standard conf.
                              # Wireless tables score ~0.17 vs bordered ~0.98.
MIN_CONFIDENCE = 0.30       # post-process: discard low-confidence detections
MERGE_IOU = 0.5             # IoU threshold for merging overlapping detections
CONTAINMENT_RATIO = 0.75    # if box A contains >=75% of box B, discard B
RULING_RECOVER_CONF = 0.50  # synthetic confidence for tables recovered purely
                              # from ruling rectangles (YOLO missed them).
                              # Still gated downstream by the PP-LCNet classifier.
MIN_TABLE_WIDTH = 200       # discard bboxes narrower than this (pt)
MIN_TABLE_HEIGHT = 50       # discard bboxes shorter than this (pt)
MIN_RULING_HEIGHT = 14      # ruling-rect candidates are backed by real drawn
                              # borders; a single header-row table (~16 pt) or
                              # a 1-2 row continuation fragment is still a
                              # genuine table, so use a smaller height floor
                              # than the visual-noise floor for YOLO-only boxes.
HIGH_CONF_EXEMPT = 0.75     # a YOLO detection this confident is trusted table
                              # evidence, so it skips the tall visual-noise floor
                              # (a short 1-2 row wireless table scores high yet
                              # falls under MIN_TABLE_HEIGHT).
PAGE_ZOOM = 2.0             # rendering resolution

HDR_BAND = 90.0             # running headers live in the top page band (pt);
                              # deepest observed across 13 books is ~80.
HDR_MIN_PAGES = 6           # a header signature must repeat on at least this
HDR_MIN_FRAC = 0.10         # many pages AND this fraction of the document —
                              # sectioned books swap header text mid-document
                              # (中石油 uses a per-part header on ~28% of pages),
                              # so each variant qualifies on its own count.
REMNANT_BAND = 130.0        # a continuation remnant grid sits at the very top
                              # of the page, right under the running header.
REMNANT_PREV_BOT = 90.0     # ...and only when the previous page's table runs
                              # into its bottom margin (rules within this band).
REMNANT_X_TOL = 2.0         # vertical-rule x-skeleton match tolerance across
REMNANT_X_MATCH = 0.8         # the page break, and the required match share.


# ──────────────────────────────────────────────────────────────────────────────
# YOLO singleton (loaded once, reused across pages)
# ──────────────────────────────────────────────────────────────────────────────

_yolo_model = None


def _get_yolo():
    """Lazy-load YOLO model (singleton). Auto-uses GPU if available."""
    global _yolo_model
    if _yolo_model is None:
        from doclayout_yolo import YOLOv10
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _log.info(f"Loading YOLO: {Path(YOLO_MODEL_PATH).name} ({device})")
        _yolo_model = YOLOv10(YOLO_MODEL_PATH)
        if torch.cuda.is_available():
            _yolo_model.model.to(device)
    return _yolo_model


# ──────────────────────────────────────────────────────────────────────────────
# YOLO detection → image-pixel coords
# ──────────────────────────────────────────────────────────────────────────────


def detect_tables_image(
    pix: fitz.Pixmap,
    conf: float = YOLO_CONFIDENCE,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Run YOLO on a rendered page image.

    Returns:
        List of (x0, y0, x1, y1, confidence) in image-pixel coordinates.
    """
    import tempfile
    model = _get_yolo()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        pix.save(tf.name)
        img_path = tf.name

    try:
        results = model.predict(source=img_path, imgsz=1024, conf=conf, verbose=False)
    finally:
        Path(img_path).unlink(missing_ok=True)

    tables = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_name = result.names[int(box.cls[0])].lower()
            # Accept only exact 'table' class — exclude 'table_caption',
            # 'table_footnote' which are sub-elements, not full tables
            if cls_name != 'table':
                continue
            img_box = box.xyxy[0].tolist()
            score = float(box.conf[0])
            tables.append((
                img_box[0], img_box[1], img_box[2], img_box[3], score
            ))

    return tables


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate mapping: image pixels → PDF points
# ──────────────────────────────────────────────────────────────────────────────


def image_to_pdf(
    tables: List[Tuple[float, float, float, float, float]],
    img_w: int, img_h: int,
    pdf_w: float, pdf_h: float,
) -> List[Tuple[float, float, float, float, float]]:
    """Map image-pixel table bboxes to PDF-point coordinates."""
    sx = pdf_w / img_w
    sy = pdf_h / img_h
    return [
        (t[0] * sx, t[1] * sy, t[2] * sx, t[3] * sy, t[4])
        for t in tables
    ]


# ──────────────────────────────────────────────────────────────────────────────
# NMS: containment filter + IoU merge
# ──────────────────────────────────────────────────────────────────────────────


def _containment_ratio(a, b):
    """
    How much of bbox b is contained inside bbox a.
    Returns ratio 0..1 (intersection area / area of b).
    """
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / area_b if area_b > 0 else 0.0


def _iou_xy(a, b):
    """IoU of two (x0, y0, x1, y1) boxes."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def merge_overlapping(
    tables,
    iou_threshold=MERGE_IOU,
    contain_threshold=CONTAINMENT_RATIO,
):
    """
    Merge overlapping table detections.

    Two-phase:
      1. Containment: if higher-conf A contains >=contain_threshold of B, discard B.
      2. IoU merge: merge remaining overlapping boxes by union.
    """
    if len(tables) <= 1:
        return tables

    tables = sorted(tables, key=lambda t: t[4], reverse=True)
    keep = [True] * len(tables)

    # Phase 1: Containment (two-directional).
    REVERSE_CONTAINMENT_RATIO = 0.70  # how much of i must be inside j
    CONFIDENCE_GAP = 0.60  # j's conf must be < 60% of i's conf to discard j
    for i in range(len(tables)):
        if not keep[i]:
            continue
        ti = (tables[i][0], tables[i][1], tables[i][2], tables[i][3])
        ci = tables[i][4]
        hi = ti[3] - ti[1]
        for j in range(i + 1, len(tables)):
            if not keep[j]:
                continue
            tj = (tables[j][0], tables[j][1], tables[j][2], tables[j][3])
            cj = tables[j][4]
            cr_ij = _containment_ratio(ti, tj)  # how much of j is inside i
            cr_ji = _containment_ratio(tj, ti)  # how much of i is inside j

            if cr_ij >= contain_threshold:
                keep[j] = False
            elif cr_ji >= REVERSE_CONTAINMENT_RATIO and cj < ci * CONFIDENCE_GAP:
                # A fragment too short to survive the size gate on its own
                # must not veto a full-height box that covers it: the row
                # sliver dies later anyway and would have killed the only
                # candidate carrying the whole table (a dead box vetoing a
                # live one).  Short-vs-short stays as before — there the
                # confident sliver genuinely outranks a noisier one.
                if hi < MIN_TABLE_HEIGHT and tj[3] - tj[1] >= MIN_TABLE_HEIGHT:
                    continue
                keep[j] = False

    kept = [t for i, t in enumerate(tables) if keep[i]]
    merged = []
    used = [False] * len(kept)

    for i, t1 in enumerate(kept):
        if used[i]:
            continue
        x0, y0, x1, y1 = t1[0], t1[1], t1[2], t1[3]
        best_conf = t1[4]
        used[i] = True

        # Phase 2: IoU merge
        for j, t2 in enumerate(kept):
            if used[j]:
                continue
            ix0 = max(x0, t2[0]); iy0 = max(y0, t2[1])
            ix1 = min(x1, t2[2]); iy1 = min(y1, t2[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area1 = (x1 - x0) * (y1 - y0)
            area2 = (t2[2] - t2[0]) * (t2[3] - t2[1])
            iou = inter / (area1 + area2 - inter)
            if iou > iou_threshold:
                x0 = min(x0, t2[0]); y0 = min(y0, t2[1])
                x1 = max(x1, t2[2]); y1 = max(y1, t2[3])
                best_conf = max(best_conf, t2[4])
                used[j] = True
        merged.append((x0, y0, x1, y1, best_conf))

    return merged


def _dedup_contained(tables, contain_threshold=0.80, ruling_rects=()):
    """
    Remove any bbox that is substantially contained within another.

    In financial reports nested table bboxes are always errors — tables don't
    nest. Keeps the larger bbox; ties broken by confidence.

    Two data-driven exceptions to "larger wins" (both verified across the
    8-book ground truth without a single false flip):
      * ruling-snap: a box that sits exactly on a drawn ruling grid beats a
        loose box that covers no grid and trails far behind in confidence —
        the loose one is a hallucinated blanket over the real table.
      * wireless blanket veto: a box that swallows >= 2 higher-confidence
        boxes while NONE of the three covers any drawn grid is a low-conf
        whole-notes-region blanket (underline-table zones); keep the parts.
        When the parts each snap to their own ruling rect the grouping is
        legitimate (a table's grid often splits into stacked bands), so the
        veto stays out of ruled zones.
    """
    if len(tables) <= 1:
        return tables

    def _rr_snap(t):
        return max((_iou_xy(t[:4], r) for r in ruling_rects), default=0.0)

    keep = [True] * len(tables)

    # Wireless blanket veto (pre-pass, needs the group view).
    for i in range(len(tables)):
        ti = tables[i]
        if _rr_snap(ti) >= 0.5:
            continue
        swallowed = [
            t for j, t in enumerate(tables)
            if j != i and t[4] > ti[4] and _rr_snap(t) < 0.8
            and _containment_ratio(ti[:4], t[:4]) >= contain_threshold
        ]
        if len(swallowed) >= 2:
            keep[i] = False

    for i in range(len(tables)):
        if not keep[i]:
            continue
        ti = (tables[i][0], tables[i][1], tables[i][2], tables[i][3])
        area_i = (ti[2] - ti[0]) * (ti[3] - ti[1])
        if area_i <= 0:
            continue
        for j in range(i + 1, len(tables)):
            if not keep[j]:
                continue
            tj = (tables[j][0], tables[j][1], tables[j][2], tables[j][3])
            area_j = (tj[2] - tj[0]) * (tj[3] - tj[1])
            if area_j <= 0:
                continue
            ix0 = max(ti[0], tj[0]); iy0 = max(ti[1], tj[1])
            ix1 = min(ti[2], tj[2]); iy1 = min(ti[3], tj[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            ratio_ij = inter / area_j  # how much of j is in i
            ratio_ji = inter / area_i  # how much of i is in j
            # Substantial mutual overlap => same table; keep the LARGER box
            # (matches the docstring).  Comparing areas rather than trusting
            # loop order matters when the pair neither strictly nests: a wide
            # but bottom-clipped detection can hold >=80% of a taller, fuller
            # underline-recovered box -- order alone would wrongly drop the
            # taller one that carries the missing trailing data rows.
            if ratio_ij >= contain_threshold or ratio_ji >= contain_threshold:
                # Ruling-snap exception: exact-grid box beats a gridless
                # blanket that is far less confident, regardless of area.
                si, sj = _rr_snap(tables[i]), _rr_snap(tables[j])
                ci = tables[i][4] if len(tables[i]) > 4 else 0.0
                cj = tables[j][4] if len(tables[j]) > 4 else 0.0
                if si >= 0.8 and sj < 0.5 and ci >= cj + 0.3:
                    keep[j] = False
                    continue
                if sj >= 0.8 and si < 0.5 and cj >= ci + 0.3:
                    keep[i] = False
                    break
                if area_i >= area_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [t for idx, t in enumerate(tables) if keep[idx]]


def _passes_size(c, ruling_rects, H, page=None) -> bool:
    """
    Size gate for a candidate ``(x0, y0, x1, y1, conf, backed)``.

    A candidate that encloses a drawn ruling grid (or was flagged ``backed``:
    split segment / ruling recovery / underline-grid recovery), or that
    contains several drawn horizontal rules of its own (a wireless table's
    header/value underlines), is backed by real table ink, so a short 1-2 row
    fragment is still a genuine table — apply the smaller ruling height floor.
    Candidates with no ruling backing keep the taller visual-noise floor that
    rejects stray captions / thin decorations.

    Text-structure rescue: a short unbacked box whose text reads as a
    complete mini table — a multi-cell header row plus a data row with >= 2
    numeric cells — is a genuine 2-row wireless table (关键管理人员薪酬 /
    合同负债 two-liners carry no ink at all).  A one-row sliver or a caption
    strip never shows both row kinds, so the floor still rejects those.
    """
    if (c[2] - c[0]) < MIN_TABLE_WIDTH:
        return False
    backed = c[5] or any(
        _containment_ratio(c[:4], r) >= 0.9 for r in ruling_rects)
    if not backed:
        backed = sum(
            1 for hx0, hx1, hy in H
            if c[1] + 1 < hy < c[3] - 1 and c[0] - 1 <= (hx0 + hx1) / 2 <= c[2] + 1
        ) >= 3
    high_conf = len(c) > 4 and c[4] >= HIGH_CONF_EXEMPT
    min_h = MIN_RULING_HEIGHT if (backed or high_conf) else MIN_TABLE_HEIGHT
    if (c[3] - c[1]) >= min_h:
        return True
    if page is not None and (c[3] - c[1]) >= MIN_RULING_HEIGHT:
        rows = _rows(page, c[:4])
        if len(rows) >= 2:
            has_data = any(
                len(r[2]) >= 2 and _row_kind(r[2]) == "data" for r in rows)
            has_header = any(
                len(r[2]) >= 2 and _row_kind(r[2]) == "label" for r in rows)
            if has_data and has_header:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Running page-header trim
# ──────────────────────────────────────────────────────────────────────────────

_hdr_sig_cache: dict = {}

_HDR_STRIP_RE = _re.compile(r"[\d\s]+")


def _hdr_sig(text: str) -> str:
    """Digit/whitespace-free signature of a running-header line."""
    return _HDR_STRIP_RE.sub("", text)


# ──────────────────────────────────────────────────────────────────────────────
# Statement / notes caption calibration (C2 报表页眉, C6 持续表接口)
# ──────────────────────────────────────────────────────────────────────────────
#
# 报表页眉 = 财报附注段每页顶部重复印刷的报表标题块
# （"XX公司 合并资产负债表(续) 2025年12月31日 人民币元"）。它有两种失稳形态：
#   * 跨页丢失——页眉共识过滤或"视觉行伙伴"连带删除把它从文本流剥掉
#     （平安 p122-p296：报表名行紧贴公司名行，被 ±2pt 交叠规则一并删除）；
#   * 吸入表格首格——未被识别时它留在表格框内，落成首行首格
#     （紫光 p99 "| 合并利润表 | 附注五 | …"）。
# 本模块在全书扫描阶段（calibrate_page_headers 的同一趟遍历）识别两类页眉行：
#   Class A 报表名行：签名整行匹配 报表名模式（允许 "1、" 编号前缀与 "(续)"），
#           记录在顶部带（×页高）内的出现页，以及全书每页最后一个报表名行
#           （会话锚点——续页缺页眉时由输出层继承上一报表名 + (续)）；
#   Class B 附注页眉行：顶部带内含"附注"且签名被运行页眉共识剥除的行
#           （"财务报表附注(续)"），同样属于被剥掉的内容型页眉。
# 识别结果只读地暴露给输出层（statement_caption_info），以块 metadata
# 形式挂回文本块/表格 prov——新增键，不改既有键语义。

CAPTION_TOP_FRAC = 0.25     # 报表名行的顶部识别带（×页高）：紫光表内标题行
                            # 在 ~0.15H，平安报表名行在 ~0.08H，均落入

_STMT_NAME_RE = _re.compile(
    r"^[、.．·]?[一-鿿]{0,4}"
    r"(?:资产负债表|利润表|现金流量表|股东权益变动表|所有者权益变动表)"
    r"(?:[(（]续[)）])?$")

_stmt_cap_cache: dict = {}


def _stmt_canon(sig: str) -> str:
    """Canonical statement name: strip leading enumerator and (续) marker."""
    s = _re.sub(r"^[、.．·]+", "", sig)
    s = _re.sub(r"[(（]续[)）]$", "", s)
    return s


def _file_stem(s: str) -> str:
    """Filename minus directory and a trailing .pdf (NOT Path.stem — book
    stems contain dots, e.g. '000001.SZ', which Path.stem would eat)."""
    n = Path(s).name
    return n[:-4] if n.lower().endswith(".pdf") else n


def statement_caption_info(stem: str) -> dict:
    """
    Caption metadata for the output layer (C2/C6), keyed by file stem.

    Returns {"pages": {page_no: caption entry}, "sessions": {page_no: entry}}
    or {} when no calibration ran (standalone use — fail-open, no behaviour
    change).  Populated as a side effect of calibrate_page_headers, which the
    extractor always calls in Phase 1, in-process, before export.
    """
    want = _file_stem(stem)
    for (name, _n), v in _stmt_cap_cache.items():
        if _file_stem(name) == want:
            return v
    return {}


def calibrate_page_headers(doc) -> set:
    """
    Learn this document's running-header line signatures.

    A running header is a document-level invariant: the same text (modulo
    page numbers) printed at the same y in the top page band on a large
    share of pages.  No genuine table row can repeat at one fixed y across
    that many pages, so a signature qualifying here is never table content.
    Signatures are keyed with a quantised y so a repeated column-header of
    a long continuation table (whose y floats with content) cannot qualify,
    while a book that moves its header between sections (平安, 中石油) yields
    one qualifying key per placement.  Returns a set of signature strings;
    empty set = headerless document, trim disabled (fail-open to the
    current behaviour — a missed trim never breaks anything).

    Side effect (C2/C6): the same single pass also collects statement-name
    caption lines and stripped 附注 header lines into _stmt_cap_cache for the
    output layer — see statement_caption_info.
    """
    key = (doc.name, len(doc))
    if key in _hdr_sig_cache:
        return _hdr_sig_cache[key]

    counts: dict = {}
    cap_pages: dict = {}      # page_no -> top-band caption entry (Class A/B)
    cap_sessions: dict = {}   # page_no -> last statement-name entry on page
    notes_cand: dict = {}     # page_no -> 附注 header candidate (pre-filter)
    plines: dict = {}         # page_no -> raw lines (caption pages only)
    for page in doc:
        seen = set()
        top_stmt = None
        last_stmt = None
        page_lines = []
        ph = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                lx0, ly0, lx1, ly1 = line["bbox"]
                txt = "".join(s["text"] for s in line["spans"]).strip()
                sig = _hdr_sig(txt)
                if txt:
                    page_lines.append((lx0, ly0, lx1, ly1, txt, sig))
                if _STMT_NAME_RE.match(sig):
                    canon = _stmt_canon(sig)
                    entry = {
                        "kind": "statement",
                        "name": canon,
                        "text": txt,
                        "bbox": (lx0, ly0, lx1, ly1),
                        "printed_continued": "续" in sig and sig != canon,
                    }
                    if ly0 < CAPTION_TOP_FRAC * ph and top_stmt is None:
                        top_stmt = entry
                    last_stmt = entry
                if ly0 >= HDR_BAND:
                    continue
                if len(sig) >= 4:
                    seen.add((sig, round(ly0 / 3.0)))
                    if "附注" in sig and page.number not in notes_cand:
                        notes_cand[page.number] = {
                            "kind": "notes",
                            "name": sig,
                            "text": txt,
                            "bbox": (lx0, ly0, lx1, ly1),
                        }
        for k in seen:
            counts[k] = counts.get(k, 0) + 1
        if top_stmt is not None:
            cap_pages[page.number] = top_stmt
        if last_stmt is not None:
            cap_sessions[page.number] = last_stmt
        if top_stmt is not None or page.number in notes_cand:
            plines[page.number] = page_lines

    floor = max(HDR_MIN_PAGES, HDR_MIN_FRAC * len(doc))
    sigs = {sig for (sig, _), n in counts.items() if n >= floor}
    # Class B 附注页眉: only a signature the running-header consensus actually
    # strips is a lost caption worth re-emitting; statement entries win ties.
    for pno, nentry in notes_cand.items():
        if nentry["name"] in sigs and pno not in cap_pages:
            cap_pages[pno] = nentry
    # Caption-block context: the date/unit lines printed with the caption are
    # deleted by the same partner rule (平安 p150 "2024年度").  Record the
    # non-running-header lines in a ±2.2×line-height window below each
    # caption so the output layer can restore the whole block, not just the
    # statement name.  Company-name lines qualify as running-header sigs and
    # are deliberately excluded — that trim is by design.
    for pno, entry in cap_pages.items():
        ex0, ey0, ex1, ey1 = entry["bbox"]
        ch = max(ey1 - ey0, 6.0)
        ctx = []
        for (lx0, ly0, lx1, ly1, txt, sig) in plines.get(pno, ()):
            if ly0 < ey0 - 0.2 * ch or ly0 > ey1 + 2.2 * ch:
                continue
            if abs(ly0 - ey0) < 1.0 and txt == entry["text"]:
                continue                    # the caption line itself
            if sig in sigs or _STMT_NAME_RE.match(sig):
                continue                    # running header / another caption
            if len(_re.sub(r"\s+", "", txt)) < 2:
                continue
            ctx.append({"text": txt, "bbox": (lx0, ly0, lx1, ly1)})
        ctx.sort(key=lambda c: c["bbox"][1])
        if ctx:
            entry["context"] = ctx[:3]
    _hdr_sig_cache[key] = sigs
    _stmt_cap_cache[key] = {"pages": cap_pages, "sessions": cap_sessions}
    return sigs


def _recover_top_remnant(page, H, V):
    """
    Recover a 1-row continuation fragment at the very top of the page.

    The last row of a bordered table can spill alone onto the next page as
    a closed grid only ~12 pt tall — under every height floor — yet it is
    unambiguous given cross-page evidence.  Three factors must all hold:
    a closed multi-column grid in the top band, the previous page's table
    running into its bottom margin, and the vertical-rule x-skeleton
    matching across the break rule-for-rule.  Purely additive: the caller
    only appends the box when it overlaps no existing candidate.
    """
    if page.number == 0:
        return None
    top_h = [h for h in H if h[2] <= REMNANT_BAND
             and h[1] - h[0] >= MIN_TABLE_WIDTH]
    if len(top_h) < 2:
        return None
    y0 = min(h[2] for h in top_h)
    y1 = max(h[2] for h in top_h)
    if not 6.0 <= y1 - y0 < MIN_RULING_HEIGHT:
        return None            # taller fragments already pass normal gates
    xs = sorted({v[2] for v in V
                 if v[0] <= y0 + 2.0 and v[1] >= y1 - 2.0})
    if len(xs) < 4 or xs[-1] - xs[0] < MIN_TABLE_WIDTH:
        return None
    prev = page.parent[page.number - 1]
    pH, pV = extract_line_segments(prev)
    cut = prev.rect.height - REMNANT_PREV_BOT
    if not any(h[2] >= cut for h in pH):
        return None            # previous page's table doesn't reach the margin
    # A drawn table's inner column rules often stop a hair short of its
    # bottom rule (typesetters inset them), so verticals ending just above
    # the bottom-margin band still belong to the table that reached it —
    # a bare `>= cut` misses them and the skeleton vote dies (城发 p16:
    # inner rules end at ~750 vs cut 752, match 2/8).  The cross-page
    # match requirement itself stays strict.
    pxs = {v[2] for v in pV if v[1] >= cut - 15.0}
    hit = sum(1 for x in xs
              if any(abs(x - px) <= REMNANT_X_TOL for px in pxs))
    if hit < REMNANT_X_MATCH * len(xs):
        return None
    # same 2 pt pad refine_bbox puts around a snapped ruling rect
    return (xs[0] - 2.0, y0 - 2.0, xs[-1] + 2.0, y1 + 2.0)


def top_remnant_candidate(page) -> bool:
    """
    Cheap cross-page continuation check for pages the numeric/text
    preflight would skip: a page holding ONLY a spilled continuation row
    plus prose (滨海 p102, 诚志 p189) carries no digit density, no ruling
    rect and no text-table geometry, so page_has_table is False and the
    page never reaches the detector.  The remnant test itself is precise
    (three cross-page factors), so letting it vouch for such a page costs
    one line-segment extraction and admits almost nothing else.
    """
    if page.number == 0:
        return False
    H, V = extract_line_segments(page)
    return _recover_top_remnant(page, H, V) is not None


FRAME_EXTEND_MIN_DEPTH = 25.0   # a containing grid must reach this far
                                # below the box before the box counts as
                                # its trimmed head (pt)


def _extend_to_containing_frame(box, ruling_rects, page):
    """
    Undo a value-grid-bottom trim when the box sits inside a deeper
    closed grid: a bordered disclosure frame whose head is a data table
    and whose tail is prose （恒逸 p82: 分红数据表 + 说明文字 in one drawn
    frame).  The trim reads the prose tail as outside the value grid and
    amputates it; the containing grid is the frame's own border, so its
    bottom is the table's true bottom (annotation convention: a closed
    frame is the table).

    Guards keep this from becoming "grow into the next table": the box
    must lie fully inside the grid, the grid's bottom must reach at least
    FRAME_EXTEND_MIN_DEPTH below the box (stacked sibling tables are
    separate grids whose bottoms coincide with the box — measured 0/4555
    TPs sit inside a deeper grid), and the extension strip must carry
    text (an empty decorative tail is not table content).
    """
    x0, y0, x1, y1 = box
    best = None
    for r in ruling_rects:
        if (r[0] - 2.0 <= x0 and r[1] - 2.0 <= y0
                and r[2] + 2.0 >= x1 and r[3] >= y1 + FRAME_EXTEND_MIN_DEPTH):
            if best is None or r[3] < best[3]:
                best = r
    if best is None:
        return box
    strip = fitz.Rect(x0, y1 + 1.0, x1, best[3])
    if not page.get_text("text", clip=strip).strip():
        return box
    return (x0, y0, x1, best[3])


def _trim_page_headers(page, boxes):
    """
    Clip a detected box's top below the running header it swallowed.

    A continuation page opens with bare table body; top-refinement finds no
    anchor above it and the box bleeds over the page header, dragging the
    header text into the table's cells downstream.  Pure shrink: a box is
    clipped only when a calibrated header line lies inside its top edge AND
    nothing but header lines sits between the old top and the clip line, so
    a box whose top already holds real content is never touched.  Ground
    truth never includes the running header, so the clip is monotone — it
    can only move a box toward its true extent, never away.
    """
    if not boxes:
        return boxes
    sigs = calibrate_page_headers(page.parent)
    if not sigs:
        return boxes

    lines = []          # (x0, y0, x1, y1, is_header) in the top band region
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly0 >= HDR_BAND + 40:    # clip decisions live near the top
                continue
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if not txt:
                continue
            is_hdr = ly0 < HDR_BAND and _hdr_sig(txt) in sigs
            lines.append((lx0, ly0, lx1, ly1, is_hdr))
    if not any(l[4] for l in lines):
        return boxes

    out = []
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        # any horizontally-overlapping header line the box top pokes into
        # (fully swallowed or clipped by even a point — a box grazing the
        # header's bottom edge still drags header ink into its cells)
        clip = max((l[3] for l in lines
                    if l[4] and y0 < l[3] < y1
                    and min(x1, l[2]) > max(x0, l[0])), default=None)
        if clip is not None and not any(
                not l[4]
                and min(x1, l[2]) > max(x0, l[0])
                and y0 - 1 < (l[1] + l[3]) / 2 < clip
                for l in lines):
            box = (x0, clip + 1.0, *box[2:])
        out.append(box)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Grid-continuity merge
# ──────────────────────────────────────────────────────────────────────────────

MERGE_GAP = 40.0        # max y-gap between fragments of one grid (pt)


def _merge_grid_continuous(candidates, V):
    """
    Merge vertically adjacent fragments of ONE ruled grid.

    Two x-aligned candidates whose union is spanned top-to-bottom by a
    single vertical-rule band (_v_bands clustering is touch-tolerant, so
    per-row rule fragments chain together) are pieces of a single
    continuous grid — e.g. a header fragment above and a body fragment
    below with section rows stranded between them.  Closed stacked grids
    are safe: their verticals stop at each grid's own border, so no band
    ever spans the gap between two tables.
    """
    if len(candidates) < 2:
        return candidates
    from assay.splitting import _v_bands
    cands = sorted(candidates, key=lambda c: (c[1], c[0]))
    out = []
    for c in cands:
        if out:
            p = out[-1]
            gap = c[1] - p[3]
            ovx = min(p[2], c[2]) - max(p[0], c[0])
            if -2.0 <= gap <= MERGE_GAP and ovx > 0:
                union = (min(p[0], c[0]), p[1], max(p[2], c[2]), max(p[3], c[3]))
                spanned = any(
                    by0 <= p[1] + 2.0 and by1 >= c[3] - 2.0
                    for by0, by1, _xs in _v_bands(union, V))
                if spanned:
                    out[-1] = (min(p[0], c[0]), p[1],
                               max(p[2], c[2]), max(p[3], c[3]),
                               max(p[4], c[4]),
                               p[5] or (len(c) > 5 and c[5]))
                    continue
        out.append(tuple(c))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Full page detection pipeline
# ──────────────────────────────────────────────────────────────────────────────


def detect_tables_on_page(
    page: fitz.Page,
    pix: fitz.Pixmap = None,
    zoom: float = PAGE_ZOOM,
):
    """
    Full detection pipeline for a single PDF page.

    Returns list of (x0, y0, x1, y1, confidence) in PDF points.
    """
    if pix is None:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

    # ── Vector ruling rectangles (independent, high-precision candidates) ──
    ruling_rects, H, V = detect_ruling_rects(page)

    # ── YOLO detect — dual-pass when the page looks like a table ──
    text = page.get_text("text")
    total_chars = len(text)
    if total_chars > 0:
        digit_count = sum(1 for c in text if c.isdigit())
        density = digit_count / total_chars
        heuristic_table = (digit_count >= 40 and density >= 0.04) or \
                          (digit_count >= 100 and density >= 0.08)
    else:
        heuristic_table = False
    if not heuristic_table:
        # Borderless all-text tables (related-party registers, entity lists)
        # carry no digits at all; the column-gutter geometry is the only
        # signal that the low-confidence fallback pass is worth running.
        heuristic_table = looks_text_table(page, tuple(page.rect))

    raw = detect_tables_image(pix)  # standard confidence pass

    # Map → PDF and merge the standard pass FIRST.  The fallback dedup below
    # must run against the NMS *survivors*: a low-confidence page-spanning
    # box can suppress a genuine fallback detection as "duplicate" and then
    # be discarded itself by reverse-containment — losing the table entirely
    # (net effect: a dead box vetoed a live one).
    pdf_tables = []
    if raw:
        pdf_tables = image_to_pdf(
            [t for t in raw if t[4] >= MIN_CONFIDENCE],
            img_w=pix.width, img_h=pix.height,
            pdf_w=page.rect.width, pdf_h=page.rect.height,
        )
        pdf_tables = merge_overlapping(pdf_tables)

    if heuristic_table:
        raw_fb = detect_tables_image(pix, conf=YOLO_CONFIDENCE_FALLBACK)
        if raw_fb:
            fb_pdf = image_to_pdf(
                raw_fb,
                img_w=pix.width, img_h=pix.height,
                pdf_w=page.rect.width, pdf_h=page.rect.height,
            )
            survivors = [(t[0], t[1], t[2], t[3]) for t in pdf_tables]
            fresh = []
            for t_fb in fb_pdf:
                fb_box = (t_fb[0], t_fb[1], t_fb[2], t_fb[3])
                is_dup = any(
                    _containment_ratio(eb, fb_box) >= 0.7 or
                    _containment_ratio(fb_box, eb) >= 0.7
                    for eb in survivors
                )
                if not is_dup:
                    fresh.append(t_fb)
            if fresh:
                pdf_tables = merge_overlapping(pdf_tables + fresh)

    # ── Refine each YOLO bbox: snap to ruling rect, else shrink to text ──
    # Then split boxes that span several stacked tables (header-restart
    # fission).  A split segment is validated evidence (header + data rows),
    # so it carries a backed=True flag for the size filter below.  Finally
    # absorb a scope label (本集团 / 本集团及本行 …) sitting just above the
    # top edge — the detector often leaves it out and shrink-only
    # refinement can never recover it.
    candidates = []
    for x0, y0, x1, y1, conf in pdf_tables:
        refined, _src = refine_bbox(page, (x0, y0, x1, y1), ruling_rects, H, V)
        refined = _extend_to_containing_frame(refined, ruling_rects, page)
        segments = split_stacked_tables(page, refined, H, V)
        if len(segments) == 1:
            # header-mirror fission declined; try the narrow period-band rule
            # (stacked statements sharing one column grid, each with its own
            # lone period row and drawn ruling band).
            segments = split_by_period_bands(page, refined, H, ruling_rects)
        if len(segments) == 1:
            # Split-then-refine safety net.  Refinement is a SINGLE-table
            # operation (value-grid-bottom trim, caption drop); run on a box
            # that stacks two identically-structured tables it can amputate
            # the lower table before the splitter ever sees it, collapsing
            # both into one over-run box.  Re-test the splitter on the
            # un-trimmed detector box; if it resolves more tables, the trim
            # destroyed a genuine split -- take the raw split and refine each
            # piece on its own so the trim runs per single table.
            raw_segs = split_stacked_tables(page, (x0, y0, x1, y1), H, V)
            if len(raw_segs) > 1:
                segments = [refine_bbox(page, s, ruling_rects, H, V)[0]
                            for s in raw_segs]
        # Grid-band fission: a box spanning several closed ruling grids is
        # cut per grid from vertical-rule evidence alone — handles stacks
        # the header-mirror splitter cannot (short boxes, grids whose
        # headers differ).  Each piece is ruling-backed by construction.
        gb = []
        for seg in segments:
            gb.extend(split_by_grid_bands(page, seg, H, V))
        segments = gb
        for seg in segments:
            seg = absorb_scope_label(page, seg)
            # Framed prose veto: a low-confidence, unsplit detection whose
            # area is not substantially covered by a drawn ruling grid and
            # with not a single numeric data row among its text rows is boxed
            # prose (key-audit-matters pages framed by a decorative border),
            # not a table.  Coverage, not containment: a thin header band
            # inside a page-tall box must not count as grid backing.  A
            # borderless all-text table has no data row either but shows the
            # column-gutter geometry prose never has -- let it through.
            if len(segments) == 1 and conf < 0.40 \
                    and not any(_containment_ratio(r, seg) >= 0.5
                                for r in ruling_rects) \
                    and not any(_row_kind(r[2]) == "data"
                                for r in _rows(page, seg)) \
                    and not looks_text_table(page, seg):
                continue
            candidates.append((*seg, conf, len(segments) > 1))

    # ── Recover ruling rectangles that YOLO missed entirely ──
    # Single-cell frames (no internal ruling line at all) are flowchart nodes
    # or decorative borders, never tables — a drawn table grid separates its
    # rows/columns with at least one internal rule.  An internal rule must
    # span half the rect (a flowchart connector arrow poking through the
    # border must not count).  And a drawn table always closes with a REAL
    # top and bottom rule: a rect whose edge carries no line was inflated by
    # stray strokes (an arrow shaft glued to a node frame) and is no table.
    for r in ruling_rects:
        covered = any(
            _containment_ratio(r, c[:4]) >= 0.5 or
            # a rect 30 %+ claimed by an existing box is a FRAGMENT of that
            # table (e.g. its ruled header band glued to the page-header
            # underline 33 pt above — 中石油 p158), never a missed table:
            # a truly missed table overlaps no candidate at all.
            _containment_ratio(c[:4], r) >= 0.3
            for c in candidates
        )
        has_inner = any(
            r[1] + 3.0 < hy < r[3] - 3.0
            and min(hx1, r[2]) - max(hx0, r[0]) >= 0.5 * (r[2] - r[0])
            for hx0, hx1, hy in H
        ) or any(
            r[0] + 3.0 < vx < r[2] - 3.0
            and min(vy1, r[3]) - max(vy0, r[1]) >= 0.5 * (r[3] - r[1])
            for vy0, vy1, vx in V
        )
        closed = all(
            any(abs(hy - edge) <= 3.0
                and min(hx1, r[2]) - max(hx0, r[0]) >= 0.5 * (r[2] - r[0])
                for hx0, hx1, hy in H)
            for edge in (r[1], r[3])
        )
        # A 2-rule H-band (no interior rule, no vertical anywhere near it)
        # bracketing ONLY multi-span data rows is a mini ruled table (中石油
        # p274 应占联营公司度量: two rules, two data rows) — not a flowchart
        # node (those are closed grids WITH verticals) nor a framed prose
        # strip (prose rows are single-span).  Text-row check is the only
        # evidence left when no interior rule exists, so it must be strict:
        # every row data, at least two rows.
        if not has_inner and not covered and closed and not any(
                r[0] - 3.0 <= vx <= r[2] + 3.0
                and vy1 >= r[1] - 3.0 and vy0 <= r[3] + 3.0
                for vy0, vy1, vx in V):
            drows = _rows(page, r)
            has_inner = len(drows) >= 2 and all(
                len(dr[2]) >= 2 and _row_kind(dr[2]) == "data"
                for dr in drows)
        if not covered and has_inner and closed:
            candidates.append((r[0], r[1], r[2], r[3], RULING_RECOVER_CONF, False))

    # ── Recover double-underline wireless tables YOLO dropped ──
    # Collapse duplicate/overlapping candidates first so the containment counts
    # below reflect *distinct* tables (YOLO-refine and ruling recovery can leave
    # two boxes on the same region); the final dedup runs again after sizing.
    candidates = _dedup_contained(candidates, ruling_rects=ruling_rects)
    # Drop candidates that cannot survive the size filter *before* reconciling,
    # so a sub-floor fragment (e.g. a 49pt unbacked wireless box) cannot block a
    # fuller ruling-backed ug box that would represent the same table properly.
    candidates = [c for c in candidates if _passes_size(c, ruling_rects, H, page)]
    # Add a per-table box anchored on an accounting double underline where the
    # detector under-performed.  Defensive rules keep this from displacing good
    # boxes (a ug box can over-split at a subtotal or over-merge a stack):
    #   * skip if a candidate already matches it well (IoU >= 0.5) — UNLESS
    #     that candidate sits almost entirely inside the ug box (>= 0.8) AND
    #     the ug bottom double rule lies clearly below the candidate's bottom:
    #     then the candidate is a clipped fragment whose data rows were cut
    #     (text-shrink stopped early) and the fuller ug box must be allowed in
    #     so the final containment dedup can retire the fragment.  The bottom
    #     anchor is the ug box's hard evidence; a ug box that merely pokes
    #     above the candidate's top has soft-inferred fluff (prose pulled into
    #     its header walk-up) and must NOT displace a good box;
    #   * skip if it would swallow >= 2 existing candidates (it merged a stack
    #     across a subtotal double rule — keep the finer boxes);
    #   * skip if it sits inside an existing candidate (redundant sub-fragment)
    #     — UNLESS the ug bottom lies clearly below that candidate's bottom, in
    #     which case the candidate is the clipped one and the taller ug box must
    #     survive (mirror of the IoU rule above; without it a wide bottom-clipped
    #     detection swallows the fuller box that carries the trailing rows).
    # Otherwise add it; when it encloses a single small partial box, the
    # containment dedup below drops that partial in favour of the fuller box.
    for u in recover_underline_grids(page, H, V):
        u_cand = (u[0], u[1], u[2], u[3], RULING_RECOVER_CONF, True)
        if not _passes_size(u_cand, ruling_rects, H):
            continue
        if any(_iou_xy(c[:4], u) >= 0.5
               and not (_containment_ratio(u, c[:4]) >= 0.8
                        and u[3] > c[3] + 8.0)
               for c in candidates):
            continue
        # A ug box that SWALLOWS a candidate without reaching below its
        # bottom anchor adds only above-top fluff (header walk-up climbed
        # through prose into the table above — 中石油 p190); the candidate
        # already owns the double rule, so the ug box brings no evidence.
        if any(_containment_ratio(u, c[:4]) >= 0.8
               and not u[3] > c[3] + 8.0
               for c in candidates):
            continue
        if sum(1 for c in candidates
               if _containment_ratio(u, c[:4]) >= 0.6) >= 2:
            continue
        if any(_containment_ratio(c[:4], u) >= 0.8
               and not u[3] > c[3] + 8.0
               for c in candidates):
            continue
        candidates.append(u_cand)

    # ── Recover borderless all-text tables (no digits, no rulings) ──
    # YOLO is near-blind to label/label tables (related-party registers,
    # entity lists score ~0.07); the shared column-gutter geometry is the
    # only evidence.  This recovery is a last-resort net: only add a box
    # over ground no candidate touches at all — any overlap means the
    # visual/ruling evidence already spoke there, and a gutter run that
    # pokes into an existing box has bridged across a section boundary.
    # One exception: a recovered box sitting right below an x-aligned text
    # table candidate (gap within a row spacing) is the continuation of
    # that same table — its middle rows wrap full-width and break the
    # gutter run — so extend the candidate instead (sibling stacked tables
    # sit >= 40 pt apart and never trigger this).
    for r in recover_text_tables(page):
        t_cand = (r[0], r[1], r[2], r[3], RULING_RECOVER_CONF, True)
        if not _passes_size(t_cand, ruling_rects, H):
            continue
        # A borderless visual box that swallowed the section headings above
        # a text table (紫光 p217 "3. 套期"): when a ruling-free candidate
        # contains the recovered row band and holds nothing table-like
        # beyond it, snap it to the band — the gutter rows are the only
        # tabular ink the box actually covers.
        for i, c in enumerate(candidates):
            ovx = min(c[2], r[2]) - max(c[0], r[0])
            if ovx <= 0 or min(c[3], r[3]) - max(c[1], r[1]) <= 0:
                continue
            if c[1] <= r[1] + 2.0 and c[3] >= r[3] - 2.0 \
                    and ovx / (r[2] - r[0]) >= 0.9 \
                    and not any(min(c[2], rr[2]) > max(c[0], rr[0])
                                and min(c[3], rr[3]) > max(c[1], rr[1])
                                for rr in ruling_rects):
                strips = [(c[0], c[1], c[2], r[1]), (c[0], r[3], c[2], c[3])]
                # headings only: a single-span line or an enumerator pair
                # ("3. 套期").  Any other multi-span row — even one the
                # row-level vetoes reject, like a Latin-keyed bond row —
                # is table ink and forbids the snap.  And at least one row
                # must open with an explicit enumerator: full-width note
                # lines and period captions inside a table are also
                # single-span, but never enumerated.
                def _heading(row):
                    sp = row[2]
                    return len(sp) == 1 or (len(sp) == 2
                                            and not _colrow_ok(sp))
                def _enum_head(row):
                    head = row[2][0][2].strip().split()[0] if row[2] else ''
                    return any(rx.match(head) for rx in _ENUM_HEAD_RES)
                srows = [row for s in strips if s[3] - s[1] > 4.0
                         for row in _rows(page, s)]
                if srows and all(map(_heading, srows)) \
                        and any(map(_enum_head, srows)):
                    candidates[i] = (c[0], r[1], c[2], r[3], c[4], c[5])
            break
        if any(min(c[2], r[2]) - max(c[0], r[0]) > 0
               and min(c[3], r[3]) - max(c[1], r[1]) > 0
               for c in candidates):
            continue
        bridged = False
        for i, c in enumerate(candidates):
            gap = r[1] - c[3]
            ov = min(c[2], r[2]) - max(c[0], r[0])
            narrow = min(c[2] - c[0], r[2] - r[0])
            if 0 <= gap <= 32.0 and narrow > 0 and ov / narrow >= 0.85 \
                    and looks_text_table(page, c[:4]):
                candidates[i] = (min(c[0], r[0]), c[1],
                                 max(c[2], r[2]), r[3], c[4], c[5])
                bridged = True
                break
        if not bridged:
            candidates.append(t_cand)

    # ── Cross-page continuation remnant (closed grid too short for any
    # height floor, recovered from three-factor cross-page evidence) ──
    rem = _recover_top_remnant(page, H, V)
    if rem is not None and not any(
            min(c[2], rem[2]) > max(c[0], rem[0])
            and min(c[3], rem[3]) > max(c[1], rem[1])
            for c in candidates):
        candidates.append((*rem, RULING_RECOVER_CONF, True))

    # ── Merge fragments of one continuous grid ──
    # Mirror of grid-band fission: fission splits only at hard gaps no
    # vertical rule crosses, so when vertical rules CROSS the y-gap
    # between two x-aligned candidates, both are fragments of one grid
    # (a header fragment + a body fragment with section rows stranded
    # between them) and must join.  Stacked closed grids never trigger:
    # their verticals stay inside each grid, nothing crosses the gap.
    candidates = _merge_grid_continuous(candidates, V)

    # ── Final containment dedup ──
    return _trim_page_headers(
        page, _dedup_contained([c[:5] for c in candidates],
                               ruling_rects=ruling_rects))
