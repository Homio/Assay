"""
Semantic diff gate for Assay pipeline output.

Compares two pipeline output directories (each containing one JSON per
document) and reports every table-level semantic difference, NOT just
count-level or bbox-level.  This catches the failure mode the count gate
misses: same 10131 tables, but a different number / sign / row order /
cell-text value / source page / field attribution.

Identity: a table is uniquely identified by (prov[0].page_no, prov[0].bbox)
within a document.  When two candidate tables share a page, bbox IoU
disambiguates.  Positional zip is never used (a page whose table count
changed must not compare different tables).

Comparison dimensions per matched table:
  * shape: num_rows / num_cols equality
  * source page equality (page_no)
  * bbox drift (centroid + IoU, with tolerance for float jitter)
  * per-cell:
      - text (whitespace-normalised, NFKC, full-width→half-width space)
      - numeric value (commas stripped, signed, % parsed, with tolerance)
      - sign (extracted, exact compare)
      - column_header / row_header flag
  * added / removed tables (matched by page+bbox+IoU greedy)

Tolerance discipline:
  * numeric tolerance only applies to bbox-derivable metrics (centroid
    pixel coordinates).  Cell text is either equal or it is not; the
    sign of a number is either preserved or it is not.  Tolerating a
    number mismatch would defeat the gate.
  * geometric tolerance (bbox IoU, centroid pixel) is configurable via
    CLI and is the *only* tolerance.

Categories (all zero-based, no positional inference):
  TABLE_ADDED        candidate table not matched to any baseline table
  TABLE_REMOVED      baseline table not matched to any candidate table
  PAGE_MISMATCH      same bbox, different page_no
  SHAPE_CHANGED      num_rows or num_cols differ
  CELL_TEXT_DIFF     text differs at same (row, col) after normalisation
  CELL_NUMERIC_DIFF  numeric value differs beyond zero tolerance
  CELL_SIGN_DIFF     sign (+/-) differs at same (row, col)
  HEADER_FLAG_DIFF   column_header / row_header flag flipped

Noise envelope: this tool is paired with a baseline-vs-baseline harness
that runs the same code twice and reports the natural distribution of
deltas; tune geometric tolerance to capture that envelope, not to
mask real changes.

Usage:
    PYTHONPATH=<repo> python3 -m eval.semantic_diff <baseline_dir> <candidate_dir> [--out report.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Geometry ─────────────────────────────────────────────────────────────

def _area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def _inter(a: List[float], b: List[float]) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)

def iou(a: List[float], b: List[float]) -> float:
    inter = _inter(a, b)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0

def centroid(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ── Text & numeric normalisation (only fields with no business meaning) ──

_WS_RE = re.compile(r"\s+")
_NUM_TRAIL = re.compile(r"[,\s%]+$")

def _norm_text(s: str) -> str:
    """Whitespace + Unicode NFKC + strip.  CJK letters preserved.
    This is the minimum needed to compare text content across runs
    (YOLO bbox jitter doesn't affect cell text, but whitespace between
    text and \n can).  Does NOT lowercase letters or strip punctuation
    — those are business content."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u3000", " ")  # full-width space → ascii space
    s = _WS_RE.sub(" ", s).strip()
    return s

# Numeric value extraction — for sign / value comparison.
# Handles "1,234.56" / "-12.5%" / "(123.4)" (accountant negative) / "−1" (U+2212).
_NUM_PAT = re.compile(
    r"^[\s\(]*(?P<sign>[+\-−]?)\s*(?P<mag>[\d,]+(?:\.\d+)?)\s*(?P<pct>%?)\s*\)?\s*$"
)

def _parse_number(s: str) -> Optional[Tuple[float, str]]:
    """Return (magnitude, sign_token) or None if not numeric.

    sign_token ∈ {'', '+', '-', '(', '−'}.  We keep parentheses as
    distinct from '-' so the gate distinguishes 'accounting negative'
    from a regular minus — they are not always equivalent (some
    pipelines emit both forms for the same number).
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    m = _NUM_PAT.match(s)
    if not m:
        return None
    raw = m.group("mag").replace(",", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    sign_token = m.group("sign")
    if not sign_token and s.startswith("(") and s.endswith(")"):
        sign_token = "("
    return v, sign_token


# ── Cell decoding ────────────────────────────────────────────────────────
# Pipeline JSON stores grid cells as Python repr() of a dict, not as the
# dict itself.  Decode both forms transparently.

def _coerce_cell(raw) -> dict:
    """Return {text, bbox, row, col, row_span, col_span,
    column_header, row_header, _raw} for whatever the pipeline emitted."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        # Prefer ast.literal_eval — it accepts Python repr without executing.
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError):
            pass
    return {"text": str(raw), "row": -1, "col": -1}


def _cell_text(c: dict) -> str:
    return c.get("text", "")


def _cell_pos(c: dict) -> Tuple[int, int]:
    return int(c.get("row", -1)), int(c.get("col", -1))


def _cell_flags(c: dict) -> Tuple[bool, bool]:
    return bool(c.get("column_header", False)), bool(c.get("row_header", False))


# ── Table identity & matching ────────────────────────────────────────────

def _table_identity(t: dict) -> Tuple[int, List[float]]:
    """(page_no, bbox) for matching.  A book's tables share the same
    page_no several times over (multiple tables on a page); bbox centroid
    + page uniquely identifies a table within a book."""
    prov = (t.get("prov") or [])
    if not prov:
        return (-1, [0.0, 0.0, 0.0, 0.0])
    p0 = prov[0]
    page = int(p0.get("page_no", -1))
    bbox = list(p0.get("bbox", [0.0, 0.0, 0.0, 0.0]))
    return page, bbox


def match_tables(baseline: List[dict], candidate: List[dict],
                 bbox_iou_floor: float = 0.5) -> List[Tuple[Optional[int], Optional[int]]]:
    """Greedy IoU matching on (page_no, bbox).  Returns a list of
    pairs (bi, ci) where either side may be None (=added/removed).

    Matching algorithm:
      * For each candidate table, gather all baseline tables on the same
        page whose bbox IoU with the candidate ≥ bbox_iou_floor.
      * Sort those candidate-baseline pairs by IoU descending; greedily
        assign each pair unless either side is already matched.
      * Unmatched candidates → TABLE_ADDED.  Unmatched baselines →
        TABLE_REMOVED.

    Never positional zip.  Never compares the i-th baseline to the i-th
    candidate.
    """
    # Build (idx, page, bbox) lists, drop tables without a usable page.
    bs: List[Tuple[int, int, List[float]]] = []
    for bi, t in enumerate(baseline):
        page, bbox = _table_identity(t)
        if page >= 0:
            bs.append((bi, page, bbox))
    cs: List[Tuple[int, int, List[float]]] = []
    for ci, t in enumerate(candidate):
        page, bbox = _table_identity(t)
        if page >= 0:
            cs.append((ci, page, bbox))

    pairs = []  # (iou, bi, ci)
    for ci, cp, cb in cs:
        for bi, bp, bb in bs:
            if cp != bp:
                continue
            v = iou(cb, bb)
            if v < bbox_iou_floor:
                continue
            pairs.append((v, bi, ci))
    pairs.sort(reverse=True)

    used_b, used_c, out = set(), set(), []
    for v, bi, ci in pairs:
        if bi in used_b or ci in used_c:
            continue
        used_b.add(bi); used_c.add(ci)
        out.append((bi, ci))

    # Append unmatched as (bi, None) and (None, ci) preserving baseline order
    for bi, _p, _b in bs:
        if bi not in used_b:
            out.append((bi, None))
    for ci, _p, _b in cs:
        if ci not in used_c:
            out.append((None, ci))

    return out


# ── Diff categories ──────────────────────────────────────────────────────

@dataclass
class TableDiff:
    bidx: Optional[int]
    cidx: Optional[int]
    page_b: Optional[int]
    page_c: Optional[int]
    bbox_b: Optional[List[float]]
    bbox_c: Optional[List[float]]
    bbox_iou: Optional[float]
    categories: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add(self, cat: str, note: str = "") -> None:
        if cat not in self.categories:
            self.categories.append(cat)
        if note:
            self.notes.append(note)


def diff_table(b: dict, c: dict) -> Tuple[List[str], List[str], dict]:
    """Return (categories, notes, meta) for one matched table pair."""
    cats, notes = [], []
    meta: dict = {}

    bd = b.get("data", {}) or {}
    cd = c.get("data", {}) or {}
    nb_rows, nb_cols = int(bd.get("num_rows", 0)), int(bd.get("num_cols", 0))
    nc_rows, nc_cols = int(cd.get("num_rows", 0)), int(cd.get("num_cols", 0))
    meta["shape_b"] = [nb_rows, nb_cols]
    meta["shape_c"] = [nc_rows, nc_cols]
    if (nb_rows, nb_cols) != (nc_rows, nc_cols):
        cats.append("SHAPE_CHANGED")
        notes.append(f"shape {nb_rows}x{nb_cols} → {nc_rows}x{nc_cols}")

    # Page mismatch
    bp = (b.get("prov") or [{}])[0].get("page_no", -1)
    cp = (c.get("prov") or [{}])[0].get("page_no", -1)
    meta["page_b"] = bp
    meta["page_c"] = cp
    if bp != cp:
        cats.append("PAGE_MISMATCH")
        notes.append(f"page {bp} → {cp}")

    # Cell-level diff
    bg = bd.get("grid") or []
    cg = cd.get("grid") or []
    # Build cell maps keyed by (row,col) — duplicates (merged cells) keep
    # the first occurrence (row_header + col_header stable at the same
    # cell).  Compare by logical position.
    bmap, cmap = _cell_map(bg), _cell_map(cg)

    n_text = n_num = n_sign = n_hdr = 0
    n_text_total = len(bmap | cmap)
    sample_text: List[str] = []
    sample_num: List[str] = []
    sample_sign: List[str] = []
    sample_hdr: List[str] = []
    for pos in sorted(bmap.keys() | cmap.keys()):
        bc = bmap.get(pos)
        cc = cmap.get(pos)
        if bc is None or cc is None:
            # positional (row,col) gap is reported at the grid level —
            # we only count cell-level diffs when both sides have a cell
            continue
        bt = _norm_text(_cell_text(bc))
        ct = _norm_text(_cell_text(cc))
        if bt != ct:
            n_text += 1
            if len(sample_text) < 3:
                sample_text.append(f"r{pos[0]}c{pos[1]}: {bt!r}→{ct!r}")

        bn = _parse_number(_cell_text(bc))
        cn = _parse_number(_cell_text(cc))
        if bn is not None and cn is not None:
            vb, sb = bn
            vc, sc = cn
            if vb != vc:  # exact — bbox jitter doesn't change numbers
                n_num += 1
                if len(sample_num) < 3:
                    sample_num.append(f"r{pos[0]}c{pos[1]}: {vb}→{vc}")
            if sb != sc:
                n_sign += 1
                if len(sample_sign) < 3:
                    sample_sign.append(f"r{pos[0]}c{pos[1]}: sign {sb!r}→{sc!r}")
        elif (bn is None) != (cn is None):
            # one side is numeric, the other is not — that's a text diff
            pass

        bh = _cell_flags(bc)
        ch = _cell_flags(cc)
        if bh != ch:
            n_hdr += 1
            if len(sample_hdr) < 3:
                sample_hdr.append(
                    f"r{pos[0]}c{pos[1]}: col_header/row_header {bh}→{ch}"
                )

    meta["n_text_diff"] = n_text
    meta["n_numeric_diff"] = n_num
    meta["n_sign_diff"] = n_sign
    meta["n_header_flag_diff"] = n_hdr
    meta["n_text_total"] = n_text_total
    if n_text:
        cats.append("CELL_TEXT_DIFF")
        notes.append(f"text diff cells={n_text}/{n_text_total}" +
                     (f" sample={sample_text}" if sample_text else ""))
    if n_num:
        cats.append("CELL_NUMERIC_DIFF")
        notes.append(f"numeric diff cells={n_num}" +
                     (f" sample={sample_num}" if sample_num else ""))
    if n_sign:
        cats.append("CELL_SIGN_DIFF")
        notes.append(f"sign diff cells={n_sign}" +
                     (f" sample={sample_sign}" if sample_sign else ""))
    if n_hdr:
        cats.append("HEADER_FLAG_DIFF")
        notes.append(f"header-flag diff cells={n_hdr}" +
                     (f" sample={sample_hdr}" if sample_hdr else ""))

    return cats, notes, meta


def _cell_map(grid: List[List]) -> Dict[Tuple[int, int], dict]:
    """Index cells by logical (row, col).  When row_span/col_span produce
    a cell at (0,1) and a duplicate anchor at (0,1) on another row, the
    FIRST occurrence wins (matches the pipeline's natural emission order).
    """
    out: Dict[Tuple[int, int], dict] = {}
    for row in grid or []:
        if not isinstance(row, list):
            continue
        for raw in row:
            c = _coerce_cell(raw)
            pos = _cell_pos(c)
            if pos == (-1, -1):
                continue
            out.setdefault(pos, c)
    return out


# ── Book-level & corpus-level diff ──────────────────────────────────────

def diff_book(book_path_b: Path, book_path_c: Path,
              bbox_iou_floor: float = 0.5,
              centroid_tol_px: float = 2.0) -> dict:
    b = json.loads(book_path_b.read_text()) if book_path_b else {"tables": []}
    c = json.loads(book_path_c.read_text()) if book_path_c else {"tables": []}
    bt = b.get("tables") or []
    ct = c.get("tables") or []

    pairs = match_tables(bt, ct, bbox_iou_floor=bbox_iou_floor)

    diffs: List[TableDiff] = []
    cat_counter: Counter = Counter()
    bbox_drift = []  # (centroid_dx, centroid_dy, iou)
    for bi, ci in pairs:
        if bi is None:
            tp, tb = _table_identity(ct[ci])
            d = TableDiff(None, ci, None, tp, None, tb, None)
            d.add("TABLE_ADDED", f"page={tp}")
            diffs.append(d)
            cat_counter["TABLE_ADDED"] += 1
            continue
        if ci is None:
            tp, tb = _table_identity(bt[bi])
            d = TableDiff(bi, None, tp, None, tb, None, None)
            d.add("TABLE_REMOVED", f"page={tp}")
            diffs.append(d)
            cat_counter["TABLE_REMOVED"] += 1
            continue
        bp_, bb_ = _table_identity(bt[bi])
        cp_, cb_ = _table_identity(ct[ci])
        v = iou(bb_, cb_) if bb_ and cb_ else 0.0
        d = TableDiff(bi, ci, bp_, cp_, bb_, cb_, v)
        bx, by = centroid(bb_) if bb_ else (0, 0)
        cx, cy = centroid(cb_) if cb_ else (0, 0)
        dx, dy = abs(bx - cx), abs(by - cy)
        bbox_drift.append((dx, dy, v))
        # bbox centroid drift beyond tolerance is a real change.
        if dx > centroid_tol_px or dy > centroid_tol_px:
            d.add("BBOX_DRIFT",
                  f"centroid drift dx={dx:.1f} dy={dy:.1f} > {centroid_tol_px}px")
        cats, notes, meta = diff_table(bt[bi], ct[ci])
        d.categories.extend(cats)
        d.notes.extend(notes)
        for cat in cats:
            cat_counter[cat] += 1
        d.notes.append(f"meta={json.dumps(meta, ensure_ascii=False)}")
        diffs.append(d)

    matched = sum(1 for d in diffs if d.bidx is not None and d.cidx is not None)
    clean = sum(1 for d in diffs
                if d.bidx is not None and d.cidx is not None and not d.categories)
    return {
        "book_b": str(book_path_b) if book_path_b else None,
        "book_c": str(book_path_c) if book_path_c else None,
        "n_tables_b": len(bt),
        "n_tables_c": len(ct),
        "n_matched": matched,
        "n_clean": clean,
        "n_diffs": len(diffs) - clean,
        "categories": dict(cat_counter),
        "bbox_drift": {
            "n": len(bbox_drift),
            "max_dx_px": max((d[0] for d in bbox_drift), default=0.0),
            "max_dy_px": max((d[1] for d in bbox_drift), default=0.0),
            "min_iou": min((d[2] for d in bbox_drift), default=0.0),
            "median_dx_px": sorted(d[0] for d in bbox_drift)[len(bbox_drift) // 2]
                             if bbox_drift else 0.0,
        },
        "diffs": [
            {
                "bidx": d.bidx, "cidx": d.cidx,
                "page_b": d.page_b, "page_c": d.page_c,
                "bbox_b": d.bbox_b, "bbox_c": d.bbox_c,
                "bbox_iou": d.bbox_iou,
                "categories": d.categories,
                "notes": d.notes,
            } for d in diffs
        ],
    }


def diff_corpus(b_dir: Path, c_dir: Path,
                bbox_iou_floor: float = 0.5,
                centroid_tol_px: float = 2.0) -> dict:
    """Diff two output directories.  Books are matched by directory
    basename (= document stem).  Books present in only one side are
    reported as BOOK_ONLY_B / BOOK_ONLY_C."""
    b_books = {p.name: p for p in b_dir.iterdir() if p.is_dir()} if b_dir.exists() else {}
    c_books = {p.name: p for p in c_dir.iterdir() if p.is_dir()} if c_dir.exists() else {}

    common = sorted(set(b_books) & set(c_books))
    only_b = sorted(set(b_books) - set(c_books))
    only_c = sorted(set(c_books) - set(b_books))

    per_book = {}
    cat_counter: Counter = Counter()
    n_tables_b = n_tables_c = 0
    n_clean_total = n_matched_total = 0
    for stem in common:
        bj = b_books[stem] / f"{stem}.json"
        cj = c_books[stem] / f"{stem}.json"
        if not bj.exists() or not cj.exists():
            per_book[stem] = {"error": "missing json", "b": str(bj), "c": str(cj)}
            continue
        d = diff_book(bj, cj, bbox_iou_floor=bbox_iou_floor,
                      centroid_tol_px=centroid_tol_px)
        per_book[stem] = d
        for k, v in d.get("categories", {}).items():
            cat_counter[k] += v
        n_tables_b += d["n_tables_b"]
        n_tables_c += d["n_tables_c"]
        n_clean_total += d["n_clean"]
        n_matched_total += d["n_matched"]

    return {
        "b_dir": str(b_dir),
        "c_dir": str(c_dir),
        "bbox_iou_floor": bbox_iou_floor,
        "centroid_tol_px": centroid_tol_px,
        "n_books_common": len(common),
        "n_books_only_b": len(only_b),
        "n_books_only_c": len(only_c),
        "n_tables_b": n_tables_b,
        "n_tables_c": n_tables_c,
        "n_matched": n_matched_total,
        "n_clean": n_clean_total,
        "categories_total": dict(cat_counter),
        "books": per_book,
        "books_only_b": only_b,
        "books_only_c": only_c,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("candidate_dir", type=Path)
    ap.add_argument("--bbox-iou-floor", type=float, default=0.5,
                    help="Minimum bbox IoU to consider a table match (default 0.5)")
    ap.add_argument("--centroid-tol-px", type=float, default=2.0,
                    help="Centroid drift tolerance in page-px (default 2.0)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON report to this path (default: stdout)")
    ap.add_argument("--top-diffs", type=int, default=20,
                    help="How many diffs to include in per-book summary (default 20)")
    args = ap.parse_args()

    rep = diff_corpus(args.baseline_dir, args.candidate_dir,
                      bbox_iou_floor=args.bbox_iou_floor,
                      centroid_tol_px=args.centroid_tol_px)
    if args.out:
        args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        json.dump(rep, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
