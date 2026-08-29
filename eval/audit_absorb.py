#!/usr/bin/env python
"""
Absorption audit: run detection over every GT page and report

  1. the absorb word-list -- every text line absorb_scope_label actually
     pulled into a box.  Healthy output is a tiny vocabulary of scope
     labels (本集团 / 本行 / 本集团及本行).  Any stray word is a red flag:
     the gate is leaking and must be TIGHTENED (never widened).

  2. the parenthesised-line reconciliation -- every fully parenthesised
     line found in the absorption candidate band (just above a predicted
     box top) is checked against the GT: does the annotator include or
     exclude it?  The paren veto encodes an observed annotation standard
     (75:0 excluded on 平安银行), not a semantic assumption; a single
     "GT includes" counter-example means the veto must be re-examined.

Usage (project root):
    ~/sda/facai/py_venv/Assay/bin/python eval/audit_absorb.py

Run after every pipeline change and after annotating a new PDF.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assay.splitting as splitting          # noqa: E402
import assay.detection as detection          # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GT_PATH = EVAL_DIR / "ground_truth.json"
PDF_DIRS = [Path("/home/homio/sda/1111/label"),
            Path("/home/homio/sda/1111/original")]
BAND = splitting.SCOPE_GAP_MAX   # candidate band above a box top (pt)


def _find_pdf(stem: str) -> Path:
    for d in PDF_DIRS:
        p = d / f"{stem}.pdf"
        if p.exists():
            return p
    sys.exit(f"PDF not found for stem {stem} in {PDF_DIRS}")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _page_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if txt:
                out.append((*line["bbox"], txt))
    return out


def main() -> None:
    if not GT_PATH.exists():
        sys.exit("no ground_truth.json -- run import_annotations.py first")
    gt_all = json.loads(GT_PATH.read_text())

    # hook absorb_scope_label to log every absorption with its text
    absorbed = []            # (stem, page_no, text)
    _orig = splitting.absorb_scope_label

    def spy(page, box):
        out = _orig(page, box)
        if out != box:
            texts = [t for lx0, ly0, lx1, ly1, t in _page_lines(page)
                     if out[1] - 1 <= ly0 and ly1 <= box[1] + 1]
            absorbed.append((page.number + 1, " | ".join(texts)))
        return out

    splitting.absorb_scope_label = spy
    detection.absorb_scope_label = spy

    inc, exc = [], []        # paren reconciliation
    for stem, pages in sorted(gt_all.items()):
        pdf = fitz.open(str(_find_pdf(stem)))
        for pg_str in sorted(pages, key=int):
            page = pdf[int(pg_str) - 1]
            preds = [list(b[:4]) for b in detection.detect_tables_on_page(page)]
            lines = _page_lines(page)
            gts = pages[pg_str]
            for box in preds:
                x0, y0, x1, y1 = box
                for lx0, ly0, lx1, ly1, t in lines:
                    if not (y0 - BAND <= ly1 <= y0 + 1 and lx1 > x0 and lx0 < x1):
                        continue
                    if not (t[0] in "（(" and t[-1] in "）)"):
                        continue
                    g = max(gts, key=lambda gb: _iou(box, gb), default=None)
                    if g is None or _iou(box, g) < 0.3:
                        continue
                    if g[1] <= ly0 + 1:
                        inc.append((pg_str, t))
                    elif g[1] >= ly1 - 1:
                        exc.append((pg_str, t))

    print(f"\n== 1. absorb word-list ({len(absorbed)} absorptions) ==")
    for txt, n in Counter(t for _, t in absorbed).most_common():
        pgs = [str(p) for p, t in absorbed if t == txt]
        print(f"  {n:4d}x  「{txt}」  p{','.join(pgs[:10])}"
              f"{'...' if len(pgs) > 10 else ''}")

    print(f"\n== 2. parenthesised-line reconciliation ==")
    print(f"  GT excludes (veto agrees): {len(exc)}")
    for txt, n in Counter(t for _, t in exc).most_common(8):
        print(f"    {n:4d}x 「{txt}」")
    print(f"  GT INCLUDES (veto WRONG -- red flag): {len(inc)}")
    for pg, txt in inc:
        print(f"    p{pg}: 「{txt}」")
    if inc:
        print("\n  !! counter-examples found: the paren veto no longer matches")
        print("     the annotation standard -- re-examine before trusting eval.")


if __name__ == "__main__":
    main()
