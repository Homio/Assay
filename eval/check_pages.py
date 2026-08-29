#!/usr/bin/env python
"""
Visual per-page check: render pages with GT boxes (green) and the CURRENT
pipeline's detection boxes (red) overlaid, so a disagreement is visible at
a glance.  This is the standing test artifact for detection work.

Usage (project root):
    ~/sda/facai/py_venv/Assay/bin/python eval/check_pages.py 195 202 ...
    ~/sda/facai/py_venv/Assay/bin/python eval/check_pages.py --all-failing

Output: eval/checks/p<NNNN>.png   (one image per page)

--all-failing re-scores every GT page against a fresh detection pass and
renders only pages where box count or IoU@0.7 matching disagrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assay.detection import detect_tables_on_page  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GT_PATH = EVAL_DIR / "ground_truth.json"
OUT_DIR = EVAL_DIR / "checks"
PDF_DIRS = [Path("/home/homio/sda/1111/label"),
            Path("/home/homio/sda/1111/original")]
ZOOM = 2.0


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
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua


def render(page: fitz.Page, gt_boxes, pred_boxes, out: Path) -> None:
    shape = page.new_shape()
    for b in gt_boxes:
        shape.draw_rect(fitz.Rect(*b))
        shape.finish(color=(0, 0.7, 0), width=1.4)
    for b in pred_boxes:
        shape.draw_rect(fitz.Rect(b[0] + 2, b[1] + 2, b[2] - 2, b[3] - 2))
        shape.finish(color=(0.9, 0, 0), width=1.0, dashes="[3] 0")
    shape.commit()
    page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM)).save(str(out))


def main() -> None:
    if not GT_PATH.exists():
        sys.exit("no ground_truth.json -- run import_annotations.py first")
    gt_all = json.loads(GT_PATH.read_text())
    OUT_DIR.mkdir(exist_ok=True)

    all_failing = "--all-failing" in sys.argv
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not want and not all_failing:
        sys.exit(__doc__)

    for stem, pages in sorted(gt_all.items()):
        pdf = fitz.open(str(_find_pdf(stem)))
        targets = sorted(int(p) for p in pages) if all_failing else want
        n_bad = 0
        for pg in targets:
            if str(pg) not in pages:
                print(f"p{pg}: not in GT, rendering anyway")
            gt_boxes = pages.get(str(pg), [])
            page = pdf[pg - 1]
            preds = [list(b[:4]) for b in detect_tables_on_page(page)]
            ok = len(preds) == len(gt_boxes) and all(
                max((_iou(p, g) for g in gt_boxes), default=0.0) >= 0.7
                for p in preds)
            if all_failing and ok:
                continue
            n_bad += 1
            out = OUT_DIR / f"p{pg:04d}.png"
            render(page, gt_boxes, preds, out)
            flag = "OK " if ok else "BAD"
            print(f"{flag} p{pg}: pred={len(preds)} gt={len(gt_boxes)} -> {out}")
        if all_failing:
            print(f"== {stem}: {n_bad} disagreeing pages rendered ==")


if __name__ == "__main__":
    main()
