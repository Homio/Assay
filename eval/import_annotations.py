#!/usr/bin/env python3
"""
Convert a Label Studio JSON export into eval/ground_truth.json.

Only pages that carry at least one *annotation* (a submitted review, even if
it contains zero rectangles) enter the ground truth -- an un-reviewed page is
unknown, not "empty".  Rectangle percentages are mapped back to PDF points
via the page size, so the GT lives in the same coordinate system as the
pipeline output.

Usage:
    python3 eval/import_annotations.py <ls_export.json> [more_exports...]

Merges into (creating if absent):
    eval/ground_truth.json
      { "<pdf_stem>": { "<viewer_page>": [[x0,y0,x1,y1], ...], ... }, ... }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

LABEL_ROOT = Path("/home/homio/sda/1111/label")
GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"


def convert(export_path: str, gt: dict) -> int:
    tasks = json.loads(Path(export_path).read_text())
    n_pages = 0
    page_sizes: dict = {}

    for task in tasks:
        data = task.get("data", {})
        stem = data.get("pdf")
        page_no = data.get("page")
        annotations = [a for a in task.get("annotations", [])
                       if not a.get("was_cancelled")]
        if not stem or not page_no or not annotations:
            continue

        if stem not in page_sizes:
            pdf = LABEL_ROOT / f"{stem}.pdf"
            if not pdf.exists():
                sys.exit(f"PDF not found for sizing: {pdf}")
            doc = fitz.open(str(pdf))
            page_sizes[stem] = [(p.rect.width, p.rect.height) for p in doc]
            doc.close()
        pw, ph = page_sizes[stem][int(page_no) - 1]

        # Latest annotation wins if the page was reviewed more than once.
        ann = max(annotations, key=lambda a: a.get("updated_at") or "")
        boxes = []
        for res in ann.get("result", []):
            if res.get("type") != "rectanglelabels":
                continue
            v = res["value"]
            x0 = v["x"] / 100.0 * pw
            y0 = v["y"] / 100.0 * ph
            x1 = (v["x"] + v["width"]) / 100.0 * pw
            y1 = (v["y"] + v["height"]) / 100.0 * ph
            boxes.append([round(x0, 2), round(y0, 2),
                          round(x1, 2), round(y1, 2)])

        gt.setdefault(stem, {})[str(page_no)] = boxes
        n_pages += 1

    return n_pages


def main(paths: list) -> None:
    gt = json.loads(GT_PATH.read_text()) if GT_PATH.exists() else {}
    total = 0
    for p in paths:
        n = convert(p, gt)
        print(f"{p}: {n} reviewed pages")
        total += n
    GT_PATH.write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    n_boxes = sum(len(b) for pages in gt.values() for b in pages.values())
    n_pages = sum(len(pages) for pages in gt.values())
    print(f"\nground_truth.json: {len(gt)} PDFs, {n_pages} pages, "
          f"{n_boxes} table boxes -> {GT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: import_annotations.py <ls_export.json> [...]")
    main(sys.argv[1:])
