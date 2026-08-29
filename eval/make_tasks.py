#!/usr/bin/env python3
"""
Generate a Label Studio pre-annotation task package for one PDF.

For every page:
  1. Render to PNG at the pipeline's zoom (fitz.Matrix(2,2)) so the annotation
     canvas and the pipeline share one coordinate system.
  2. Run the *real* pipeline (AssayExtractor) once; its final table bboxes
     become Label Studio ``predictions`` (pre-drawn rectangles the annotator
     only has to correct, not draw).
  3. Flag suspect pages that deserve close human review -- the places where
     pre-annotation anchoring is dangerous:
       - digits-no-table : numeric-dense page but pipeline output nothing
       - yolo-dropped    : a confident raw YOLO box has no surviving table
       - ruling-no-table : a drawn ruling grid has no table covering it

Usage:
    PYTHONPATH=<repo> python3 eval/make_tasks.py <pdf_path>

Outputs (under ../label/):
    render/<stem>/page_NNNN.png   one image per viewer page
    tasks/<stem>_tasks.json       Label Studio import file (with predictions)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABEL_ROOT = Path("/home/homio/sda/1111/label")
ZOOM = 2.0


def _containment(a, b) -> float:
    """How much of box b lies inside box a (0..1)."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return (ix1 - ix0) * (iy1 - iy0) / area_b if area_b > 0 else 0.0


def _pct(v: float, total: float) -> float:
    """Clamp a PDF-point coordinate to a 0-100 percentage of the page."""
    return max(0.0, min(100.0, v / total * 100.0))


def main(pdf_path: str) -> None:
    from assay.models import TableModel
    from assay.extractor import AssayExtractor
    from assay.detection import detect_tables_image, image_to_pdf
    from assay.rulings import detect_ruling_rects

    pdf = Path(pdf_path)
    stem = pdf.stem
    render_dir = LABEL_ROOT / "render" / stem
    tasks_dir = LABEL_ROOT / "tasks"
    render_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/2] pipeline extract: {pdf.name}", flush=True)
    model = TableModel(use_gpu=True)
    doc_out = AssayExtractor(model).extract(str(pdf))
    by_page: dict = {}
    for t in doc_out.tables:
        by_page.setdefault(t.page_no, []).append(tuple(t.table_bbox))
    print(f"      {len(doc_out.tables)} tables on {len(by_page)} pages "
          f"({time.time() - t0:.0f}s)", flush=True)

    print(f"[2/2] render + raw-signal scan", flush=True)
    doc = fitz.open(str(pdf))
    mat = fitz.Matrix(ZOOM, ZOOM)
    tasks = []
    n_suspect = 0

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_name = f"page_{i + 1:04d}.png"
        pix.save(str(render_dir / img_name))

        finals = by_page.get(i, [])
        pw, ph = page.rect.width, page.rect.height

        # ── suspect flags ──
        reasons = set()
        text = page.get_text("text")
        tc = len(text)
        dc = sum(1 for c in text if c.isdigit())
        dens = dc / tc if tc else 0.0
        if ((dc >= 40 and dens >= 0.04) or (dc >= 100 and dens >= 0.08)) \
                and not finals:
            reasons.add("digits-no-table")

        raw = detect_tables_image(pix)  # standard-confidence YOLO pass
        raw_pdf = image_to_pdf(raw, pix.width, pix.height, pw, ph)
        for rb in raw_pdf:
            rbox = rb[:4]
            # Fraction of the raw box's area explained by final tables.  A
            # small final box *inside* a large raw box must not count as
            # coverage (that is exactly the page-202 failure shape).
            area = (rbox[2] - rbox[0]) * (rbox[3] - rbox[1])
            inter = sum(_containment(fb, rbox) for fb in finals) * area
            if area > 0 and min(inter / area, 1.0) < 0.5:
                reasons.add("yolo-dropped")
                break

        ruling_rects, _, _ = detect_ruling_rects(page)
        for r in ruling_rects:
            if not any(_containment(fb, r) >= 0.5 for fb in finals):
                reasons.add("ruling-no-table")
                break

        # two final boxes bleeding into each other (e.g. one table's box
        # swallowing the next table's header rows)
        for a in range(len(finals)):
            for b in range(a + 1, len(finals)):
                fa, fb2 = finals[a], finals[b]
                iw = min(fa[2], fb2[2]) - max(fa[0], fb2[0])
                ih = min(fa[3], fb2[3]) - max(fa[1], fb2[1])
                if iw > 5 and ih > 3:  # pdf points
                    reasons.add("box-overlap")
                    break
            if "box-overlap" in reasons:
                break

        if reasons:
            n_suspect += 1

        # ── Label Studio prediction rectangles (percent coordinates) ──
        result = []
        for k, (x0, y0, x1, y1) in enumerate(finals):
            result.append({
                "id": f"p{i + 1}t{k}",
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": pix.width,
                "original_height": pix.height,
                "value": {
                    "x": _pct(x0, pw),
                    "y": _pct(y0, ph),
                    "width": _pct(x1, pw) - _pct(x0, pw),
                    "height": _pct(y1, ph) - _pct(y0, ph),
                    "rotation": 0,
                    "rectanglelabels": ["table"],
                },
            })

        tasks.append({
            "data": {
                "image": f"/data/local-files/?d=render/{stem}/{img_name}",
                "pdf": stem,
                "page": i + 1,
                "n_pred": len(finals),
                "suspect": ",".join(sorted(reasons)) if reasons else "-",
            },
            "predictions": [{
                "model_version": "assay-pipeline",
                "result": result,
            }],
        })

        if (i + 1) % 50 == 0:
            print(f"      page {i + 1}/{doc.page_count}", flush=True)

    doc.close()

    out = tasks_dir / f"{stem}_tasks.json"
    out.write_text(json.dumps(tasks, ensure_ascii=False))
    print(f"\nDONE  {len(doc_out.tables)} predictions, "
          f"{n_suspect}/{len(tasks)} suspect pages "
          f"({time.time() - t0:.0f}s total)")
    print(f"tasks : {out}")
    print(f"images: {render_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_tasks.py <pdf_path>")
    main(sys.argv[1])
