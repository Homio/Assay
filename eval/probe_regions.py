#!/usr/bin/env python3
"""
Probe helper for the batch-2 review fixes (紫光股份 / 一彬科技).

For every requested page it
  1. clusters the PDF vector drawings into table-frame regions
     (validated on 一彬 p139: rect union with gap<=6pt, keep w>100 & h>8pt),
  2. pulls the live prediction boxes from Label Studio,
  3. prints regions vs boxes and flags regions not covered by any box.

The output is *evidence for the human-in-the-loop*, the actual FIXES table in
apply_fixes_batch2.py is still written by eyeballing this against the
verdicts.json notes - no silent auto-patching.

Usage
    python3 eval/probe_regions.py <stem> --project N --pages 139,143,...
"""

import argparse
import json
import sys
from pathlib import Path

import fitz
import requests

from ls_auth import base_url as _ls_base, get_session as _ls_get_session
BASE = _ls_base()
LABEL_DIR = Path("/home/homio/sda/1111/label")
REVIEW_DIR = Path("/home/homio/sda/1111/output/review")


def ls_session():
    return _ls_get_session()


def frame_regions(page, gap=6.0, min_w=100.0, min_h=8.0):
    """Cluster vector rectangles into table-frame regions (page pt coords)."""
    rects = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.width <= 0 and r.height <= 0:
            continue
        rects.append([r.x0, r.y0, r.x1, r.y1])
    changed = True
    while changed:
        changed = False
        out = []
        for r in rects:
            hit = None
            for o in out:
                if (r[0] - gap <= o[2] and o[0] - gap <= r[2] and
                        r[1] - gap <= o[3] and o[1] - gap <= r[3]):
                    hit = o
                    break
            if hit:
                hit[0] = min(hit[0], r[0]); hit[1] = min(hit[1], r[1])
                hit[2] = max(hit[2], r[2]); hit[3] = max(hit[3], r[3])
                changed = True
            else:
                out.append(list(r))
        rects = out
    return sorted([r for r in rects
                   if r[2] - r[0] > min_w and r[3] - r[1] > min_h],
                  key=lambda r: r[1])


def to_pct(r, pw, ph):
    return (round(r[0] / pw * 100, 2), round(r[1] / ph * 100, 2),
            round(r[2] / pw * 100, 2), round(r[3] / ph * 100, 2))


def covered(reg, boxes):
    """Fraction of region area covered by the best-overlapping box (pct)."""
    x0, y0, x1, y1 = reg
    best = 0.0
    for b in boxes:
        bx0, by0 = b["x"], b["y"]
        bx1, by1 = b["x"] + b["width"], b["y"] + b["height"]
        iw = min(x1, bx1) - max(x0, bx0)
        ih = min(y1, by1) - max(y0, by0)
        if iw > 0 and ih > 0:
            area = (x1 - x0) * (y1 - y0)
            best = max(best, iw * ih / area if area else 0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--pages", required=True)
    args = ap.parse_args()

    pages = [int(p) for p in args.pages.split(",")]
    man = json.loads((REVIEW_DIR / args.stem / "manifest.json").read_text())
    doc = fitz.open(LABEL_DIR / f"{args.stem}.pdf")
    s = ls_session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}

    for pn in pages:
        m = man[str(pn)]
        pr = s.get(f"{BASE}/api/predictions?task={m['task_id']}",
                   headers=h).json()
        preds = pr if isinstance(pr, list) else pr.get("results", pr)
        boxes = [it["value"] for it in preds[0]["result"]] if preds else []
        pg = doc[pn - 1]
        pw, ph = pg.rect.width, pg.rect.height
        regs = [to_pct(r, pw, ph) for r in frame_regions(pg)]
        print(f"\n=== p{pn}  boxes={len(boxes)} regions={len(regs)}")
        for i, b in enumerate(boxes):
            print(f"  box#{i}: x={b['x']:.2f} y={b['y']:.2f} "
                  f"w={b['width']:.2f} h={b['height']:.2f} "
                  f"(y1={b['y']+b['height']:.2f})")
        for r in regs:
            cov = covered(r, boxes)
            flag = "MISS" if cov < 0.5 else ("part" if cov < 0.95 else "ok  ")
            print(f"  reg {flag} x0={r[0]:6.2f} y0={r[1]:6.2f} "
                  f"x1={r[2]:6.2f} y1={r[3]:6.2f} cov={cov:.2f}")


if __name__ == "__main__":
    main()
