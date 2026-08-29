#!/usr/bin/env python3
"""Characterise every WEAK match (TP but IoU < threshold).

For each weak pred/GT pair, print the signed per-edge offset d = pred - gt
(positive dL/dT => pred edge is further right/down => pred is SMALLER on that
side; positive dR/dB => pred edge is further right/down => pred is LARGER),
and dump the text lines sitting in the strip that GT includes but pred excludes
so the exact amputated content is visible.  Read-only; uses the pred cache and
the label PDFs, no GPU pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABEL_ROOT = Path("/home/homio/sda/1111/label")
EVAL_DIR = Path(__file__).resolve().parent
GT_PATH = EVAL_DIR / "ground_truth.json"
CACHE_DIR = EVAL_DIR / "pred_cache"
WEAK = 0.9


def _iou(a, b) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.7):
    pairs = sorted(((_iou(p, g), pi, gi)
                    for pi, p in enumerate(preds)
                    for gi, g in enumerate(gts)), reverse=True)
    up, ug, out = set(), set(), []
    for iou, pi, gi in pairs:
        if iou < thr:
            break
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi)
        out.append((iou, preds[pi], gts[gi]))
    return out


def lines_in_strip(page, x0, x1, ya, yb):
    """Text lines whose vertical centre lies in [ya,yb] and overlap [x0,x1]."""
    if yb - ya < 1.0:
        return []
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            lcy = (ly0 + ly1) / 2
            if ya - 0.5 <= lcy <= yb + 0.5 and lx1 > x0 and lx0 < x1:
                txt = "".join(s["text"] for s in line["spans"]).strip()
                out.append((round(ly0, 1), round(ly1, 1),
                            round(lx0, 1), round(lx1, 1), txt))
    return out


def main():
    gt = json.loads(GT_PATH.read_text())
    for stem, pages in sorted(gt.items()):
        docs = fitz.open(str(LABEL_ROOT / f"{stem}.pdf"))
        pred = json.loads((CACHE_DIR / f"{stem}.json").read_text())
        short = stem.split("_")[2] if "_" in stem else stem
        for page_no, gts in sorted(pages.items(), key=lambda kv: int(kv[0])):
            for iou, p, g in match(pred.get(page_no, []), gts):
                if iou >= WEAK:
                    continue
                dL, dT, dR, dB = (round(p[i] - g[i], 1) for i in range(4))
                page = docs[int(page_no) - 1]
                print(f"\n{short} p{page_no}  IoU={iou:.3f}  "
                      f"dL={dL:+} dT={dT:+} dR={dR:+} dB={dB:+}")
                # top strip GT keeps but pred drops (pred top below gt top)
                if p[1] - g[1] > 2:
                    for ln in lines_in_strip(page, g[0], g[2], g[1], p[1]):
                        print(f"    TOPcut {ln}")
                # bottom strip GT keeps but pred drops
                if g[3] - p[3] > 2:
                    for ln in lines_in_strip(page, g[0], g[2], p[3], g[3]):
                        print(f"    BOTcut {ln}")
        docs.close()


if __name__ == "__main__":
    main()
