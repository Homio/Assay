#!/usr/bin/env python3
"""
Classify eval failures (FP / MISS at a given IoU) into actionable buckets.

Buckets:
  split      -- N preds cover 1 GT box (pipeline fragmented a table)
  merge      -- 1 pred covers N GT boxes (pipeline glued tables together)
  bleed      -- FP/MISS pair with big 1:1 overlap but IoU below threshold
                (boundary drawn wrong, e.g. box swallowing next header)
  miss-empty -- missed GT box that is short (likely header-only table)
  miss-plain -- missed GT box, no ruling lines inside (borderless table)
  miss-ruled -- missed GT box that has ruling lines (detector plain miss)
  fp-lone    -- prediction with no GT box anywhere near (phantom table)

Usage:
    PYTHONPATH=<repo> python3 eval/bucket_failures.py [--iou 0.7]
Reads eval/ground_truth.json and eval/pred_cache/ (run eval.py first).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import fitz

LABEL_ROOT = Path("/home/homio/sda/1111/label")
EVAL_DIR = Path(__file__).resolve().parent


def _iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _cover(inner, outer):
    """Fraction of ``inner`` area inside ``outer``."""
    ix0 = max(inner[0], outer[0]); iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2]); iy1 = min(inner[3], outer[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix1 - ix0) * (iy1 - iy0) / area if area > 0 else 0.0


def _has_ruling(page, box):
    """True when the GT box contains horizontal ruling lines (bordered)."""
    n = 0
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > 50 and r.height < 4 \
                and r.x0 >= box[0] - 5 and r.x1 <= box[2] + 5 \
                and box[1] - 3 <= r.y0 <= box[3] + 3:
            n += 1
            if n >= 2:
                return True
    return False


def match_page(preds, gts, thr):
    pairs = sorted(((_iou(p, g), pi, gi)
                    for pi, p in enumerate(preds)
                    for gi, g in enumerate(gts)), reverse=True)
    used_p, used_g = set(), set()
    for iou, pi, gi in pairs:
        if iou < thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    fp = [pi for pi in range(len(preds)) if pi not in used_p]
    miss = [gi for gi in range(len(gts)) if gi not in used_g]
    return fp, miss


def bucket_page(page, preds, gts, fp_idx, miss_idx):
    """Yield (bucket, detail) for each failing box on this page."""
    fp_left, miss_left = set(fp_idx), set(miss_idx)

    # split: one missed GT covered >=80% by union of >=2 FP preds
    for gi in list(miss_left):
        g = gts[gi]
        parts = [pi for pi in fp_left if _cover(preds[pi], g) >= 0.6]
        if len(parts) >= 2:
            cov = sum(_cover(g, preds[pi]) * 0 + 0 for pi in parts)  # noqa
            union_cov = 0.0
            gy = sorted((max(preds[pi][1], g[1]), min(preds[pi][3], g[3]))
                        for pi in parts)
            covered, cur0, cur1 = 0.0, None, None
            for y0, y1 in gy:
                if cur1 is None or y0 > cur1:
                    if cur1 is not None:
                        covered += cur1 - cur0
                    cur0, cur1 = y0, y1
                else:
                    cur1 = max(cur1, y1)
            if cur1 is not None:
                covered += cur1 - cur0
            union_cov = covered / (g[3] - g[1]) if g[3] > g[1] else 0
            if union_cov >= 0.7:
                miss_left.discard(gi)
                fp_left -= set(parts)
                yield "split", f"gt#{gi} <- {len(parts)} preds"
                continue

    # merge: one FP pred covering >=2 missed GT boxes
    for pi in list(fp_left):
        p = preds[pi]
        eaten = [gi for gi in miss_left if _cover(gts[gi], p) >= 0.7]
        if len(eaten) >= 2:
            fp_left.discard(pi)
            miss_left -= set(eaten)
            yield "merge", f"pred#{pi} -> {len(eaten)} gts"
            continue

    # bleed: 1:1 FP/MISS pair with material overlap but IoU under thr
    for pi in list(fp_left):
        best_gi, best = None, 0.0
        for gi in miss_left:
            v = _iou(preds[pi], gts[gi])
            if v > best:
                best, best_gi = v, gi
        if best_gi is not None and best >= 0.30:
            fp_left.discard(pi)
            miss_left.discard(best_gi)
            yield "bleed", f"pred#{pi}~gt#{best_gi} IoU={best:.2f}"

    # leftover misses
    for gi in miss_left:
        g = gts[gi]
        h = g[3] - g[1]
        if h <= 30:
            yield "miss-empty", f"gt#{gi} h={h:.0f}pt"
        elif _has_ruling(page, g):
            yield "miss-ruled", f"gt#{gi} h={h:.0f}pt"
        else:
            yield "miss-plain", f"gt#{gi} h={h:.0f}pt"

    # leftover FPs
    for pi in fp_left:
        p = preds[pi]
        yield "fp-lone", f"pred#{pi} h={p[3] - p[1]:.0f}pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    gt = json.loads((EVAL_DIR / "ground_truth.json").read_text())
    grand = Counter()
    for stem, pages in sorted(gt.items()):
        cache = EVAL_DIR / "pred_cache" / f"{stem}.json"
        pred = json.loads(cache.read_text())
        doc = fitz.open(str(LABEL_ROOT / f"{stem}.pdf"))
        per = Counter()
        detail = defaultdict(list)
        for pno, gts in sorted(pages.items(), key=lambda kv: int(kv[0])):
            preds = pred.get(pno, [])
            fp, miss = match_page(preds, gts, args.iou)
            if not fp and not miss:
                continue
            page = doc[int(pno) - 1]
            for bucket, info in bucket_page(page, preds, gts, fp, miss):
                per[bucket] += 1
                detail[bucket].append(f"p{pno} {info}")
        doc.close()
        grand += per
        short = stem.split("_")[2][:8] or stem[:20]
        print(f"\n== {short} ==  {dict(per)}")
        for b in sorted(detail):
            for line in detail[b]:
                print(f"  [{b}] {line}")
    print(f"\n==== GRAND ====\n  {dict(grand)}")


if __name__ == "__main__":
    main()
