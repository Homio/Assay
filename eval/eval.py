#!/usr/bin/env python3
"""
Score the pipeline against eval/ground_truth.json.

For every reviewed page, greedy IoU matching between predicted and GT boxes
(highest IoU pair first).  A prediction matching a GT box at IoU >= the
threshold is a true positive; leftover predictions are false positives;
leftover GT boxes are misses.  Reports per-PDF and overall precision /
recall / F1 plus mean IoU of the matches (localisation quality), and lists
every failing page so a regression is traceable to the exact page.

Usage:
    PYTHONPATH=<repo> python3 eval/eval.py [--iou 0.7] [--pdf <stem_substr>]

Predictions are produced by running the real pipeline (AssayExtractor) on
each PDF found in the GT; per-PDF results are cached to eval/pred_cache/ so
re-scoring after an annotation update needs no GPU pass (delete the cache
after changing pipeline code).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABEL_ROOT = Path("/home/homio/sda/1111/label")
EVAL_DIR = Path(__file__).resolve().parent
GT_PATH = EVAL_DIR / "ground_truth.json"
CACHE_DIR = EVAL_DIR / "pred_cache"


def _iou(a, b) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def get_predictions(stem: str) -> dict:
    """Per-page predicted boxes {viewer_page(str): [[x0,y0,x1,y1],...]}."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{stem}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    pdf = LABEL_ROOT / f"{stem}.pdf"
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")
    print(f"  running pipeline on {stem} ...", flush=True)
    from assay.models import TableModel
    from assay.extractor import AssayExtractor
    doc_out = AssayExtractor(TableModel(use_gpu=True)).extract(str(pdf))
    pred: dict = {}
    for t in doc_out.tables:
        pred.setdefault(str(t.page_no + 1), []).append(
            [round(v, 2) for v in t.table_bbox])
    cache.write_text(json.dumps(pred))
    return pred


def match_page(preds, gts, thr):
    """Greedy best-IoU matching; returns (tp_ious, fp_boxes, miss_boxes)."""
    pairs = sorted(
        ((_iou(p, g), pi, gi)
         for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        reverse=True)
    used_p, used_g, tp = set(), set(), []
    for iou, pi, gi in pairs:
        if iou < thr:
            break
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp.append(iou)
    fp = [p for pi, p in enumerate(preds) if pi not in used_p]
    miss = [g for gi, g in enumerate(gts) if gi not in used_g]
    return tp, fp, miss


def pred_overlaps(preds, frac=0.05):
    """Pairs of predicted boxes on one page that materially overlap.

    Two genuine tables never share page area, so any pred/pred overlap
    above a border-touch tolerance is a physical impossibility -- one of
    the boxes bled into the other -- and must surface even when both
    boxes still clear the TP threshold against their GT partners."""
    out = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i], preds[j]
            ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
            ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            lo = min((a[2] - a[0]) * (a[3] - a[1]),
                     (b[2] - b[0]) * (b[3] - b[1]))
            if lo > 0 and inter / lo >= frac:
                out.append((inter / lo, a, b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--weak", type=float, default=0.9,
                    help="flag TP matches below this IoU as boundary bleed")
    ap.add_argument("--pdf", default="", help="only PDFs whose stem contains this")
    args = ap.parse_args()

    if not GT_PATH.exists():
        sys.exit("no ground_truth.json yet -- run import_annotations.py first")
    gt = json.loads(GT_PATH.read_text())

    grand = {"tp": 0, "fp": 0, "miss": 0, "iou_sum": 0.0,
             "weak": 0, "overlap": 0}
    for stem, pages in sorted(gt.items()):
        if args.pdf and args.pdf not in stem:
            continue
        pred = get_predictions(stem)
        stat = {"tp": 0, "fp": 0, "miss": 0, "iou_sum": 0.0}
        problems, weak, overlaps = [], [], []
        for page_no, gt_boxes in sorted(pages.items(), key=lambda kv: int(kv[0])):
            page_preds = pred.get(page_no, [])
            tp, fp, miss = match_page(page_preds, gt_boxes, args.iou)
            stat["tp"] += len(tp); stat["fp"] += len(fp)
            stat["miss"] += len(miss); stat["iou_sum"] += sum(tp)
            if fp or miss:
                problems.append((int(page_no), len(fp), len(miss)))
            weak.extend((int(page_no), i) for i in tp if i < args.weak)
            overlaps.extend((int(page_no), frac)
                            for frac, _, _ in pred_overlaps(page_preds))
        p = stat["tp"] / (stat["tp"] + stat["fp"]) if stat["tp"] + stat["fp"] else 0
        r = stat["tp"] / (stat["tp"] + stat["miss"]) if stat["tp"] + stat["miss"] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        miou = stat["iou_sum"] / stat["tp"] if stat["tp"] else 0
        print(f"\n== {stem} ==")
        print(f"  pages={len(pages)}  TP={stat['tp']}  FP={stat['fp']}  "
              f"MISS={stat['miss']}")
        print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  meanIoU={miou:.3f}  "
              f"@IoU>={args.iou}")
        if problems:
            print("  failing pages (page: +extra / -missed):")
            for pg, nfp, nms in problems:
                print(f"    p{pg}: +{nfp} / -{nms}")
        if overlaps:
            print("  OVERLAPPING pred boxes (physically impossible):")
            for pg, frac in sorted(overlaps):
                print(f"    p{pg}: {frac:.0%} of smaller box")
        if weak:
            print(f"  weak matches (TP but IoU < {args.weak}):")
            for pg, i in sorted(weak):
                print(f"    p{pg}: IoU={i:.3f}")
        for k in ("tp", "fp", "miss", "iou_sum"):
            grand[k] += stat[k]
        grand["weak"] += len(weak)
        grand["overlap"] += len(overlaps)

    tp, fp, ms = grand["tp"], grand["fp"], grand["miss"]
    if tp + fp + ms:
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + ms) if tp + ms else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"\n==== OVERALL ====")
        print(f"  TP={tp}  FP={fp}  MISS={ms}  "
              f"WEAK={grand['weak']}  OVERLAP={grand['overlap']}")
        print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
              f"meanIoU={grand['iou_sum'] / tp if tp else 0:.3f}  "
              f"@IoU>={args.iou}")


if __name__ == "__main__":
    main()
