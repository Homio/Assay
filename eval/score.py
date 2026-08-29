#!/usr/bin/env python3
"""
Multi-dimensional pipeline scorer — the measurement backbone.

Reads the pipeline's JSON exports (no GPU needed) and produces:
  1. a console report (per-book × dimension, worst-N lists)
  2. a JSON snapshot under eval/runs/ recording every page/table verdict —
     the input to eval/diff.py for regression gating

Dimensions (see eval/metrics.py):
  box      — P/R/F1 + meanIoU of table_bbox vs GT (same matching as eval.py)
  coverage — cell-union area share of the GT box
  text     — bag-of-chars recall/precision vs native text inside the GT box
  struct   — diagnostic: geometric text rows vs SLANet grid rows

Books without GT (no entry in ground_truth.json) run in smoke mode:
GT-free invariants only (self-coverage, self text recall, empty-cell rate,
pred/pred overlap) so generalisation drift is visible without annotations.

Usage:
    PYTHONPATH=. python3 eval/score.py [--output-dir DIR] [--tag NAME]
                                       [--top N] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fitz

EVAL_DIR = Path(__file__).resolve().parent
GT_PATH = EVAL_DIR / "ground_truth.json"
RUNS_DIR = EVAL_DIR / "runs"
LABEL_ROOT = Path("/home/homio/sda/1111/label")
ORIGINAL_ROOT = Path("/home/homio/sda/1111/original")
DEFAULT_OUTPUT = Path("/home/homio/sda/1111/output")

sys.path.insert(0, str(EVAL_DIR))
from metrics import (  # noqa: E402
    COVERAGE_FLOOR, TEXT_FLOOR, WEAK_THRESHOLD,
    area, boxes_cover, cell_boxes, cell_stats, cell_text, count_text_rows,
    inter_area, match_page, overlap_pairs, table_box, table_page,
    text_scores, union_box,
)


# ── Per-book scoring ─────────────────────────────────────────────────────


def score_gt_book(stem: str, tables: list, gt_pages: dict, doc) -> dict:
    """Full three-dimension scoring for a book with ground truth."""
    by_page: dict = {}
    for t in tables:
        b = table_box(t)
        if b:
            by_page.setdefault(str(table_page(t) + 1), []).append((b, t))

    pages: dict = {}
    for pno in sorted(set(gt_pages) | set(by_page), key=int):
        gts = gt_pages.get(pno, [])
        entries = by_page.get(pno, [])
        preds = [b for b, _ in entries]
        matches, fp, miss = match_page(preds, gts)
        page = doc[int(pno) - 1] if doc is not None else None

        match_recs = []
        for gi, pi, v in matches:
            b, t = entries[pi]
            cells = cell_boxes(t)
            u = union_box(cells)
            g = gts[gi]
            cov = inter_area(u, g) / area(g) if u and area(g) > 0 else 0.0
            trec = tprec = None
            geo = None
            if page is not None:
                ref = page.get_text("text", clip=fitz.Rect(*g))
                trec, tprec = text_scores(ref, cell_text(t))
                geo = count_text_rows(page, b)
            match_recs.append({
                "g": gi, "p": pi,
                "iou": round(v, 4), "cov": round(cov, 4),
                "trec": round(trec, 4) if trec is not None else None,
                "tprec": round(tprec, 4) if tprec is not None else None,
                "nr": t["data"]["num_rows"], "nc": t["data"]["num_cols"],
                "geo_rows": geo,
            })
        pages[pno] = {
            "matches": match_recs,
            "fp": fp and [b for _, b in fp] or [],
            "miss": miss and [b for _, b in miss] or [],
            "overlap": overlap_pairs(preds),
        }
    return pages


def score_smoke_book(stem: str, tables: list, doc) -> dict:
    """GT-free invariants for an unannotated book."""
    recs = []
    page_boxes: dict = {}
    for t in tables:
        b = table_box(t)
        if not b:
            continue
        pno = table_page(t)
        page_boxes.setdefault(pno, []).append(b)
        cells = cell_boxes(t)
        self_cov = boxes_cover(cells, b)
        total, empty = cell_stats(t)
        trec = None
        if doc is not None:
            ref = doc[pno].get_text("text", clip=fitz.Rect(*b))
            trec, _ = text_scores(ref, cell_text(t))
        recs.append({
            "page": pno + 1, "bbox": [round(v, 1) for v in b],
            "self_cov": round(self_cov, 4),
            "trec": round(trec, 4) if trec is not None else None,
            "empty_rate": round(empty / total, 4) if total else 0.0,
            "nr": t["data"]["num_rows"], "nc": t["data"]["num_cols"],
        })
    overlaps = sum(overlap_pairs(bs) for bs in page_boxes.values())
    return {"tables": recs, "overlap": overlaps}


# ── Aggregation ──────────────────────────────────────────────────────────


def aggregate(pages: dict) -> dict:
    tp = sum(len(p["matches"]) for p in pages.values())
    fp = sum(len(p["fp"]) for p in pages.values())
    miss = sum(len(p["miss"]) for p in pages.values())
    ious = [m["iou"] for p in pages.values() for m in p["matches"]]
    covs = [m["cov"] for p in pages.values() for m in p["matches"]]
    trecs = [m["trec"] for p in pages.values() for m in p["matches"]
             if m["trec"] is not None]
    weak = sum(1 for v in ious if v < WEAK_THRESHOLD)
    overlap = sum(p["overlap"] for p in pages.values())
    p_ = tp / (tp + fp) if tp + fp else 1.0
    r_ = tp / (tp + miss) if tp + miss else 1.0
    return {
        "tp": tp, "fp": fp, "miss": miss, "weak": weak, "overlap": overlap,
        "P": round(p_, 4), "R": round(r_, 4),
        "F1": round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0.0,
        "mIoU": round(sum(ious) / len(ious), 4) if ious else 1.0,
        "cov_mean": round(sum(covs) / len(covs), 4) if covs else 1.0,
        "cov_lt90": sum(1 for c in covs if c < COVERAGE_FLOOR),
        "trec_mean": round(sum(trecs) / len(trecs), 4) if trecs else None,
        "trec_lt95": sum(1 for c in trecs if c < TEXT_FLOOR),
    }


# ── Console report ───────────────────────────────────────────────────────


def _worst(pages: dict, key: str, n: int, reverse: bool = False):
    recs = [(m[key], pno, m) for pno, p in pages.items()
            for m in p["matches"] if m.get(key) is not None]
    recs.sort(key=lambda x: x[0], reverse=reverse)
    return recs[:n]


def print_book(name: str, pages: dict, agg: dict, top: int) -> None:
    fails = [f"p{pno}: +{len(p['fp'])}/-{len(p['miss'])}"
             for pno, p in sorted(pages.items(), key=lambda kv: int(kv[0]))
             if p["fp"] or p["miss"]]
    print(f"  {name}")
    print(f"    box  TP={agg['tp']} FP={agg['fp']} MISS={agg['miss']} "
          f"WEAK={agg['weak']} OVERLAP={agg['overlap']} | "
          f"P={agg['P']:.3f} R={agg['R']:.3f} F1={agg['F1']:.3f} "
          f"mIoU={agg['mIoU']:.3f}")
    print(f"    cov  mean={agg['cov_mean']:.3f} <90%: {agg['cov_lt90']} | "
          f"text recall mean={agg['trec_mean']:.3f} <95%: {agg['trec_lt95']}"
          if agg['trec_mean'] is not None else "    cov/text: no PDF")
    if fails:
        print(f"    fails: {' '.join(fails)}")
    for label, key in (("coverage", "cov"), ("text", "trec")):
        worst = _worst(pages, key, top)
        worst = [w for w in worst
                 if w[0] < (COVERAGE_FLOOR if key == "cov" else TEXT_FLOOR)]
        if worst:
            items = " ".join(f"p{pno}={v:.2f}" for v, pno, _ in worst)
            print(f"    worst {label}: {items}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--tag", default=None,
                    help="snapshot name (default: timestamp)")
    ap.add_argument("--top", type=int, default=10, help="worst-N list size")
    ap.add_argument("--quiet", action="store_true", help="snapshot only")
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text()) if GT_PATH.exists() else {}
    out_dir = Path(args.output_dir)
    stems = sorted(d.name for d in out_dir.iterdir()
                   if d.is_dir() and (d / f"{d.name}.json").exists())
    if not stems:
        sys.exit(f"no pipeline outputs found under {out_dir}")

    snapshot = {
        "meta": {
            "tag": args.tag or time.strftime("%Y%m%d_%H%M%S"),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(out_dir),
            "code_revs": {},
        },
        "books": {},
    }
    overall_pages: dict = {}

    for stem in stems:
        out = json.loads((out_dir / stem / f"{stem}.json").read_text())
        tables = out.get("tables", [])
        snapshot["meta"]["code_revs"][stem] = out.get("code_rev")
        pdf = None
        for root in (LABEL_ROOT, ORIGINAL_ROOT):
            if (root / f"{stem}.pdf").exists():
                pdf = fitz.open(str(root / f"{stem}.pdf"))
                break
        name = stem.split("_")[2] if len(stem.split("_")) > 2 else stem

        if stem in gt:
            pages = score_gt_book(stem, tables, gt[stem], pdf)
            agg = aggregate(pages)
            snapshot["books"][stem] = {
                "mode": "gt", "aggregate": agg, "pages": pages}
            overall_pages[stem] = pages
            if not args.quiet:
                print_book(name, pages, agg, args.top)
        else:
            smoke = score_smoke_book(stem, tables, pdf)
            snapshot["books"][stem] = {"mode": "smoke", "smoke": smoke}
            if not args.quiet:
                n = len(smoke["tables"])
                covs = [t["self_cov"] for t in smoke["tables"]]
                trs = [t["trec"] for t in smoke["tables"]
                       if t["trec"] is not None]
                print(f"  {name} [smoke] tables={n} "
                      f"self_cov_mean={sum(covs)/len(covs):.3f} "
                      f"self_cov<90%: {sum(1 for c in covs if c < COVERAGE_FLOOR)} "
                      f"trec_mean={sum(trs)/len(trs):.3f} "
                      f"overlap={smoke['overlap']}"
                      if trs else f"  {name} [smoke] tables={n}")
        if pdf is not None:
            pdf.close()

    if overall_pages:
        # Key by book+page: page numbers collide across books.
        merged = {f"{stem}:{pno}": p for stem, pages in overall_pages.items()
                  for pno, p in pages.items()}
        o = aggregate(merged)
        print(f"\n==== OVERALL (GT books) ====")
        print(f"  box  TP={o['tp']} FP={o['fp']} MISS={o['miss']} "
              f"WEAK={o['weak']} OVERLAP={o['overlap']} | "
              f"P={o['P']:.3f} R={o['R']:.3f} F1={o['F1']:.3f} "
              f"mIoU={o['mIoU']:.3f}")
        print(f"  cov  mean={o['cov_mean']:.3f} <90%: {o['cov_lt90']} | "
              f"text recall mean={o['trec_mean']:.3f} "
              f"<95%: {o['trec_lt95']}")
        snapshot["overall"] = o

    RUNS_DIR.mkdir(exist_ok=True)
    snap_path = RUNS_DIR / f"{snapshot['meta']['tag']}.json"
    snap_path.write_text(json.dumps(snapshot, ensure_ascii=False))
    print(f"\nsnapshot: {snap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
