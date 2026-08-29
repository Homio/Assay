#!/usr/bin/env python3
"""
Repair over-split table pre-annotations: merge boxes the pipeline cut out of a
single table back into one rectangle, then (optionally) push the corrected
rectangles into a Label Studio project as *predictions* only.

Why this is safe to automate
-----------------------------
The pipeline sometimes slices ONE table into two vertically-stacked boxes of
the same width.  Geometry alone cannot tell an over-split apart from two real
tables that happen to sit on top of each other -- the user's ground-truth
policy is: every table that carries *its own header row* is an independent box.
So we merge a stacked pair only when the LOWER box has NO header of its own,
i.e. its first value-area row is numeric (a pure data continuation).  That is
the ``first_valuerow_numeric_frac`` content test below.

Merge conditions for an (upper A, lower B) pair, all in PDF points:
  * left/right edges aligned within ALIGN_PT
  * vertical gap  B.y0 - A.y1  in [-GAP_OVERLAP, GAP_MAX]
  * first_valuerow_numeric_frac(B) >= NUM_THRESH   (B is data, not a new table)
Merges are transitive: header + data + data collapses to one box.

We are deliberately CONSERVATIVE: a missed merge just leaves the user a manual
merge (no regression), whereas a false merge would fuse two real tables.

Usage
-----
  # dry run over every page, print the plan
  python3 eval/merge_predictions.py <stem>
  # dry run + render before/after PNGs for affected pages to /tmp/cnpc_merge
  python3 eval/merge_predictions.py <stem> --render
  # push merges into LS project as predictions (skips already-annotated pages)
  python3 eval/merge_predictions.py <stem> --apply --project 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz

LABEL_ROOT = Path("/home/homio/sda/1111/label")
from ls_auth import base_url as _ls_base, get_session as _ls_get_session
BASE = _ls_base()

# geometry / content thresholds
ALIGN_PT = 12.0        # left & right edge tolerance (~2% of an A4 width)
GAP_MAX = 30.0         # max whitespace between stacked boxes to still merge
GAP_OVERLAP = 4.0      # allow a few points of vertical overlap

# A "data value" is any numeric token that is NOT a bare year.  Years (2024,
# 2023, ...) are excluded because that is how header rows label period columns
# ("2024年末", "2023年度") and would otherwise make a header look like data.
# Everything else -- amounts (2,937,981), percentages (93.21), parenthesised
# negatives ((1,836)) and small integers such as ranks or share counts (2, 0)
# -- is genuine tabular data.
NUMISH = re.compile(r'^[\(\-]?\d[\d,]*(\.\d+)?\)?%?$')
YEAR = re.compile(r'^(19|20)\d{2}$')


def is_data_value(tok: str) -> bool:
    return bool(NUMISH.match(tok)) and not YEAR.match(tok)


# ── coordinate helpers ──────────────────────────────────────────────────────
def pct_to_pts(v: dict, pw: float, ph: float):
    x0 = v["x"] / 100 * pw
    y0 = v["y"] / 100 * ph
    x1 = (v["x"] + v["width"]) / 100 * pw
    y1 = (v["y"] + v["height"]) / 100 * ph
    return [x0, y0, x1, y1]


def pts_to_pct(box, pw: float, ph: float):
    x0, y0, x1, y1 = box
    return {
        "x": x0 / pw * 100,
        "y": y0 / ph * 100,
        "width": (x1 - x0) / pw * 100,
        "height": (y1 - y0) / ph * 100,
        "rotation": 0,
        "rectanglelabels": ["table"],
    }


# ── content test: is the lower box a headerless data continuation? ───────────
def _first_row_tokens(page, box, left_frac, row_tol):
    """Tokens of the topmost text row inside ``box``.  If ``left_frac`` > 0 the
    row is taken only from the box's right value-area slab (past ``left_frac``
    of the width), which sidesteps a tall wrapped label cell on the left whose
    first visual line carries no number."""
    x0, y0, x1, y1 = box
    vx0 = x0 + left_frac * (x1 - x0)
    words = [w for w in page.get_text("words", clip=fitz.Rect(*box))
             if (w[0] + w[2]) / 2 >= vx0 and w[4].strip()]
    if not words:
        return []
    words.sort(key=lambda w: w[1])
    top = words[0][1]
    return [w[4].strip() for w in words if w[1] <= top + row_tol]


def lower_box_is_continuation(page, box, row_tol=7.0):
    """True when ``box`` (the lower box of a stacked pair) begins with a DATA
    row rather than a header row -- i.e. it is a headerless continuation of the
    table above and the two should be merged.

    A header row is pure column titles: text, plus at most year/period labels.
    A data row carries at least one real data value.  We test two candidate
    first rows and merge if either looks like data:
      * the full-width top row  -- catches tables whose numbers sit in a middle
        column and whose right-most columns are text (e.g. a bond table ending
        in "付息方式 / 交易场所"), and rank/count columns on the left;
      * the value-area top row   -- catches tables whose left column is a tall
        wrapped label cell (e.g. a fund's full name) so the numbers only show
        up once we skip past it.
    """
    full = _first_row_tokens(page, box, 0.0, row_tol)
    if any(is_data_value(t) for t in full):
        return True
    val = _first_row_tokens(page, box, 0.35, row_tol)
    return any(is_data_value(t) for t in val)


def _aligned(a, b):
    return abs(a[0] - b[0]) <= ALIGN_PT and abs(a[2] - b[2]) <= ALIGN_PT


def merge_groups(page, boxes):
    """boxes: list of [x0,y0,x1,y1] in PDF points, any order.
    Returns list of index-lists; singletons included.  Adjacent aligned pairs
    whose lower box is a data continuation are chained together."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1])
    parent = {i: i for i in order}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(order, order[1:]):
        A, B = boxes[a], boxes[b]
        if not _aligned(A, B):
            continue
        gap = B[1] - A[3]
        if gap < -GAP_OVERLAP or gap > GAP_MAX:
            continue
        if lower_box_is_continuation(page, B):
            parent[find(b)] = find(a)

    groups: dict = {}
    for i in order:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def bbox_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ── per-page planning ────────────────────────────────────────────────────────
def plan_page(page, result):
    """Given a page's LS prediction ``result`` list, return (new_result, merges)
    where merges is a list of (n_boxes_merged,) group sizes >1."""
    pw, ph = page.rect.width, page.rect.height
    boxes = [pct_to_pts(r["value"], pw, ph) for r in result]
    groups = merge_groups(page, boxes)
    merges = [g for g in groups if len(g) > 1]
    if not merges:
        return None, []
    new_result = []
    for gi, g in enumerate(groups):
        union = bbox_union([boxes[i] for i in g])
        new_result.append({
            "id": f"m{gi}",
            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
            "original_width": result[g[0]]["original_width"],
            "original_height": result[g[0]]["original_height"],
            "value": pts_to_pct(union, pw, ph),
        })
    return new_result, [len(g) for g in merges]


def build_plan(stem):
    tasks = json.loads((LABEL_ROOT / "tasks" / f"{stem}_tasks.json").read_text())
    doc = fitz.open(str(LABEL_ROOT / f"{stem}.pdf"))
    plan = {}  # page -> (old_result, new_result, group_sizes)
    for t in tasks:
        page_no = int(t["data"]["page"])
        result = t["predictions"][0]["result"]
        if len(result) < 2:
            continue
        new_result, sizes = plan_page(doc[page_no - 1], result)
        if new_result is not None:
            plan[page_no] = (result, new_result, sizes)
    return doc, plan


# ── render before/after ──────────────────────────────────────────────────────
def render_pair(doc, page_no, old_result, new_result, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    page = doc[page_no - 1]
    pw, ph = page.rect.width, page.rect.height
    mat = fitz.Matrix(2, 2)
    for tag, res, color in (("before", old_result, (1, 0, 0)),
                            ("after", new_result, (0, 0.5, 1))):
        pix = page.get_pixmap(matrix=mat)
        d = fitz.open("pdf", pix.pdfocr_tobytes()) if False else None
        # draw on a fresh single-page pdf built from the pixmap image
        img = fitz.open()
        p = img.new_page(width=pix.width, height=pix.height)
        p.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
        for r in res:
            b = pct_to_pts(r["value"], pw, ph)
            rect = fitz.Rect(b[0] * 2, b[1] * 2, b[2] * 2, b[3] * 2)
            p.draw_rect(rect, color=color, width=3)
        p.get_pixmap().save(str(outdir / f"p{page_no:04d}_{tag}.png"))
        img.close()


# ── LS apply ─────────────────────────────────────────────────────────────────
def ls_session():
    return _ls_get_session()


def apply_to_ls(plan, project):
    s = ls_session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}
    # map page -> task
    tasks = []
    page = 1
    while True:
        r = s.get(f"{BASE}/api/tasks?project={project}&page_size=200&page={page}",
                  headers=h)
        chunk = r.json().get("tasks", [])
        if not chunk:
            break
        tasks.extend(chunk)
        page += 1
    by_page = {int(t["data"]["page"]): t for t in tasks}

    applied = skipped_labeled = 0
    for page_no, (_old, new_result, _sizes) in sorted(plan.items()):
        t = by_page.get(page_no)
        if t is None:
            print(f"  p{page_no}: no task on server, skip")
            continue
        if t.get("is_labeled") or t.get("total_annotations", 0) > 0:
            print(f"  p{page_no}: already annotated by user -> SKIP")
            skipped_labeled += 1
            continue
        pr = s.get(f"{BASE}/api/predictions?task={t['id']}", headers=h).json()
        preds = pr if isinstance(pr, list) else pr.get("results", pr)
        if not preds:
            print(f"  p{page_no}: no prediction to patch, skip")
            continue
        pid = preds[0]["id"]
        rp = s.patch(f"{BASE}/api/predictions/{pid}/", headers=h,
                     json={"result": new_result})
        rp.raise_for_status()
        applied += 1
        print(f"  p{page_no}: prediction {pid} updated")
    print(f"\napplied={applied}  skipped_labeled={skipped_labeled}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--project", type=int, default=0)
    ap.add_argument("--outdir", default="/tmp/cnpc_merge")
    args = ap.parse_args()

    doc, plan = build_plan(args.stem)
    total_before = sum(len(v[0]) for v in plan.values())
    total_after = sum(len(v[1]) for v in plan.values())
    print(f"pages with a merge: {len(plan)}")
    for page_no, (old, new, sizes) in sorted(plan.items()):
        print(f"  p{page_no}: {len(old)} -> {len(new)} boxes "
              f"(merged groups of {sizes})")
    print(f"\ntotal boxes on affected pages: {total_before} -> {total_after} "
          f"(removed {total_before - total_after})")

    if args.render:
        outdir = Path(args.outdir)
        for page_no, (old, new, _sizes) in sorted(plan.items()):
            render_pair(doc, page_no, old, new, outdir)
        print(f"rendered before/after to {outdir}")

    if args.apply:
        if not args.project:
            sys.exit("--apply requires --project <id>")
        print(f"\napplying to LS project {args.project} ...")
        apply_to_ls(plan, args.project)


if __name__ == "__main__":
    main()
