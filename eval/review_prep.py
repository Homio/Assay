#!/usr/bin/env python3
"""
Build a page-by-page visual review workbench for one PDF's Label Studio
pre-annotations.

Why this exists
---------------
Heuristics cannot judge whether a pre-drawn rectangle is a *sensible* table
box.  That needs eyes on the page: does the box start at the header row, does it
end at the last data row, does it swallow a paragraph, are two real tables fused
into one box, is a whole table missing a box at all?  So we dump, for every
page, the page image with the CURRENT server-side prediction rectangles drawn
and numbered, plus a machine-readable manifest of those rectangles.  A reviewer
(human or model) then looks at each image and records a verdict.

Outputs (under OUT/<stem>/)
    manifest.json         page -> {task_id, pred_id, is_labeled, n_anno, boxes}
    pages/pXXXX.png       full-resolution overlay, one per page
    sheets/sXX.png        contact sheets (SHEET_N pages each) for fast triage
    verdicts.json         created empty; the review log, appended to over time

Usage
    python3 eval/review_prep.py <stem> --project 4
    python3 eval/review_prep.py <stem> --project 4 --only 34,51,97   # re-render
    python3 eval/review_prep.py <stem> --project 4 --no-sheets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

LABEL_ROOT = Path("/home/homio/sda/1111/label")
OUT = Path("/home/homio/sda/1111/output/review")
from ls_auth import base_url as _ls_base, get_session as _ls_get_session
BASE = _ls_base()

ZOOM = 2.0          # full-res overlay: A4 -> ~1190 px wide
SHEET_COLS = 3
SHEET_ROWS = 2
SHEET_N = SHEET_COLS * SHEET_ROWS
SHEET_ZOOM = 0.62   # per-tile zoom on the contact sheet


# ── Label Studio ────────────────────────────────────────────────────────────
def ls_session():
    return _ls_get_session()


def fetch_manifest(project: int) -> dict:
    """page -> task/prediction state, straight from the server."""
    s = ls_session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}

    tasks, page = [], 1
    while True:
        r = s.get(f"{BASE}/api/tasks?project={project}&page_size=200&page={page}",
                  headers=h)
        chunk = r.json().get("tasks", [])
        if not chunk:
            break
        tasks.extend(chunk)
        page += 1

    man = {}
    for t in tasks:
        pn = int(t["data"]["page"])
        pr = s.get(f"{BASE}/api/predictions?task={t['id']}", headers=h).json()
        preds = pr if isinstance(pr, list) else pr.get("results", pr)
        boxes, pred_id = [], None
        if preds:
            pred_id = preds[0]["id"]
            for r in preds[0]["result"]:
                v = r["value"]
                boxes.append([v["x"], v["y"], v["width"], v["height"]])
        man[str(pn)] = {
            "task_id": t["id"],
            "pred_id": pred_id,
            "is_labeled": bool(t.get("is_labeled")),
            "n_anno": t.get("total_annotations", 0),
            "suspect": t["data"].get("suspect", "-"),
            "boxes": boxes,
        }
        if len(man) % 40 == 0:
            print(f"  fetched {len(man)} pages", flush=True)
    return man


# ── drawing ─────────────────────────────────────────────────────────────────
def _pct_rect(b, pw, ph, zoom):
    x, y, w, hgt = b
    return fitz.Rect(x / 100 * pw * zoom, y / 100 * ph * zoom,
                     (x + w) / 100 * pw * zoom, (y + hgt) / 100 * ph * zoom)


def overlay_page(doc, pn, boxes, zoom):
    """Return a pixmap of page ``pn`` with numbered prediction boxes drawn."""
    page = doc[pn - 1]
    pw, ph = page.rect.width, page.rect.height
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = fitz.open()
    p = img.new_page(width=pix.width, height=pix.height)
    p.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
    for i, b in enumerate(boxes):
        rect = _pct_rect(b, pw, ph, zoom)
        p.draw_rect(rect, color=(1, 0, 0), width=2)
        tag = fitz.Rect(rect.x0, max(0, rect.y0 - 15), rect.x0 + 30, rect.y0)
        p.draw_rect(tag, color=(1, 0, 0), fill=(1, 1, 1), width=1)
        p.insert_text((rect.x0 + 4, rect.y0 - 4), f"#{i}",
                      fontsize=11, color=(1, 0, 0))
    p.insert_text((6, 14), f"p{pn}  boxes={len(boxes)}",
                  fontsize=12, color=(0, 0.4, 0))
    out = p.get_pixmap()
    img.close()
    return out


def contact_sheet(doc, pages, man, path):
    """SHEET_ROWS x SHEET_COLS grid of small overlays, for fast triage."""
    tiles = [overlay_page(doc, pn, man[str(pn)]["boxes"], SHEET_ZOOM)
             for pn in pages]
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    img = fitz.open()
    p = img.new_page(width=tw * SHEET_COLS, height=th * SHEET_ROWS)
    for k, t in enumerate(tiles):
        r, c = divmod(k, SHEET_COLS)
        p.insert_image(fitz.Rect(c * tw, r * th, c * tw + t.width,
                                 r * th + t.height), pixmap=t)
        p.draw_rect(fitz.Rect(c * tw, r * th, (c + 1) * tw, (r + 1) * th),
                    color=(0.7, 0.7, 0.7), width=1)
    p.get_pixmap().save(str(path))
    img.close()


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--only", default="", help="comma page list to re-render")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--refetch", action="store_true",
                    help="re-pull manifest from LS even if cached")
    args = ap.parse_args()

    root = OUT / args.stem
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "sheets").mkdir(parents=True, exist_ok=True)
    man_path = root / "manifest.json"

    if man_path.exists() and not args.refetch:
        man = json.loads(man_path.read_text())
        print(f"manifest (cached): {len(man)} pages")
    else:
        print(f"fetching manifest from LS project {args.project} ...")
        man = fetch_manifest(args.project)
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print(f"manifest: {len(man)} pages -> {man_path}")

    doc = fitz.open(str(LABEL_ROOT / f"{args.stem}.pdf"))
    all_pages = sorted(int(k) for k in man)
    todo = ([int(x) for x in args.only.split(",") if x.strip()]
            if args.only else all_pages)

    for pn in todo:
        pix = overlay_page(doc, pn, man[str(pn)]["boxes"], ZOOM)
        pix.save(str(root / "pages" / f"p{pn:04d}.png"))
        if pn % 40 == 0:
            print(f"  rendered p{pn}", flush=True)
    print(f"pages rendered: {len(todo)} -> {root/'pages'}")

    if not args.no_sheets and not args.only:
        for i in range(0, len(all_pages), SHEET_N):
            grp = all_pages[i:i + SHEET_N]
            contact_sheet(doc, grp, man,
                          root / "sheets" / f"s{i//SHEET_N:03d}"
                          f"_p{grp[0]:04d}-p{grp[-1]:04d}.png")
        print(f"contact sheets -> {root/'sheets'}")

    vpath = root / "verdicts.json"
    if not vpath.exists():
        vpath.write_text(json.dumps({}, indent=1))
        print(f"verdict log created: {vpath}")

    # a compact index to drive the review order
    lab = [p for p in all_pages if man[str(p)]["is_labeled"]]
    nb = {p: len(man[str(p)]["boxes"]) for p in all_pages}
    print(f"\nalready annotated (skip): {len(lab)} pages")
    print(f"pages with >=1 box: {sum(1 for p in all_pages if nb[p])}")
    print(f"pages with 0 boxes: {sum(1 for p in all_pages if not nb[p])}")
    doc.close()


if __name__ == "__main__":
    main()
