#!/usr/bin/env python3
"""
Apply the page-by-page visual review fixes (verdicts.json, status=fix) to the
Label Studio predictions of project 4 (中国石油 2024 annual report).

Every fix below was decided by eyeballing the rendered overlay and then
anchored to the PDF text layer (see /tmp/probe_fix.py output), so the numbers
are content-snapped, not guessed.  Already-annotated tasks are never touched.

Fix types
    extend  - keep the box, move its bottom edge down to the real last row
    merge   - two boxes over one table -> replace with their union
    reshape - move top edge (p190 #1 grabbed half of the table above it)
    delete  - spurious box over plain text (p158 #2)
    add     - missed table gets a brand-new box (p274)

Usage
    python3 eval/apply_review_fixes.py            # dry-run, prints plan
    python3 eval/apply_review_fixes.py --apply    # PATCH to Label Studio
"""

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path

import requests

from ls_auth import base_url as _ls_base, get_session as _ls_get_session
BASE = _ls_base()
STEM = "20250331_601857.SH_中国石油_中国石油天然气股份有限公司2024年度报告_e9f55911"
MAN = Path(f"/home/homio/sda/1111/output/review/{STEM}/manifest.json")

# page -> list of ops on the prediction result (indices follow overlay #N)
FIXES = {
    54:  [("extend", 0, {"y1": 27.35})],          # 3 data rows + 合计 were cut
    62:  [("merge", (0, 1), {})],                 # header block + √-mark rows
    93:  [("extend", 0, {"y1": 47.95})],          # 张明禄 data row was cut
    100: [("merge", (0, 1), {})],                 # bond table split in two
    155: [("extend", 1, {"y1": 61.45})],          # rows after 综合收益总额 cut
    158: [("delete", 2, {})],                     # box over section title text
    190: [("reshape", 1, {"y0": 35.20})],         # #1 must start at 2023 liabilities header
    251: [("extend", 0, {"y1": 88.20})],          # rows down to 合计 114,290 cut
    272: [("extend", 1, {"y1": 30.55}),           # 本、勘探和开发成本 row
          ("extend", 0, {"y1": 78.63})],          # 动经营业绩的利润合计 row
    274: [("add", None, {"box": (14.18, 63.90, 74.31, 4.00)})],  # missed 2-row table
}


def ls_session():
    return _ls_get_session()


def apply_ops(result, ops):
    """Return a new result list with the ops applied."""
    items = copy.deepcopy(result)
    drop = set()
    for op, idx, arg in ops:
        if op == "extend":
            v = items[idx]["value"]
            v["height"] = round(arg["y1"] - v["y"], 2)
        elif op == "reshape":
            v = items[idx]["value"]
            y1 = v["y"] + v["height"]
            v["y"] = arg["y0"]
            v["height"] = round(y1 - arg["y0"], 2)
        elif op == "merge":
            a, b = (items[i]["value"] for i in idx)
            x0 = min(a["x"], b["x"])
            y0 = min(a["y"], b["y"])
            x1 = max(a["x"] + a["width"], b["x"] + b["width"])
            y1 = max(a["y"] + a["height"], b["y"] + b["height"])
            a.update(x=round(x0, 2), y=round(y0, 2),
                     width=round(x1 - x0, 2), height=round(y1 - y0, 2))
            drop.add(idx[1])
        elif op == "delete":
            drop.add(idx)
        elif op == "add":
            tpl = copy.deepcopy(items[0])
            x, y, w, h = arg["box"]
            tpl["id"] = uuid.uuid4().hex[:10]
            tpl["value"].update(x=x, y=y, width=w, height=h)
            items.append(tpl)
    return [it for i, it in enumerate(items) if i not in drop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    man = json.loads(MAN.read_text())
    s = ls_session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}

    for pn, ops in sorted(FIXES.items()):
        m = man[str(pn)]
        if m["is_labeled"]:
            print(f"p{pn}: SKIP, already annotated")
            continue
        pr = s.get(f"{BASE}/api/predictions?task={m['task_id']}",
                   headers=h).json()
        preds = pr if isinstance(pr, list) else pr.get("results", pr)
        pred = preds[0]
        new_result = apply_ops(pred["result"], ops)
        old_n, new_n = len(pred["result"]), len(new_result)
        boxes = [(round(r['value']['x'], 1), round(r['value']['y'], 1),
                  round(r['value']['width'], 1), round(r['value']['height'], 1))
                 for r in new_result]
        print(f"p{pn}: {old_n} -> {new_n} boxes  {boxes}")
        if args.apply:
            r = s.patch(f"{BASE}/api/predictions/{pred['id']}/",
                        json={"result": new_result}, headers=h)
            r.raise_for_status()
            print(f"      PATCH ok (pred {pred['id']})")
    print("\nmode:", "APPLIED" if args.apply else "dry-run")


if __name__ == "__main__":
    main()
