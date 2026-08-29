#!/usr/bin/env python3
"""
Batch-2 review fixes: 紫光股份 (LS project 6) + 一彬科技 (LS project 7).

Every op below was decided by page-by-page visual review (verdicts.json) and
then coordinate-anchored:
  - 一彬: PDF vector table frames via eval/probe_regions.py (probe_yb.txt),
    add boxes = frame region padded to the pipeline's own box style
    (x 9.50->9.16 / 90.52->90.86, y +-0.30).
  - 紫光: borderless gray tables, anchored to the text layer line dump
    (zg_lines.txt), x matches the pipeline style x0=14.77.

Ops: add (x0,y0,x1,y1 pct) / merge (i,j) / delete i / reshape i {y0,y1}.
Pages with an empty prediction result get the box template from a donor
prediction of the same project (same page geometry).

Usage
    python3 eval/apply_fixes_batch2.py            # dry-run
    python3 eval/apply_fixes_batch2.py --apply    # PATCH to Label Studio
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
REVIEW = Path("/home/homio/sda/1111/output/review")

YB = "20260415_001278.SZ_一彬科技_2025年年度报告_fa2f5298"
ZG = "20260415_000938.SZ_紫光股份_2025年年度报告_9e6c5fd6"

A = "add"


def box(x0, y0, x1, y1):
    return (A, None, {"box": (x0, y0, x1, y1)})


# 一彬: 空表头/跨页残段补框 + p172/p179拆分 + p235合并
FIXES_YB = {
    136: [box(9.16, 88.41, 90.86, 90.98)],
    137: [box(9.16, 31.30, 90.86, 33.87)],
    139: [box(9.16, 67.06, 90.86, 69.63), box(9.16, 88.62, 90.86, 91.18)],
    143: [box(9.16, 26.04, 90.86, 28.61), box(9.16, 78.96, 90.86, 81.51),
          box(9.16, 88.28, 90.86, 90.84)],
    146: [box(9.16, 67.93, 90.86, 70.50)],
    147: [box(9.16, 8.25, 90.86, 10.82), box(9.16, 76.37, 90.86, 78.94),
          box(9.16, 85.22, 90.86, 87.79)],
    148: [box(9.16, 31.16, 90.86, 33.71), box(9.16, 44.77, 90.86, 47.32),
          box(9.16, 51.00, 90.86, 53.56), box(9.16, 73.16, 90.86, 75.73),
          box(9.16, 82.01, 90.86, 84.58)],
    149: [box(9.16, 20.58, 90.86, 23.13), box(9.16, 34.18, 90.86, 36.75),
          box(9.16, 39.47, 90.86, 42.03)],
    150: [box(9.16, 8.25, 90.86, 10.82)],
    152: [box(9.16, 59.02, 90.86, 61.57), box(9.16, 70.01, 90.86, 72.58),
          box(9.16, 76.84, 90.86, 79.41), box(9.16, 87.71, 90.86, 90.27)],
    155: [box(9.16, 13.35, 90.86, 15.92), box(9.16, 79.32, 90.86, 81.89)],
    156: [box(9.16, 78.28, 90.86, 80.84), box(9.16, 87.60, 90.86, 90.17)],
    157: [box(9.16, 37.10, 90.86, 39.65), box(9.16, 74.27, 90.86, 76.83)],
    158: [box(9.16, 33.35, 90.86, 35.92), box(9.16, 77.51, 90.86, 80.08)],
    161: [box(9.16, 68.51, 90.86, 71.07), box(9.16, 82.13, 90.86, 84.68)],
    162: [box(9.16, 8.25, 90.86, 10.82)],
    165: [box(9.16, 8.25, 90.86, 10.82), box(9.16, 46.82, 90.86, 49.37),
          box(9.16, 55.67, 90.86, 58.22)],
    166: [box(9.16, 65.29, 90.86, 67.86), box(9.16, 74.15, 90.86, 76.70),
          box(9.16, 83.00, 90.86, 85.55)],
    167: [box(9.16, 20.01, 90.86, 22.58)],
    168: [box(8.20, 55.03, 92.26, 58.42)],
    171: [box(9.16, 40.62, 90.86, 43.17)],
    # 误合并: #1 罩住两张空表头 -> 顶段收缩 + 补下段
    172: [("reshape", 1, {"y1": 14.25}), box(9.16, 16.28, 90.86, 18.83)],
    176: [box(9.16, 69.78, 90.86, 72.34)],
    177: [box(9.16, 22.96, 90.86, 25.51), box(9.16, 69.88, 90.86, 72.45)],
    178: [box(9.16, 26.07, 90.86, 28.63), box(9.16, 32.31, 90.86, 34.86),
          box(9.16, 45.91, 90.86, 48.47), box(9.16, 76.65, 90.86, 79.22)],
    # #3 罩住两张空表头 -> 删除后按线框拆两框, 另补3处漏检
    179: [("delete", 3, {}),
          box(9.16, 25.65, 90.86, 28.21), box(9.16, 29.52, 90.86, 32.07),
          box(9.16, 16.80, 90.86, 19.37), box(9.16, 47.03, 90.86, 49.59),
          box(9.16, 52.32, 90.86, 54.88)],
    181: [box(9.16, 8.25, 90.86, 10.82)],
    182: [box(9.16, 76.51, 90.86, 79.06)],
    183: [box(9.16, 8.25, 90.86, 10.82), box(9.16, 72.16, 90.86, 74.73),
          box(9.16, 81.01, 90.86, 83.57)],
    184: [box(9.16, 13.35, 90.86, 15.92), box(9.16, 30.42, 90.86, 32.97),
          box(9.16, 41.87, 90.86, 44.43), box(9.16, 48.10, 90.86, 50.67),
          box(9.16, 53.39, 90.86, 55.96)],
    185: [box(9.16, 30.83, 90.86, 33.38)],
    186: [box(9.16, 30.83, 90.86, 33.38), box(9.16, 76.70, 90.86, 79.25)],
    189: [box(9.16, 47.36, 90.86, 49.93)],
    191: [box(9.16, 50.84, 90.86, 53.41)],
    194: [box(9.16, 62.76, 90.86, 65.32), box(9.16, 83.72, 90.86, 86.29)],
    195: [box(9.16, 30.17, 90.86, 32.74)],
    197: [box(9.16, 11.70, 90.86, 14.25)],
    198: [box(9.16, 20.97, 90.86, 33.71), box(9.16, 44.75, 90.86, 57.51),
          box(9.16, 87.06, 90.86, 89.61)],
    199: [box(9.16, 13.35, 90.86, 15.92)],
    201: [box(9.16, 55.52, 90.86, 58.08), box(9.16, 61.76, 90.86, 64.31),
          box(9.16, 68.94, 90.86, 71.51)],
    217: [box(9.16, 73.70, 90.86, 76.26)],
    218: [box(9.16, 50.24, 90.86, 52.81)],
    219: [box(8.20, 21.68, 92.26, 25.06)],
    220: [box(9.16, 15.98, 90.86, 18.53), box(9.16, 22.80, 90.86, 25.36),
          box(9.16, 49.12, 90.86, 51.68)],
    222: [box(9.16, 76.51, 90.86, 79.06), box(9.16, 83.69, 90.86, 86.26)],
    223: [box(9.16, 35.86, 90.86, 38.43), box(9.16, 43.04, 90.86, 45.61),
          box(9.16, 84.83, 90.86, 87.39)],
    227: [box(9.16, 59.02, 90.86, 61.57), box(9.16, 70.01, 90.86, 72.58),
          box(9.16, 76.25, 90.86, 78.81)],
    228: [box(9.16, 45.31, 90.86, 47.88), box(9.16, 54.16, 90.86, 56.71),
          box(9.16, 82.58, 90.86, 85.15)],
    229: [box(9.16, 13.35, 90.86, 15.92), box(9.16, 19.60, 90.86, 22.15),
          box(9.16, 41.76, 90.86, 44.33), box(9.16, 50.61, 90.86, 53.17),
          box(9.16, 74.27, 90.86, 76.84), box(9.16, 87.88, 90.86, 90.45)],
    230: [box(9.16, 11.93, 90.86, 14.50)],
    232: [box(9.16, 61.14, 90.86, 63.70), box(9.16, 75.10, 90.86, 77.65),
          box(9.16, 81.33, 90.86, 83.89)],
    # 过切: 表头段+数据段同一张表
    235: [("merge", (0, 1), {})],
    237: [box(9.16, 17.40, 90.86, 19.95)],
}

# 紫光: 无边框灰底表漏检补框(文本层锚定) + p239拆3框
FIXES_ZG = {
    115: [box(14.77, 22.57, 85.50, 84.01)],   # 重要性标准两列表
    130: [box(14.77, 31.19, 85.50, 40.15)],   # 无形资产使用寿命表
    132: [box(14.77, 26.11, 85.50, 33.35)],   # 长期待摊费用摊销期表
    146: [box(14.77, 35.03, 85.50, 91.36)],   # 境外子公司税率表
    164: [box(14.77, 29.58, 85.50, 33.80)],   # 长期应收款坏账准备变动
    190: [box(14.77, 47.01, 85.50, 74.69)],   # 51销售费用
    191: [box(14.77, 23.03, 85.50, 47.34),    # 52管理费用
          box(14.77, 52.02, 85.50, 74.62)],   # 53研发费用
    205: [box(14.77, 26.41, 85.50, 36.99)],   # 处置子公司财务信息
    210: [box(14.77, 65.08, 85.50, 91.62)],   # 新华三主要财务信息
    223: [box(14.77, 22.42, 85.50, 87.23)],   # 其他关联方名单
    224: [box(14.77, 20.88, 85.50, 68.76)],   # 其他关联方名单(续)
    229: [box(14.77, 83.94, 85.50, 89.44)],   # 关键管理人员薪酬
    # 误合并: 1大框罩3张表 -> 删除+按各表文本范围补3框
    239: [("delete", 0, {}),
          box(14.77, 26.41, 85.50, 47.44),    # 账龄分析
          box(14.77, 55.76, 85.50, 65.03),    # 2025年计提
          box(14.77, 69.78, 85.50, 79.04)],   # 2024年计提
    241: [box(14.77, 41.39, 85.50, 53.79)],   # 应收股利分类
    249: [box(14.77, 67.18, 85.50, 72.68)],   # 合同负债年初账面价值
}


def ls_session():
    return _ls_get_session()


def new_item(tpl, x0, y0, x1, y1):
    it = copy.deepcopy(tpl)
    it["id"] = uuid.uuid4().hex[:10]
    it["value"].update(x=round(x0, 2), y=round(y0, 2),
                       width=round(x1 - x0, 2), height=round(y1 - y0, 2))
    return it


def apply_ops(result, ops, tpl):
    items = copy.deepcopy(result)
    drop = set()
    adds = []
    for op, idx, arg in ops:
        if op == A:
            adds.append(new_item(tpl, *arg["box"]))
        elif op == "reshape":
            v = items[idx]["value"]
            if "y0" in arg:
                y1 = v["y"] + v["height"]
                v["y"] = arg["y0"]
                v["height"] = round(y1 - arg["y0"], 2)
            if "y1" in arg:
                v["height"] = round(arg["y1"] - v["y"], 2)
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
    return [it for i, it in enumerate(items) if i not in drop] + adds


def donor_template(s, h, man):
    """First result item of any non-empty prediction = box template."""
    for pn in sorted(man, key=int):
        if man[pn]["boxes"]:
            pr = s.get(f"{BASE}/api/predictions?task={man[pn]['task_id']}",
                       headers=h).json()
            preds = pr if isinstance(pr, list) else pr.get("results", pr)
            if preds and preds[0]["result"]:
                return preds[0]["result"][0]
    sys.exit("no donor prediction found")


def run(stem, fixes, s, h, apply):
    man = json.loads((REVIEW / stem / "manifest.json").read_text())
    tpl = donor_template(s, h, man)
    for pn, ops in sorted(fixes.items()):
        m = man[str(pn)]
        if m["is_labeled"]:
            print(f"p{pn}: SKIP, already annotated")
            continue
        pr = s.get(f"{BASE}/api/predictions?task={m['task_id']}",
                   headers=h).json()
        preds = pr if isinstance(pr, list) else pr.get("results", pr)
        pred = preds[0]
        new_result = apply_ops(pred["result"], ops, tpl)
        print(f"p{pn}: {len(pred['result'])} -> {len(new_result)} boxes")
        if apply:
            r = s.patch(f"{BASE}/api/predictions/{pred['id']}/",
                        json={"result": new_result}, headers=h)
            r.raise_for_status()
            print(f"      PATCH ok (pred {pred['id']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    s = ls_session()
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}
    print("### 一彬科技 (project 7)")
    run(YB, FIXES_YB, s, h, args.apply)
    print("\n### 紫光股份 (project 6)")
    run(ZG, FIXES_ZG, s, h, args.apply)
    print("\nmode:", "APPLIED" if args.apply else "dry-run")


if __name__ == "__main__":
    main()
