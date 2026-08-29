#!/usr/bin/env python3
"""
Snapshot diff — the anti-whack-a-mole gate.

Compares two eval/score.py snapshots and reports every page that changed
category, so a fix's full effect (what it repaired AND what it broke) is
visible before acceptance:

  REGRESSED  page was clean in baseline, has FP/MISS in candidate
  FIXED      page had FP/MISS in baseline, is clean in candidate
  CHANGED    page was and stays dirty, but error counts moved
  LOST       a GT box matched in baseline is unmatched in candidate
  GAINED     a GT box unmatched in baseline is matched in candidate
  DEGRADED   a matched table's quality dropped materially
             (IoU -0.05 / coverage -0.10 / text recall -0.05)
  IMPROVED   mirror image of DEGRADED

--strict exits 1 when any regression exists (usable as a commit gate).

Usage:
    python3 eval/diff.py <baseline.json> <candidate.json> [--strict] [--top N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

D_IOU = 0.05     # quality deltas that count as degraded/improved
D_COV = 0.10
D_TREC = 0.05


def _page_status(page: dict):
    """(n_fp, n_miss) for a GT-mode page record."""
    return len(page.get("fp", [])), len(page.get("miss", []))


def _name(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) > 2 else stem


def diff(baseline: dict, candidate: dict, top: int) -> dict:
    regressed, fixed, changed = [], [], []
    lost, gained, degraded, improved = [], [], [], []

    bb, cb = baseline["books"], candidate["books"]
    for stem in sorted(set(bb) & set(cb)):
        if bb[stem].get("mode") != "gt" or cb[stem].get("mode") != "gt":
            continue
        name = _name(stem)
        pages_b, pages_c = bb[stem]["pages"], cb[stem]["pages"]

        for pno in sorted(set(pages_b) | set(pages_c), key=int):
            pb = pages_b.get(pno, {"matches": [], "fp": [], "miss": []})
            pc = pages_c.get(pno, {"matches": [], "fp": [], "miss": []})
            sb, sc = _page_status(pb), _page_status(pc)
            clean_b, clean_c = sb == (0, 0), sc == (0, 0)

            if clean_b and not clean_c:
                regressed.append(f"{name} p{pno}: +{sc[0]}/-{sc[1]}")
            elif not clean_b and clean_c:
                fixed.append(f"{name} p{pno} (was +{sb[0]}/-{sb[1]})")
            elif not clean_b and not clean_c and sb != sc:
                changed.append(f"{name} p{pno}: +{sb[0]}/-{sb[1]} "
                               f"→ +{sc[0]}/-{sc[1]}")

            # table-level: key matches by GT index
            mb = {m["g"]: m for m in pb["matches"]}
            mc = {m["g"]: m for m in pc["matches"]}
            for g in sorted(set(mb) | set(mc)):
                tag = f"{name} p{pno} gt#{g}"
                if g in mb and g not in mc:
                    lost.append(f"{tag} (was IoU={mb[g]['iou']:.2f})")
                elif g not in mb and g in mc:
                    gained.append(f"{tag} (now IoU={mc[g]['iou']:.2f})")
                else:
                    a, b2 = mb[g], mc[g]
                    drops = []
                    if b2["iou"] < a["iou"] - D_IOU:
                        drops.append(f"IoU {a['iou']:.2f}→{b2['iou']:.2f}")
                    if b2["cov"] < a["cov"] - D_COV:
                        drops.append(f"cov {a['cov']:.2f}→{b2['cov']:.2f}")
                    ta, tb = a.get("trec"), b2.get("trec")
                    if ta is not None and tb is not None \
                            and tb < ta - D_TREC:
                        drops.append(f"trec {ta:.2f}→{tb:.2f}")
                    if drops:
                        degraded.append(f"{tag}: {', '.join(drops)}")
                        continue
                    gains = []
                    if b2["iou"] > a["iou"] + D_IOU:
                        gains.append(f"IoU {a['iou']:.2f}→{b2['iou']:.2f}")
                    if b2["cov"] > a["cov"] + D_COV:
                        gains.append(f"cov {a['cov']:.2f}→{b2['cov']:.2f}")
                    if ta is not None and tb is not None \
                            and tb > ta + D_TREC:
                        gains.append(f"trec {ta:.2f}→{tb:.2f}")
                    if gains:
                        improved.append(f"{tag}: {', '.join(gains)}")

    # smoke books: invariant drift summary.  Tables are matched between
    # snapshots by greedy bbox IoU on the same page — never by list slot:
    # a page whose table count changed must not compare different tables.
    from metrics import iou as _iou  # local import: eval/ on sys.path
    smoke = []
    for stem in sorted(set(bb) & set(cb)):
        if bb[stem].get("mode") != "smoke" or cb[stem].get("mode") != "smoke":
            continue
        sb, sc = bb[stem]["smoke"], cb[stem]["smoke"]
        name = _name(stem)
        tb, tc = sb["tables"], sc["tables"]
        pairs = sorted(
            ((_iou(a["bbox"], b["bbox"]), ai, bi)
             for ai, a in enumerate(tb) for bi, b in enumerate(tc)
             if a["page"] == b["page"]),
            reverse=True)
        ua, ub, matched = set(), set(), []
        for v, ai, bi in pairs:
            if v < 0.5:
                break
            if ai in ua or bi in ub:
                continue
            ua.add(ai); ub.add(bi); matched.append((tb[ai], tc[bi]))
        worse = [f"p{a['page']}: cov {a['self_cov']:.2f}→{b['self_cov']:.2f}"
                 + (f" trec {a['trec']:.2f}→{b['trec']:.2f}"
                    if a.get("trec") is not None and b.get("trec") is not None
                    and b["trec"] < a["trec"] - D_TREC else "")
                 for a, b in matched
                 if b["self_cov"] < a["self_cov"] - D_COV]
        added = len(tc) - len(matched)
        removed = len(tb) - len(matched)
        if worse or added or removed or sb["overlap"] != sc["overlap"]:
            smoke.append(f"{name}: cov↓[{'; '.join(worse)}] "
                         f"+{added}/-{removed} tables, "
                         f"overlap {sb['overlap']}→{sc['overlap']}")

    return {
        "REGRESSED": regressed, "FIXED": fixed, "CHANGED": changed,
        "LOST": lost, "GAINED": gained,
        "DEGRADED": degraded, "IMPROVED": improved,
        "SMOKE_DRIFT": smoke,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any regression")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    print(f"baseline : {args.baseline} ({baseline['meta'].get('ts')})")
    print(f"candidate: {args.candidate} ({candidate['meta'].get('ts')})")

    d = diff(baseline, candidate, args.top)
    n_regress = (len(d["REGRESSED"]) + len(d["LOST"]) + len(d["DEGRADED"]))
    n_fix = len(d["FIXED"]) + len(d["GAINED"]) + len(d["IMPROVED"])

    for section in ("REGRESSED", "LOST", "DEGRADED",
                    "FIXED", "GAINED", "IMPROVED", "CHANGED", "SMOKE_DRIFT"):
        items = d[section]
        if not items:
            continue
        head = "── ⚠ " if section in ("REGRESSED", "LOST", "DEGRADED") else "── "
        print(f"\n{head}{section}: {len(items)}")
        for line in items[: args.top]:
            print(f"  {line}")
        if len(items) > args.top:
            print(f"  ... and {len(items) - args.top} more")

    print(f"\n{'='*60}")
    print(f"  regressions={n_regress}  fixes={n_fix}  "
          f"changed-dirty={len(d['CHANGED'])}")
    o_b, o_c = baseline.get("overall"), candidate.get("overall")
    if o_b and o_c:
        print(f"  F1 {o_b['F1']:.4f}→{o_c['F1']:.4f} | "
              f"cov<90% {o_b['cov_lt90']}→{o_c['cov_lt90']} | "
              f"trec<95% {o_b['trec_lt95']}→{o_c['trec_lt95']}")
    print(f"{'='*60}")

    if args.strict and n_regress:
        print("STRICT: regressions present — gate FAILED")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
