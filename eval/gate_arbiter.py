"""
Gate arbiter — exchangeability test for semantic gates under run-to-run noise.

Three runs over the same corpus:  A (baseline run 1), A2 (same code, run 2),
C (candidate).  A is the shared reference for both comparisons:

    noise  = dirty(A vs A2)      change = dirty(A vs C)

If C is semantically equivalent to the baseline code, C is just another
noisy draw like A2, so a table dirty in exactly one of the two comparisons
is equally likely to fall on either side (McNemar).  The old rule
"change - noise must be empty" fails even for A2 itself, because two
independent noise draws almost never hit the same tables.

Verdict (all must hold):
  1. Hard categories (TABLE_ADDED / TABLE_REMOVED / PAGE_MISMATCH) are 0
     in the change comparison, or no higher than in the noise comparison.
  2. One-sided exact binomial test on discordant tables:
     P(X >= change_only | n = change_only + noise_only, p = 0.5) >= alpha.
  3. Rate bound: change_rate <= noise_rate * (1 + rate_margin).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.semantic_diff import diff_corpus, centroid  # noqa: E402

SOFT = ("SHAPE_CHANGED", "CELL_TEXT_DIFF", "CELL_NUMERIC_DIFF",
        "CELL_SIGN_DIFF", "HEADER_FLAG_DIFF", "BBOX_DRIFT")
HARD = ("TABLE_ADDED", "TABLE_REMOVED", "PAGE_MISMATCH")


def _dirty_keys(rep: dict, cats) -> Dict[str, Set[Tuple]]:
    out = {c: set() for c in cats}
    out["ANY"] = set()
    for stem, bd in rep.get("books", {}).items():
        if not isinstance(bd, dict):
            continue
        if "error" in bd:
            out.setdefault("BOOK_ERROR", set()).add((stem,))
            continue
        for d in bd.get("diffs", []):
            cs = [c for c in d.get("categories", []) if c in cats]
            if not cs:
                continue
            page = d.get("page_b") if d.get("page_b") is not None else d.get("page_c")
            bbox = d.get("bbox_b") or d.get("bbox_c")
            cx, cy = centroid(bbox) if bbox else (0.0, 0.0)
            key = (stem, page, round(cx, 1), round(cy, 1))
            for c in cs:
                out[c].add(key)
            out["ANY"].add(key)
    return out


def binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def arbitrate(a: Path, a2: Path, c: Path, alpha: float, rate_margin: float,
              bbox_iou_floor: float, centroid_tol_px: float) -> dict:
    rn = diff_corpus(a, a2, bbox_iou_floor=bbox_iou_floor, centroid_tol_px=centroid_tol_px)
    rc = diff_corpus(a, c, bbox_iou_floor=bbox_iou_floor, centroid_tol_px=centroid_tol_px)
    cats = SOFT + HARD
    kn, kc = _dirty_keys(rn, cats), _dirty_keys(rc, cats)
    n_tables = rn["n_tables_b"]
    per_cat = {}
    for cat in ("ANY",) + cats:
        sn, sc = kn.get(cat, set()), kc.get(cat, set())
        co, no = len(sc - sn), len(sn - sc)
        per_cat[cat] = {"noise": len(sn), "change": len(sc), "overlap": len(sn & sc),
                        "change_only": co, "noise_only": no,
                        "p_one_sided": round(binom_sf(co, co + no), 4)}
    anyc = per_cat["ANY"]
    noise_rate = anyc["noise"] / max(n_tables, 1)
    change_rate = anyc["change"] / max(n_tables, 1)
    hard_change = sum(per_cat[h]["change"] for h in HARD)
    hard_noise = sum(per_cat[h]["noise"] for h in HARD)
    checks = {
        "books_match": rn["n_books_common"] == rc["n_books_common"]
                        and not rc["n_books_only_b"] and not rc["n_books_only_c"],
        "hard_categories": hard_change <= hard_noise,
        "exchangeability": anyc["p_one_sided"] >= alpha,
        "rate_bound": change_rate <= noise_rate * (1 + rate_margin),
        "no_book_errors": not kc.get("BOOK_ERROR"),
    }
    return {
        "a": str(a), "a2": str(a2), "c": str(c),
        "params": {"alpha": alpha, "rate_margin": rate_margin,
                   "bbox_iou_floor": bbox_iou_floor, "centroid_tol_px": centroid_tol_px},
        "n_books": rc["n_books_common"], "n_tables": n_tables,
        "noise_rate": round(noise_rate, 5), "change_rate": round(change_rate, 5),
        "per_category": per_cat,
        "change_only_keys": sorted(kc["ANY"] - kn["ANY"])[:200],
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--a", type=Path, required=True, help="baseline run 1 (shared reference)")
    ap.add_argument("--a2", type=Path, required=True, help="baseline run 2, same code")
    ap.add_argument("--c", type=Path, required=True, help="candidate run")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rate-margin", type=float, default=0.20)
    ap.add_argument("--bbox-iou-floor", type=float, default=0.5)
    ap.add_argument("--centroid-tol-px", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = arbitrate(args.a, args.a2, args.c, args.alpha, args.rate_margin,
                    args.bbox_iou_floor, args.centroid_tol_px)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text)
    summary = {k: rep[k] for k in ("n_books", "n_tables", "noise_rate", "change_rate", "checks", "verdict")}
    summary["ANY"] = rep["per_category"]["ANY"]
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(_cli())
