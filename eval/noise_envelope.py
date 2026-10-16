"""
Noise envelope analysis — distinguishes natural pipeline jitter from
real semantic changes introduced by a code change.

Method:
  1. baseline1 vs baseline2 (same code, two runs) — gives the natural
     noise set: tables whose semantic diff is explained by pipeline
     non-determinism.
  2. baseline vs candidate (code change applied) — gives the change
     set: tables whose semantic diff is attributable to the new code.
  3. Subtract: change_set − noise_set = real_new_regressions.

Tables that are in BOTH the noise set and the change set are NOT
regressions — they are explained by pipeline jitter.  Tables that are
ONLY in the change set are real new semantic changes.  Tables that
are ONLY in the noise set are noise-only (no code-change effect).

The same (book_stem, page_no, bbox_centroid) tuple identifies a table
across runs.  This is the noise tolerance discipline: a regression
that lives inside the noise envelope is acceptable; a regression
outside it is a real new failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import the diff tool from the same package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.semantic_diff import (  # noqa: E402
    diff_corpus, _table_identity, centroid, _norm_text, _parse_number,
    _cell_text, _cell_pos, _cell_flags, _coerce_cell, _cell_map, iou,
    diff_table,
)


def _table_key(stem: str, table: dict) -> Tuple:
    """Stable identity: stem + page + centroid (rounded to 0.1px to
    absorb float jitter that does not change the visual location)."""
    page, bbox = _table_identity(table)
    cx, cy = (round(centroid(bbox)[0], 1), round(centroid(bbox)[1], 1))
    return (stem, page, cx, cy)


def _category_to_set(book_results: dict, category: str) -> Set[Tuple]:
    """Collect (book_stem, page, cx, cy) keys for every table that
    triggered the given diff category in a corpus diff result."""
    out: Set[Tuple] = set()
    for stem, bd in book_results.get("books", {}).items():
        if not isinstance(bd, dict) or "diffs" not in bd:
            continue
        for d in bd["diffs"]:
            if category in d.get("categories", []):
                page = d.get("page_b") or d.get("page_c")
                if page is None:
                    continue
                bbox = d.get("bbox_b") or d.get("bbox_c")
                if bbox is None:
                    continue
                cx, cy = centroid(bbox)
                out.add((stem, page, round(cx, 1), round(cy, 1)))
    return out


def _per_table_cell_diffs(book_results: dict) -> Dict[Tuple, Dict[str, int]]:
    """For each (book, page, cx, cy), accumulate per-category cell counts
    in the matched table diff metadata."""
    out: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: Counter())
    for stem, bd in book_results.get("books", {}).items():
        if not isinstance(bd, dict) or "diffs" not in bd:
            continue
        for d in bd["diffs"]:
            if d.get("bidx") is None or d.get("cidx") is None:
                continue
            page = d.get("page_b")
            bbox = d.get("bbox_b")
            if page is None or bbox is None:
                continue
            cx, cy = centroid(bbox)
            key = (stem, page, round(cx, 1), round(cy, 1))
            for note in d.get("notes", []):
                if note.startswith("meta="):
                    try:
                        meta = json.loads(note[len("meta="):])
                    except (ValueError, SyntaxError):
                        continue
                    for k in ("n_text_diff", "n_numeric_diff",
                              "n_sign_diff", "n_header_flag_diff"):
                        if meta.get(k):
                            out[key][k] += int(meta[k])
    return out


def _per_table_cell_text_diffs(stem: str, baseline_dir: Path,
                                candidate_dir: Path) -> Dict[Tuple, List[dict]]:
    """For a single book pair, return per-key list of (pos, bt, ct)
    text differences (whitespace-normalised, after both runs)."""
    out: Dict[Tuple, List[dict]] = defaultdict(list)
    for label, bdir, cdir in [("noise", baseline_dir, candidate_dir)]:
        pass  # unused; we work per-book
    return out  # not used — kept for symmetry


def noise_envelope(baseline1: Path, baseline2: Path,
                   baseline: Path, candidate: Path,
                   bbox_iou_floor: float = 0.5,
                   centroid_tol_px: float = 2.0) -> dict:
    """Three-way comparison.

    Returns a dict with:
      noise_set:      tables dirty in baseline1 vs baseline2 (noise only)
      change_set:     tables dirty in baseline vs candidate (code change)
      overlap:        tables dirty in BOTH (explained by noise, not new)
      noise_only:     noise_set - change_set  (noise we already had)
      change_only:    change_set - noise_set  (NEW real changes)
      category_breakdown: per-category counts of overlap vs change_only
    """
    rep_noise = diff_corpus(baseline1, baseline2,
                            bbox_iou_floor=bbox_iou_floor,
                            centroid_tol_px=centroid_tol_px)
    rep_change = diff_corpus(baseline, candidate,
                             bbox_iou_floor=bbox_iou_floor,
                             centroid_tol_px=centroid_tol_px)

    cat_breakdown = {}
    for cat in ("SHAPE_CHANGED", "CELL_TEXT_DIFF", "CELL_NUMERIC_DIFF",
                "CELL_SIGN_DIFF", "HEADER_FLAG_DIFF", "TABLE_ADDED",
                "TABLE_REMOVED", "BBOX_DRIFT", "PAGE_MISMATCH"):
        s_noise = _category_to_set(rep_noise, cat)
        s_change = _category_to_set(rep_change, cat)
        s_overlap = s_noise & s_change
        s_change_only = s_change - s_noise
        s_noise_only = s_noise - s_change
        cat_breakdown[cat] = {
            "noise": len(s_noise),
            "change": len(s_change),
            "overlap": len(s_overlap),
            "change_only": len(s_change_only),
            "noise_only": len(s_noise_only),
            "overlap_keys": sorted(s_overlap)[:50],
            "change_only_keys": sorted(s_change_only)[:50],
            "noise_only_keys": sorted(s_noise_only)[:50],
        }

    # Per-cell counts: how many of the candidate's CELL_*_DIFF cells
    # are explained by the same cells being dirty in noise?
    noise_cells = _per_table_cell_diffs(rep_noise)
    change_cells = _per_table_cell_diffs(rep_change)
    cell_summary = {}
    for kind in ("n_text_diff", "n_numeric_diff", "n_sign_diff",
                 "n_header_flag_diff"):
        noise_total = sum(c[kind] for c in noise_cells.values())
        change_total = sum(c[kind] for c in change_cells.values())
        cell_summary[kind] = {
            "noise_total": noise_total,
            "change_total": change_total,
        }

    return {
        "baseline1": str(baseline1),
        "baseline2": str(baseline2),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "noise_summary": {
            "n_books_common": rep_noise["n_books_common"],
            "n_tables_b": rep_noise["n_tables_b"],
            "n_tables_c": rep_noise["n_tables_c"],
            "n_matched": rep_noise["n_matched"],
            "n_clean": rep_noise["n_clean"],
            "categories_total": rep_noise["categories_total"],
        },
        "change_summary": {
            "n_books_common": rep_change["n_books_common"],
            "n_tables_b": rep_change["n_tables_b"],
            "n_tables_c": rep_change["n_tables_c"],
            "n_matched": rep_change["n_matched"],
            "n_clean": rep_change["n_clean"],
            "categories_total": rep_change["categories_total"],
        },
        "category_breakdown": cat_breakdown,
        "cell_summary": cell_summary,
        "noise_per_book": {
            stem: {
                "n_diffs": bd.get("n_diffs"),
                "n_clean": bd.get("n_clean"),
                "categories": bd.get("categories"),
            } for stem, bd in rep_noise.get("books", {}).items()
            if isinstance(bd, dict) and "categories" in bd
        },
        "change_per_book": {
            stem: {
                "n_diffs": bd.get("n_diffs"),
                "n_clean": bd.get("n_clean"),
                "categories": bd.get("categories"),
            } for stem, bd in rep_change.get("books", {}).items()
            if isinstance(bd, dict) and "categories" in bd
        },
    }


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise1", type=Path, required=True,
                    help="First baseline run dir (for noise envelope)")
    ap.add_argument("--noise2", type=Path, required=True,
                    help="Second baseline run dir (same code, for noise)")
    ap.add_argument("--baseline", type=Path, required=True,
                    help="Reference baseline for candidate comparison")
    ap.add_argument("--candidate", type=Path, required=True,
                    help="Candidate (code change applied) dir")
    ap.add_argument("--bbox-iou-floor", type=float, default=0.5)
    ap.add_argument("--centroid-tol-px", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rep = noise_envelope(args.noise1, args.noise2, args.baseline,
                          args.candidate,
                          bbox_iou_floor=args.bbox_iou_floor,
                          centroid_tol_px=args.centroid_tol_px)
    if args.out:
        args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        json.dump(rep, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
