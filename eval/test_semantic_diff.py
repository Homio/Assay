"""
Unit tests for eval.semantic_diff and eval.noise_envelope.

Tests are self-contained: no real output dirs, no GPU, no model imports.
All inputs are constructed inline from dicts and small JSON literals.

Run with:
    PYTHONPATH=<repo> python3 eval/test_semantic_diff.py
or under pytest (one file, no test_*.py pattern required).

Categories covered:
  1. Geometry helpers (iou / centroid)
  2. Text normalisation (NFKC, whitespace, full-width)
  3. Numeric parsing (commas, %, signs, accountant negative)
  4. Cell decoding (str-repr, dict, scalar)
  5. Identity & matching (no positional zip, bbox IoU, page filter)
  6. Per-table diff (shape, page, text, numeric, sign, header)
  7. Corpus diff (book presence, per-book rollup)
  8. Noise envelope (3-way: noise1↔noise2, baseline↔candidate, overlap)
  9. Identity guarantee (permuting the candidate list does not change match)
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.semantic_diff import (  # noqa: E402
    iou, centroid, _norm_text, _parse_number, _coerce_cell, _cell_map,
    _table_identity, match_tables, diff_table, diff_book, diff_corpus,
)
from eval.noise_envelope import (  # noqa: E402
    _table_key, _category_to_set, noise_envelope,
)


# ── 1. Geometry helpers ──────────────────────────────────────────────

class GeometryTests(unittest.TestCase):
    def test_iou_identical(self):
        b = [0.0, 0.0, 100.0, 100.0]
        self.assertAlmostEqual(iou(b, b), 1.0)

    def test_iou_disjoint(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_iou_partial(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        # inter = 5*5=25; union = 100+100-25 = 175
        self.assertAlmostEqual(iou(a, b), 25 / 175, places=4)

    def test_iou_containment(self):
        a = [0, 0, 10, 10]
        b = [2, 2, 8, 8]
        # inter=36, union=100+36-36=100
        self.assertAlmostEqual(iou(a, b), 36 / 100, places=4)

    def test_centroid(self):
        self.assertEqual(centroid([0, 0, 10, 20]), (5.0, 10.0))
        self.assertEqual(centroid([2.5, 7.5, 12.5, 17.5]), (7.5, 12.5))


# ── 2. Text normalisation ───────────────────────────────────────────

class TextNormTests(unittest.TestCase):
    def test_nfkc_fullwidth_space(self):
        self.assertEqual(_norm_text("a\u3000b"), "a b")

    def test_strip_and_collapse(self):
        self.assertEqual(_norm_text("  hello\n\tworld  "), "hello world")

    def test_preserves_cjk(self):
        self.assertEqual(_norm_text("平安银行"), "平安银行")

    def test_preserves_punctuation(self):
        # comma inside number is business content, but we don't touch it
        # unless it is part of a number (that's _parse_number's job)
        self.assertEqual(_norm_text("1,234.56"), "1,234.56")

    def test_preserves_case(self):
        self.assertEqual(_norm_text("Apple"), "Apple")
        self.assertNotEqual(_norm_text("APPLE"), "Apple")


# ── 3. Numeric parsing ──────────────────────────────────────────────

class NumericParseTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_parse_number("1234"), (1234.0, ""))

    def test_comma(self):
        self.assertEqual(_parse_number("1,234,567"), (1234567.0, ""))

    def test_decimal(self):
        self.assertEqual(_parse_number("1.23"), (1.23, ""))

    def test_percent(self):
        # magnitude preserved, sign_token includes no %
        v, s = _parse_number("12.5%")
        self.assertEqual(v, 12.5)
        self.assertEqual(s, "")

    def test_negative_minus(self):
        v, s = _parse_number("-12.5")
        self.assertEqual(v, 12.5)
        self.assertEqual(s, "-")

    def test_unicode_minus(self):
        v, s = _parse_number("−1.5")
        self.assertEqual(v, 1.5)
        self.assertEqual(s, "−")

    def test_accounting_paren(self):
        v, s = _parse_number("(123.4)")
        self.assertEqual(v, 123.4)
        self.assertEqual(s, "(")

    def test_non_numeric(self):
        self.assertIsNone(_parse_number("hello"))
        self.assertIsNone(_parse_number(""))
        self.assertIsNone(_parse_number(None))


# ── 4. Cell decoding ────────────────────────────────────────────────

class CellDecodeTests(unittest.TestCase):
    def test_dict_passthrough(self):
        c = _coerce_cell({"text": "hi", "row": 0, "col": 0})
        self.assertEqual(c["text"], "hi")

    def test_str_repr_decoded(self):
        raw = "{'text': 'hi', 'row': 0, 'col': 0}"
        c = _coerce_cell(raw)
        self.assertEqual(c.get("text"), "hi")

    def test_scalar_str(self):
        c = _coerce_cell("hello")
        self.assertEqual(c.get("text"), "hello")
        self.assertEqual(c.get("row"), -1)


# ── 5. Identity & matching ──────────────────────────────────────────

class MatchTests(unittest.TestCase):
    def _t(self, page, bbox, text=""):
        return {
            "data": {"num_rows": 1, "num_cols": 1, "grid": [
                [{"text": text, "row": 0, "col": 0}]
            ]},
            "prov": [{"page_no": page, "bbox": bbox}],
        }

    def test_no_positional_zip_same_page_different_bbox(self):
        # Two tables on the same page but different positions.
        # Baseline has order [A, B]; candidate has order [B, A].
        # After matching, A should still match A and B should still match B,
        # NOT positional (i.e. baseline[0] should not match candidate[0]).
        b = [self._t(1, [0, 0, 50, 50], "A"),
             self._t(1, [60, 60, 110, 110], "B")]
        c = [self._t(1, [60, 60, 110, 110], "B"),
             self._t(1, [0, 0, 50, 50], "A")]
        pairs = match_tables(b, c, bbox_iou_floor=0.5)
        # Each pair should be a (bi, ci) — gather by baseline index
        by_bi = {bi: ci for bi, ci in pairs if bi is not None and ci is not None}
        # baseline[0] = A should match whichever candidate has bbox A
        a_match = by_bi[0]
        a_text = c[a_match]["data"]["grid"][0][0]["text"]
        self.assertEqual(a_text, "A", "baseline[0]=A must not be positionally zipped to candidate[0]=B")
        b_match = by_bi[1]
        b_text = c[b_match]["data"]["grid"][0][0]["text"]
        self.assertEqual(b_text, "B")

    def test_page_filter(self):
        # Table on page 1 in baseline, page 2 in candidate = different page
        b = [self._t(1, [0, 0, 50, 50], "X")]
        c = [self._t(2, [0, 0, 50, 50], "X")]
        pairs = match_tables(b, c, bbox_iou_floor=0.5)
        # No match (different page)
        self.assertEqual(pairs, [(0, None), (None, 0)])

    def test_added_removed(self):
        b = [self._t(1, [0, 0, 50, 50], "X")]
        c = []  # table removed
        pairs = match_tables(b, c, bbox_iou_floor=0.5)
        self.assertEqual(pairs, [(0, None)])

    def test_table_added(self):
        b = []
        c = [self._t(1, [0, 0, 50, 50], "X")]
        pairs = match_tables(b, c, bbox_iou_floor=0.5)
        self.assertEqual(pairs, [(None, 0)])

    def test_iou_floor_filters_small_overlap(self):
        # Two boxes that touch but barely overlap
        b = [self._t(1, [0, 0, 100, 100])]
        c = [self._t(1, [99, 99, 200, 200])]  # tiny overlap
        pairs = match_tables(b, c, bbox_iou_floor=0.5)
        # IoU is tiny — should not match
        matched = [p for p in pairs if p[0] is not None and p[1] is not None]
        self.assertEqual(matched, [])


# ── 6. Per-table diff ───────────────────────────────────────────────

class TableDiffTests(unittest.TestCase):
    def _t(self, page, bbox, rows, texts):
        return {
            "data": {"num_rows": len(rows), "num_cols": max(len(r) for r in rows),
                     "grid": [
                         [{"text": t, "row": ri, "col": ci,
                           "row_span": 1, "col_span": 1,
                           "column_header": ri == 0, "row_header": False}
                          for ci, t in enumerate(row)]
                         for ri, row in enumerate(rows)
                     ]},
            "prov": [{"page_no": page, "bbox": bbox}],
        }

    def test_text_match(self):
        b = self._t(1, [0, 0, 10, 10], [["A", "B"]], ["A", "B"])
        c = self._t(1, [0, 0, 10, 10], [["A", "B"]], ["A", "B"])
        cats, notes, meta = diff_table(b, c)
        self.assertEqual(cats, [])

    def test_text_diff(self):
        b = self._t(1, [0, 0, 10, 10], [["A", "B"]], ["A", "B"])
        c = self._t(1, [0, 0, 10, 10], [["A", "C"]], ["A", "C"])
        cats, notes, meta = diff_table(b, c)
        self.assertIn("CELL_TEXT_DIFF", cats)
        self.assertEqual(meta["n_text_diff"], 1)

    def test_numeric_diff(self):
        b = self._t(1, [0, 0, 10, 10], [["100"]], ["100"])
        c = self._t(1, [0, 0, 10, 10], [["200"]], ["200"])
        cats, notes, meta = diff_table(b, c)
        self.assertIn("CELL_NUMERIC_DIFF", cats)
        self.assertEqual(meta["n_numeric_diff"], 1)

    def test_sign_diff(self):
        b = self._t(1, [0, 0, 10, 10], [["-5"]], ["-5"])
        c = self._t(1, [0, 0, 10, 10], [["5"]], ["5"])
        cats, notes, meta = diff_table(b, c)
        self.assertIn("CELL_SIGN_DIFF", cats)

    def test_shape_change(self):
        b = self._t(1, [0, 0, 10, 10], [["A", "B"]], ["A", "B"])
        c = self._t(1, [0, 0, 10, 10], [["A", "B", "C"]], ["A", "B", "C"])
        cats, _, _ = diff_table(b, c)
        self.assertIn("SHAPE_CHANGED", cats)

    def test_page_mismatch(self):
        b = self._t(1, [0, 0, 10, 10], [["A"]], ["A"])
        c = self._t(2, [0, 0, 10, 10], [["A"]], ["A"])
        cats, _, _ = diff_table(b, c)
        self.assertIn("PAGE_MISMATCH", cats)

    def test_header_flag_diff(self):
        # baseline: row 0 has column_header=True
        # candidate: same text but column_header=False
        b = {
            "data": {"num_rows": 1, "num_cols": 1, "grid": [
                [{"text": "A", "row": 0, "col": 0,
                  "row_span": 1, "col_span": 1,
                  "column_header": True, "row_header": False}]
            ]},
            "prov": [{"page_no": 1, "bbox": [0, 0, 10, 10]}],
        }
        c = json.loads(json.dumps(b))
        c["data"]["grid"][0][0]["column_header"] = False
        cats, _, meta = diff_table(b, c)
        self.assertIn("HEADER_FLAG_DIFF", cats)


# ── 7. Corpus diff (filesystem) ─────────────────────────────────────

class CorpusDiffTests(unittest.TestCase):
    def _book(self, root, name, tables):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps({
            "schema_name": "test", "tables": tables,
        }))
        return d

    def setUp(self):
        import shutil
        shutil.rmtree("/tmp/_test_semantic_corpus", ignore_errors=True)

    def test_book_missing_only_in_one(self):
        b_root = Path("/tmp/_test_semantic_corpus/b")
        c_root = Path("/tmp/_test_semantic_corpus/c")
        b_root.mkdir(parents=True, exist_ok=True)
        c_root.mkdir(parents=True, exist_ok=True)
        self._book(b_root, "only_b", [{"data": {}, "prov": [{"page_no": 1, "bbox": [0, 0, 1, 1]}]}])
        self._book(c_root, "only_c", [{"data": {}, "prov": [{"page_no": 1, "bbox": [0, 0, 1, 1]}]}])
        rep = diff_corpus(b_root, c_root)
        self.assertEqual(rep["n_books_only_b"], 1)
        self.assertEqual(rep["n_books_only_c"], 1)


# ── 8. Noise envelope ───────────────────────────────────────────────

class NoiseEnvelopeTests(unittest.TestCase):
    def _book_dir(self, root, name, tables):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps({
            "schema_name": "test", "tables": tables,
        }))
        return d

    def setUp(self):
        import shutil
        for r in ["/tmp/_test_noise_b1", "/tmp/_test_noise_b2",
                  "/tmp/_test_noise_b", "/tmp/_test_noise_c"]:
            shutil.rmtree(r, ignore_errors=True)

    def test_overlap_and_change_only(self):
        # Build: noise1 has 1 dirty table; baseline has the SAME table
        # also dirty; candidate has THAT table dirty + 1 extra dirty table.
        # Expected: overlap=1, change_only=1.
        for r in ["/tmp/_test_noise_b1", "/tmp/_test_noise_b2"]:
            Path(r).mkdir(parents=True, exist_ok=True)
        # noise1: book A, table at page 1 bbox 0,0,10,10 with cell "X" (dirty)
        # noise2: same book, SAME table but cell "Y" (so diff is "X→Y")
        for r, cell in [("/tmp/_test_noise_b1", "X"),
                        ("/tmp/_test_noise_b2", "Y")]:
            self._book_dir(r, "A", [{
                "data": {"num_rows": 1, "num_cols": 1, "grid": [
                    [{"text": cell, "row": 0, "col": 0,
                      "row_span": 1, "col_span": 1,
                      "column_header": False, "row_header": False}]
                ]},
                "prov": [{"page_no": 1, "bbox": [0, 0, 10, 10]}],
            }])
        # baseline: A table with cell "X"  (matches noise1)
        self._book_dir("/tmp/_test_noise_b", "A", [{
            "data": {"num_rows": 1, "num_cols": 1, "grid": [
                [{"text": "X", "row": 0, "col": 0,
                  "row_span": 1, "col_span": 1,
                  "column_header": False, "row_header": False}]
            ]},
            "prov": [{"page_no": 1, "bbox": [0, 0, 10, 10]}],
        }])
        # candidate: SAME table dirty (cell "Y") + new table at page 2 dirty
        self._book_dir("/tmp/_test_noise_c", "A", [
            {  # same as baseline, also dirty in noise
                "data": {"num_rows": 1, "num_cols": 1, "grid": [
                    [{"text": "Y", "row": 0, "col": 0,
                      "row_span": 1, "col_span": 1,
                      "column_header": False, "row_header": False}]
                ]},
                "prov": [{"page_no": 1, "bbox": [0, 0, 10, 10]}],
            },
            {  # new table
                "data": {"num_rows": 1, "num_cols": 1, "grid": [
                    [{"text": "Z", "row": 0, "col": 0,
                      "row_span": 1, "col_span": 1,
                      "column_header": False, "row_header": False}]
                ]},
                "prov": [{"page_no": 2, "bbox": [0, 0, 10, 10]}],
            },
        ])
        rep = noise_envelope(
            Path("/tmp/_test_noise_b1"),
            Path("/tmp/_test_noise_b2"),
            Path("/tmp/_test_noise_b"),
            Path("/tmp/_test_noise_c"),
        )
        # The page=1 table should be in BOTH noise and change
        td = rep["category_breakdown"]["CELL_TEXT_DIFF"]
        # noise: 1 table (page 1, A, X→Y)
        # change: 2 tables (page 1 X→Y AND page 2 added Z)
        # BUT page 2 is TABLE_ADDED not CELL_TEXT_DIFF (no baseline match)
        # overlap: 1 (page 1), change_only: 0
        self.assertEqual(td["noise"], 1)
        self.assertEqual(td["overlap"], 1)
        self.assertEqual(td["change_only"], 0)
        # TABLE_ADDED: change=1, noise=0, change_only=1
        ta = rep["category_breakdown"]["TABLE_ADDED"]
        self.assertEqual(ta["change"], 1)
        self.assertEqual(ta["change_only"], 1)


# ── 9. Identity guarantee under permutation ─────────────────────────

class IdentityGuaranteeTests(unittest.TestCase):
    def _t(self, page, bbox, text):
        return {
            "data": {"num_rows": 1, "num_cols": 1, "grid": [
                [{"text": text, "row": 0, "col": 0}]
            ]},
            "prov": [{"page_no": page, "bbox": bbox}],
        }

    def test_match_invariant_under_permutation(self):
        b = [self._t(1, [0, 0, 10, 10], "A"),
             self._t(1, [20, 0, 30, 10], "B"),
             self._t(2, [0, 0, 10, 10], "C")]
        c1 = list(b)  # same order
        c2 = [b[2], b[0], b[1]]  # permuted
        p1 = match_tables(b, c1, bbox_iou_floor=0.5)
        p2 = match_tables(b, c2, bbox_iou_floor=0.5)
        # Both should match each baseline to a candidate with the same text
        # — never positionally.
        def text_of_candidate(pairs, c_list, target_text):
            for bi, ci in pairs:
                if ci is not None and c_list[ci]["data"]["grid"][0][0]["text"] == target_text:
                    return bi
            return None
        for txt in ("A", "B", "C"):
            self.assertEqual(text_of_candidate(p1, c1, txt), text_of_candidate(p2, c2, txt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
