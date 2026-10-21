"""P29: memoized page.get_text("dict") is value-identical to PyMuPDF and scoped to one Page object."""
import glob
import random
import sys

ROOT = "/home/homio/sda/1111"
sys.path.insert(0, f"{ROOT}/Assay")


def _pages(n=150):
    import fitz
    paths = sorted(glob.glob(f"{ROOT}/original/*.pdf"))
    rnd = random.Random(20261021)
    out = []
    for p in paths:
        doc = fitz.open(p)
        for i in rnd.sample(range(len(doc)), min(12, len(doc))):
            out.append((doc, i))
    return out[:n]


def test_memo_equals_pymupdf_and_is_page_scoped():
    import fitz
    from assay import textpage_memo
    orig = getattr(fitz.Page.get_text, "__wrapped__", fitz.Page.get_text)
    textpage_memo.install()
    for doc, i in _pages():
        page = doc[i]
        ref = orig(doc[i], "dict")
        a = page.get_text("dict")
        assert a == ref
        assert page.get_text("dict") is a            # reused within the Page
        assert doc[i].get_text("dict") is not a      # fresh Page = fresh extraction
        assert page.get_text("text") == orig(page, "text")
        r = fitz.Rect(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1 / 2)
        assert page.get_text("dict", clip=r) == orig(page, "dict", clip=r)
