"""P30: clip-keyed memo of page.get_text is value-identical to PyMuPDF and scoped to one Page object."""
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


def _boxes(page, rnd, k=6):
    r = page.rect
    out = []
    for _ in range(k):
        x0 = rnd.uniform(r.x0, r.x1 * 0.6)
        y0 = rnd.uniform(r.y0, r.y1 * 0.6)
        out.append((x0, y0, x0 + rnd.uniform(40, r.width * 0.4),
                    y0 + rnd.uniform(20, r.height * 0.4)))
    return out


def test_clip_memo_equals_pymupdf_and_is_page_scoped():
    import fitz
    from assay import textpage_memo
    orig = getattr(fitz.Page.get_text, "__wrapped__", fitz.Page.get_text)
    prev = textpage_memo._clip_enabled
    textpage_memo.install(clip=True)
    try:
        rnd = random.Random(30)
        n = 0
        for doc, i in _pages():
            page = doc[i]
            for b in _boxes(page, rnd):
                for opt in ("dict", "rawdict", "text"):
                    ref = orig(doc[i], opt, clip=fitz.Rect(*b))
                    a = page.get_text(opt, clip=fitz.Rect(*b))
                    assert a == ref
                    # same option + same clip -> same object (tuple clip too)
                    assert page.get_text(opt, clip=b) is a
                    # a fresh Page re-extracts (CPython shares "" and 1-char str objects)
                    if opt != "text" or len(a) > 1:
                        assert doc[i].get_text(opt, clip=fitz.Rect(*b)) is not a
                    # a slightly different clip is a different key
                    b2 = (b[0], b[1], b[2] + 0.01, b[3])
                    assert page.get_text(opt, clip=fitz.Rect(*b2)) == \
                        orig(page, opt, clip=fitz.Rect(*b2))
                    n += 1
            # unclipped dict memo still independent of the clip memo
            assert page.get_text("dict") == orig(page, "dict")
            assert page.get_text("text") == orig(page, "text")
            assert page.get_text("rawdict") == orig(page, "rawdict")
            # non-memoized argument forms pass through
            assert page.get_text("text", clip=fitz.Rect(*b), sort=True) == \
                orig(page, "text", clip=fitz.Rect(*b), sort=True)
        assert n > 1000
    finally:
        textpage_memo._clip_enabled = prev
