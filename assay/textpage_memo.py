"""
Per-page memo for ``page.get_text`` inside CPU workers (P29 / P30).

One detect/build task runs a dozen helpers that each call
``page.get_text("dict")`` on the same fitz.Page with no clip and no flags;
MuPDF rebuilds the full text page every time.  The result depends only on
the page, so the first call's dict is stored on the Page object and reused.
``doc[i]`` returns a fresh Page per task, so the memo never outlives a task.

P30 (``clip=True``): table helpers re-extract the same region several times
per table (``_table_char_index`` for each C3 guard and again for cell
assignment, ``_line_fragments`` for each rebuild check) with an identical
option and clip rectangle.  Those calls are memoized too, keyed on the
option and the exact clip coordinates; any other clip is a new key and
goes to PyMuPDF.

Every consumer reads the result and builds its own structures; none writes
into it (audited P29 / P30).  Calls with flags, textpage, sort or any other
argument go straight to PyMuPDF.

Switches: pipeline_config.yaml scheduler.worker_textpage_memo /
scheduler.worker_textpage_memo_clip.
"""

from __future__ import annotations

_ATTR = "_assay_dict_memo"
_CLIP_ATTR = "_assay_clip_memo"
_CLIP_OPTIONS = ("dict", "rawdict", "text")
_installed = False
_clip_enabled = False


def install(clip: bool = False) -> None:
    global _installed, _clip_enabled
    if clip:
        _clip_enabled = True
    if _installed:
        return
    import fitz

    orig = fitz.Page.get_text

    def get_text(self, *args, **kwargs):
        if not kwargs and len(args) == 1 and args[0] == "dict":
            memo = self.__dict__.get(_ATTR)
            if memo is None:
                memo = orig(self, "dict")
                self.__dict__[_ATTR] = memo
            return memo
        if (_clip_enabled and len(args) == 1 and args[0] in _CLIP_OPTIONS
                and len(kwargs) == 1 and kwargs.get("clip") is not None):
            r = fitz.Rect(kwargs["clip"])
            key = (args[0], r.x0, r.y0, r.x1, r.y1)
            memo = self.__dict__.get(_CLIP_ATTR)
            if memo is None:
                memo = self.__dict__[_CLIP_ATTR] = {}
            val = memo.get(key)
            if val is None:
                val = memo[key] = orig(self, args[0], clip=r)
            return val
        return orig(self, *args, **kwargs)

    get_text.__wrapped__ = orig
    fitz.Page.get_text = get_text
    _installed = True
