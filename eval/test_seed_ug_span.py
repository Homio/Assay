"""P29c: scheduler-merged double-underline threshold == worker-side calibrate_double_span(doc)."""
import glob
import random
import sys

import pytest

ROOT = "/home/homio/sda/1111"
sys.path.insert(0, f"{ROOT}/Assay")


def _books():
    sub = sorted(glob.glob(f"{ROOT}/analysis/perf/track_p5/p26/subset_input_500/*.pdf"))
    random.Random(20261021).shuffle(sub)
    return sorted(glob.glob(f"{ROOT}/original/*.pdf")) + sub[:80]


@pytest.mark.parametrize("path", _books())
def test_seeded_ug_span_equals_worker_scan(path):
    import fitz
    from assay import rulings as r

    doc = fitz.open(path)
    parts = [(i, r.double_span_scan_page(doc[i])) for i in range(len(doc))]
    random.Random(len(doc)).shuffle(parts)            # calib chunks finish in any order
    spans = []
    for _, s in sorted(parts):
        spans.extend(s)
    sched = r.double_span_threshold(spans)

    r._ug_span_cache.clear()
    assert r.calibrate_double_span(fitz.open(path)) == sched
