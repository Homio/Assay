"""P29: scheduler-merged header_sigs (chunked calib partials) == worker-side calibrate_page_headers(doc)."""
import glob
import os
import random
import sys

import pytest

ROOT = "/home/homio/sda/1111"
sys.path.insert(0, f"{ROOT}/Assay")
SUBSET = f"{ROOT}/analysis/perf/track_p5/p26/subset_input_500"
N_BOOKS = int(os.environ.get("SEED_SIGS_N", "120"))


def _books():
    paths = sorted(glob.glob(f"{SUBSET}/*.pdf"))
    random.Random(20261021).shuffle(paths)
    return paths[:N_BOOKS]


@pytest.mark.parametrize("path", _books())
def test_seeded_sigs_equal_worker_scan(path):
    import fitz
    import yaml
    from assay import detection as d

    chunk = int(yaml.safe_load(open(f"{ROOT}/Assay/pipeline_config.yaml"))
                ["scheduler"]["calib_chunk_pages"])
    doc = fitz.open(path)
    n = len(doc)
    partials = []
    for lo in range(0, n, chunk):          # scheduler: calib chunks, any completion order
        partials.extend(d.calibrate_page_headers_scan_page(doc[i])
                        for i in range(lo, min(lo + chunk, n)))
    random.Random(n).shuffle(partials)
    d._hdr_sig_cache.clear()
    d._stmt_cap_cache.clear()
    sched = d.register_header_calibration(path, n, partials)

    d._hdr_sig_cache.clear()
    d._stmt_cap_cache.clear()
    worker_doc = fitz.open(path)
    assert worker_doc.name == path
    assert d.calibrate_page_headers(worker_doc) == sched
