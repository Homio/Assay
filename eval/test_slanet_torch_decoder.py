"""GPU checks for the torch CUDA-Graph SLANet decoder (assay/slanet_torch_decoder.py).

Run on the PRO 6000 host with the default ORT TF32 precision:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python3 -m pytest -q eval/test_slanet_torch_decoder.py

Both tests check batch invariance (B=1 vs B=32/48) with EOS truncation.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CROPS = ROOT.parent / "analysis/perf/track_p5/inputs/slanet_crops.npy"

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "PRO 6000" not in torch.cuda.get_device_name(0)
    or not CROPS.exists(),
    reason="needs RTX PRO 6000 as device 0 and slanet_crops.npy",
)


@pytest.fixture(scope="module")
def sessions():
    from assay.models import TableStructureModel
    from assay.slanet_torch_decoder import maybe_install

    cfg = yaml.safe_load((ROOT / "pipeline_config.yaml").read_text())["gpu_server"]
    cfg = dict(cfg, slanet_torch_decoder=True, slanet_decoder_tf32=False)
    m = TableStructureModel(use_gpu=True)
    holder = m.engine.table_structure.session
    ort = holder.session
    assert maybe_install(m, cfg)
    return ort, holder.session, np.load(CROPS)


def _tokens(run, x, bs):
    """Returns token arrays truncated at first EOS (49)."""
    out = []
    for i in range(0, len(x), bs):
        probs = run(np.ascontiguousarray(x[i:i + bs]))
        for p in probs:
            t = p.argmax(-1)
            eos = np.nonzero(t == 49)[0]
            n = (eos[0] + 1) if eos.size else len(t)
            out.append(t[:n])
    return out


def test_torch_batch_invariant(sessions):
    _, td, x = sessions
    run = lambda b: td._run({"x": b})[1]
    one = _tokens(run, x, 1)
    for bs in (32, 48):
        got = _tokens(run, x, bs)
        bad = [k for k in range(len(x)) if not np.array_equal(got[k], one[k])]
        assert bad == [], f"bs={bs} differs from bs=1 at {bad}"
    assert td.fallback_count == 0


def test_ort_batch_invariant(sessions):
    ort, _, x = sessions
    run = lambda b: ort.run(None, {"x": b})[1]
    one = _tokens(run, x, 1)
    for bs in (32, 48):
        got = _tokens(run, x, bs)
        bad = [k for k in range(len(x)) if not np.array_equal(got[k], one[k])]
        assert bad == [], f"ORT bs={bs} differs from bs=1 at {bad}"
