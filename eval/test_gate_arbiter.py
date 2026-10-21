import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.gate_arbiter import binom_sf


def test_binom_sf_bounds():
    assert binom_sf(0, 0) == 1.0
    assert binom_sf(0, 10) == 1.0
    assert abs(binom_sf(10, 10) - 1 / 1024) < 1e-12


def test_binom_sf_symmetric_case_not_rejected():
    assert binom_sf(41, 92) > 0.5


def test_binom_sf_lopsided_case_rejected():
    assert binom_sf(80, 92) < 1e-6
