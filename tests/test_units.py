"""Unit tables — the lossless-switch invariants the panels rely on.
(Imports panels, which pulls PySide6 modules, but instantiates no widgets.)"""
from studio.dtypes import UNIT_PER_M
from studio.panels import _BASELINE_CFG, _BOXC_CFG, _BOXS_CFG, _ZFAR_CFG, _ZNEAR_CFG


def test_unit_factors():
    assert UNIT_PER_M == {"m": 1.0, "mm": 1000.0, "µm": 1_000_000.0}


def test_baseline_ranges_span_same_physical_range():
    """The Phase-2 fix: every unit's row must cover the SAME physical span, or
    cycling units silently clamps the baseline (a depth-scaling corruption)."""
    spans_m = {u: (lo / UNIT_PER_M[u], hi / UNIT_PER_M[u])
               for u, (lo, hi, _dec, _suf) in _BASELINE_CFG.items()}
    ref = spans_m["mm"]
    for u, span in spans_m.items():
        assert abs(span[0] - ref[0]) < 1e-12, u
        assert abs(span[1] - ref[1]) < 1e-9, u


def test_z_and_box_tables_are_exact_unit_scalings():
    for cfg in (_ZNEAR_CFG, _ZFAR_CFG):
        mm = cfg["mm"]
        for u in ("m", "µm"):
            f = UNIT_PER_M[u] / 1000.0
            row = cfg[u]
            # (min, max, default, step) scale exactly; fmt/suffix are cosmetic
            for i in range(4):
                assert abs(row[i] - mm[i] * f) < 1e-9 * max(1.0, abs(mm[i] * f)), (u, i)
    for cfg in (_BOXC_CFG, _BOXS_CFG):
        mm = cfg["mm"]
        for u in ("m", "µm"):
            f = UNIT_PER_M[u] / 1000.0
            row = cfg[u]
            for i in range(2):
                assert abs(row[i] - mm[i] * f) < 1e-9 * max(1.0, abs(mm[i] * f)), (u, i)
