"""Speed-related guards for the sum-peak component (BUGS.md: fits rebuild identical pieces)."""
import numpy as np
import pytest

import scrappyfit.fitting.fit as fitmod
from scrappyfit.fitting.fit import Component, SumPeakComponent
from scrappyfit.fitting.peakshape import ShapePars, line_profile

N = 2048


def _sum_component():
    comps = [Component('A', [(1.0, 1.0), (1.1, 0.1)]),
             Component('B', [(2.0, 1.0)])]
    comp = SumPeakComponent(comps)
    comp.update({'A': 1e5, 'B': 5e4})
    pars = ShapePars(5.0, 3.0, 500.0, 200.0, e_low=1.0, e_high=10.0)
    return comp, pars


def test_each_distinct_sum_line_is_profiled_once(monkeypatch):
    """(i, j) and (j, i) are the same sum line and must not be profiled twice."""
    comp, pars = _sum_component()
    calls = []
    orig = fitmod.line_profile

    def counting(*a, **k):
        calls.append(a[:2])
        return orig(*a, **k)

    monkeypatch.setattr(fitmod, 'line_profile', counting)
    comp.profile(pars, N)
    distinct = len(set(comp._lines))
    assert len(comp._lines) > distinct          # duplicates exist in the list
    assert len(calls) == distinct, (len(calls), distinct, len(comp._lines))


def test_sum_profile_matches_naive_loop_bitwise():
    """The sum-peak profile must stay bit-identical to the naive per-line accumulation."""
    comp, pars = _sum_component()
    sh = 1.0 - comp.sum_deficit / 100.0
    f = np.zeros(N)
    for e, w in comp._lines:
        f += line_profile(sh * e, w, pars, N, do_tail=False)
    ref = f / f.sum()
    got = comp.profile(pars, N)
    assert np.array_equal(got, ref)
    assert got.tobytes() == ref.tobytes()


def test_erfc_saturated_is_erfc_bit_for_bit():
    from scipy.special import erfc
    from scrappyfit.fitting.peakshape import erfc_saturated
    t = np.concatenate([np.linspace(-1e3, 1e3, 200001),
                        np.linspace(-12.0, 45.0, 570001),
                        [np.nan, np.inf, -np.inf, -0.0, 0.0, 1e308, -1e308]])
    assert erfc_saturated(t).tobytes() == erfc(t).tobytes()
    assert erfc_saturated(-7.0) == erfc(-7.0)


@pytest.mark.parametrize('kind', ['flat', 'ramp'])
def test_plateau_window_matches_every_channel(kind):
    """The plateau evaluated only where it can be non-zero equals the
    evaluation over every channel, bit for bit."""
    from scipy.special import erfc
    from scrappyfit.fitting.fit import PileupPlateauComponent
    comp, pars = _sum_component()
    pl = PileupPlateauComponent(comp, kind)
    got = pl.profile(pars, N)
    # reference: the pre-2026-10-20 loop over all channels with plain erfc
    f = np.zeros(N)
    x = np.arange(N, dtype=float)
    wmax = max(w for _, _, w in comp._pairs)
    for ei, ej, w in comp._pairs:
        if w < 1e-4 * wmax:
            continue
        lo = float(pars.centroid(max(ei, ej)))
        hi = float(pars.centroid(ei + ej))
        if hi - lo < 1.0 or hi < 0 or lo > N:
            continue
        sg = float(pars.fwhm_channels(ei + ej)) / np.sqrt(8.0 * np.log(2.0))
        box = (erfc((lo - x) / (np.sqrt(2.0) * sg))
               - erfc((hi - x) / (np.sqrt(2.0) * sg))) / (2.0 * (hi - lo))
        if kind == 'ramp':
            box = box * np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 2.0
        f += w * box
    ref = f / f.sum()
    assert got.tobytes() == ref.tobytes()


def test_de_shelf_matches_the_old_formula_bitwise():
    from scipy.special import erfc
    from scrappyfit.fitting.peakshape import de_shelf
    x = np.arange(200.0, 420.0)
    for c_p, s_p, c_c, s_c, S1, S2 in [(400.3, 3.1, 250.7, 3.1, 6.0, 1.0),
                                       (390.0, 0.7, 380.5, 0.0, 0.2, 12.0)]:
        def Ep(S):
            S = max(float(S), 1e-3)
            z = (x - c_p) / (s_p * S) + 1.0 / (2.0 * S * S)
            z = np.clip(z, -700.0, 700.0)
            a = erfc((1.0 / np.sqrt(2.0)) * ((x - c_p) / s_p + 1.0 / S))
            b = erfc((1.0 / np.sqrt(2.0)) * ((x - c_c) / max(s_c, 1e-6) + 1.0 / S))
            return np.exp(z) * (a - b)
        ref = 0.5 * (Ep(S1) - Ep(S2))
        assert de_shelf(x, c_p, s_p, c_c, s_c, S1, S2).tobytes() == ref.tobytes()


def test_sum_profile_with_deficit_spread_matches_naive_loop_bitwise():
    comp, pars = _sum_component()
    comp.deficit_spread = 0.5
    got = comp.profile(pars, N)
    d = comp.sum_deficit
    shifts = list(1.0 - (d + np.linspace(0.0, 0.5, 9)) / 100.0)
    f = np.zeros(N)
    for e, w in comp._lines:
        for sh in shifts:
            f += line_profile(sh * e, w / len(shifts), pars, N, do_tail=False)
    ref = f / f.sum()
    assert got.tobytes() == ref.tobytes()
