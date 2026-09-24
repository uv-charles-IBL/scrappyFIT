"""Measured-response line shape (fitting.hypermet) and the free Kb/Ka split."""
import os
import numpy as np

from scrappyfit.fitting import hypermet as H
from scrappyfit.fitting.peakshape import ShapePars
from scrappyfit.session import _split_kbeta, _kbeta_free, FitOptions

RESP = os.path.join(os.path.dirname(__file__), '..', 'scrappyfit', 'resources', 'response',
                    'amptek_c1_25mm2_20260923.json')


def _pars(n=4096):
    # 2.645 eV/ch, offset -0.593 keV, referenced to eoc as fit_spectrum does
    a, b, elo, ehi = 0.0026447, -0.5932, 0.21, 6.0
    eoc = 0.7 * elo + 0.3 * ehi
    return ShapePars(33.0, 20.0, (eoc - b) / a, 1.0 / a, 0.0, 0.0, elo, ehi)


def test_profile_has_unit_area():
    r = H.EmpiricalResponse.load(RESP)
    p = _pars()
    for E in (0.392, 1.041, 1.74, 2.622, 5.895):
        f = H.line_profile(E, 1.0, p, 4096, r, with_artefact=False)
        assert abs(f.sum() - 1.0) < 0.01, (E, f.sum())


def test_al_artefact_only_above_al_edge():
    r = H.EmpiricalResponse.load(RESP)
    r.mu_al = lambda E: 1.0 / E ** 3          # any falling curve will do here
    assert r.al_fraction(1.4) == 0.0
    assert abs(r.al_fraction(1.7398) - r.al_frac_ref) < 1e-12
    assert r.al_fraction(5.9) < r.al_fraction(2.6) < r.al_fraction(1.74)


def test_tail_jumps_at_si_k_edge():
    r = H.EmpiricalResponse.load(RESP)
    below, above = r.params(1.83)[0], r.params(1.85)[0]   # f_tail
    assert above > 2 * below


def test_split_kbeta_and_selection():
    lines = [(5.8878, 0.29), (5.8988, 0.58), (6.4904, 0.08), (6.5352, 0.001)]
    ka, kb = _split_kbeta(lines)
    assert [e for e, _ in ka] == [5.8878, 5.8988]
    assert [e for e, _ in kb] == [6.4904, 6.5352]
    o = FitOptions()
    o.free_kbeta = {'Mn'}
    assert _kbeta_free(o, 25, 'Mn') and not _kbeta_free(o, 26, 'Fe')
    o.free_kbeta = True
    assert _kbeta_free(o, 25, 'Mn') and not _kbeta_free(o, 12, 'Mg')
