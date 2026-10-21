"""Detector-derived width must follow E low (BUGS.md: detector width not
re-referenced when E low changes)."""
import math
import pathlib

import pytest

from scrappyfit import config
from scrappyfit.io.gpdetector import read_detector
from scrappyfit.session import Session

from labdata import SYNTHETIC, lab_spectrum

DATA = pathlib.Path(__file__).parent / 'data'
DET = DATA / 'Amptek_25_C1_deflector.detector'
CAL = (0.0019756398864463256, -0.4522001604198639)

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')


def _session():
    s = Session()
    # the width bookkeeping does not depend on the counts
    s.load(str(lab_spectrum('404004_sic.txt') or SYNTHETIC))
    s.set_calibration(*CAL)
    return s


def test_detector_width_follows_e_low_set_after_load_detector():
    """Guards noise staying referenced to the e_low at load_detector time."""
    det = read_detector(str(DET))
    w0, w1, a = det['w0'], det['w1'], CAL[0]
    new_elow = 0.17

    A = _session()
    A.options.e_low = new_elow
    A.load_detector(str(DET))

    B = _session()
    B.load_detector(str(DET))          # default e_low 0.20
    B.options.e_low = new_elow

    assert B.options.fano == pytest.approx(A.options.fano, rel=1e-9)
    assert B.options.noise == pytest.approx(A.options.noise, rel=1e-9)
    fano = math.sqrt(w1) / a
    assert B.options.noise ** 2 == pytest.approx(
        w0 / a**2 + fano**2 * new_elow, rel=1e-9)
    E = 1.74
    fwhm_kev = a * math.sqrt(B.options.noise ** 2
                             + B.options.fano ** 2 * (E - new_elow))
    assert fwhm_kev == pytest.approx(math.sqrt(w0 + w1 * E), rel=1e-9)


class _Captured(Exception):
    pass


def _carried_session(new_low=0.17, new_high=9.0):
    """A session holding converged start_pars at (0.20, 8.0), then new limits."""
    from scrappyfit.fitting.peakshape import ShapePars
    s = _session()
    s.options.e_low, s.options.e_high = 0.20, 8.0
    orig = ShapePars(12.0, 0.30, 100.0, 1.0 / CAL[0],
                     e_low=0.20, e_high=8.0)
    orig.a[0], orig.a[1] = 13.5, 0.27          # a fitted, not the detector's
    s.start_pars = orig
    s.options.e_low, s.options.e_high = new_low, new_high
    return s, orig


def _same_shape(got, orig):
    import numpy as np
    for E in (0.5, 1.74, 4.5, 7.0):
        assert float(got.fwhm_channels(E)) == pytest.approx(
            float(orig.fwhm_channels(E)), rel=1e-9), 'FWHM_ch at %g keV' % E
    return np


@pytest.mark.parametrize('frozen', [True, False])
def test_carried_start_pars_keep_width_and_centroid_when_range_changes(
        monkeypatch, frozen):
    """Guards start_pars.a staying referenced to the old e_low/eoc."""
    from scrappyfit.fitting import fit as fitmod
    from scrappyfit.fitting.peakshape import ShapePars
    s, orig = _carried_session()
    s.options.freeze_nonlinear = frozen
    seen = {}

    def fake(spectrum, a, b, comps, e_low, e_high, **kw):
        seen.update(a=list(kw['start_pars'].a), lo=e_low, hi=e_high)
        raise _Captured

    monkeypatch.setattr(fitmod, 'fit_spectrum', fake)
    with pytest.raises(_Captured):
        s.run_fit(['Si'])
    got = ShapePars(1, 1, 0, 1, e_low=seen['lo'], e_high=seen['hi'])
    got.a[:] = seen['a'][:len(got.a)]
    _same_shape(got, orig)
    for E in (0.5, 1.74, 4.5, 7.0):
        assert float(got.centroid(E)) == pytest.approx(
            float(orig.centroid(E)), rel=1e-9), 'centroid at %g keV' % E


def test_preview_keeps_carried_width_when_e_low_changes():
    """Guards preview() reusing start_pars.a against the new e_low."""
    s, orig = _carried_session(new_low=0.17, new_high=8.0)
    res = s.preview(['Si'])
    _same_shape(res.pars, orig)
    for E in (0.5, 1.74, 4.5, 7.0):     # the held calibration, not the boxes
        assert float(res.pars.centroid(E)) == pytest.approx(
            float(orig.centroid(E)), rel=1e-9), 'centroid at %g keV' % E
