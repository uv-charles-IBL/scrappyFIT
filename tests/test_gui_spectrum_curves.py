"""Escape and pile-up drawn as fitted curves, like GeoPIXE's display.

BUGS.md: Show escape and pile-up as components, like GeoPIXE. Synthetic data
only; offscreen Qt (conftest sets QT_QPA_PLATFORM).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from scrappyfit import config

N = 1024
BUG = 'BUGS.md: Show escape and pile-up as components, like GeoPIXE'


# ------------------------------------------------- A: the fit (library layer)

def _fit_once():
    from labdata import SYNTHETIC
    from scrappyfit.session import Session
    s = Session()
    s.load(str(SYNTHETIC))
    s.set_calibration(0.0016984, -0.37225)
    s.options.use_escape_peaks = True
    return s.run_fit(['O', 'Si', 'Ca'])


@pytest.mark.xfail(strict=True, reason=BUG)
def test_fit_result_exposes_escape_profile_and_spectrum():
    """The escape share of each component and of the fit must be separable."""
    if config.database_path(required=False) is None:
        pytest.skip('no GeoPIXE database available')
    from scrappyfit.fitting.fit import Component
    r1 = _fit_once()
    r2 = _fit_once()
    assert r1.areas.tolist() == r2.areas.tolist() and r1.chi2 == r2.chi2

    comps = r1.components
    n = len(r1.model)
    esc = [c for c in comps if isinstance(c, Component)
           and len(c.lines) > c.n_own]
    assert esc, 'no component carries escape lines (Ca should)'
    for c in esc:
        own = Component(c.name, c.lines[:c.n_own], c.tail_amp_fn,
                        c.tail_len_fn, c.beta_tail)
        own.response = c.response
        want = own.profile(r1.pars, n) + c.escape_profile(r1.pars, n)
        np.testing.assert_allclose(c.profile(r1.pars, n), want,
                                   rtol=1e-12, atol=0)
        assert c.escape_profile(r1.pars, n).sum() > 0

    expect = np.zeros(n)
    for a, c in zip(r1.areas, comps):
        if isinstance(c, Component):
            expect += a * c.escape_profile(r1.pars, n)
    got = r1.escape_spectrum()
    assert got is not None and got.sum() > 0
    np.testing.assert_allclose(got, expect, rtol=1e-12)


# ---------------------------------------------------- B: the canvas curves

def _peak(c, s=8.0):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _fake(canvas, comps=True):
    """Fake fit: Si, Ca, pileup, pileup flat, with an escape_spectrum."""
    pytest.importorskip('PyQt5.QtWidgets')
    E = 0.01 * np.arange(N)
    bk = np.full(N, 20.0)
    names = ['Si', 'Ca', 'pileup', 'pileup flat']
    prof = np.stack([_peak(c) / _peak(c).sum() for c in (170, 370, 340, 500)],
                    axis=1)
    areas = np.array([20000.0, 5000.0, 3000.0, 2000.0])
    esc = 300.0 * _peak(200, 10)
    model = bk + (prof * areas).sum(axis=1) + esc
    r = SimpleNamespace(names=names, areas=areas, profiles=prof, model=model,
                        reduced_chi2=1.0, escape_spectrum=lambda: esc)
    canvas.set_components(comps)
    canvas.show(E, model.copy(), fit=r, background=bk)
    return r, bk, esc


@pytest.fixture
def canvas(qtbot):
    from scrappyfit.gui.canvases import SpectrumCanvas
    c = SpectrumCanvas()
    qtbot.addWidget(c)
    return c


def _lines(canvas, prefix):
    return [ln for ln in canvas.ax.get_lines()
            if (ln.get_label() or '').startswith(prefix)]


def _legend(canvas):
    return [t.get_text() for t in canvas.ax.get_legend().get_texts()]


@pytest.mark.xfail(strict=True, reason=BUG)
def test_artefact_curves_draw_one_escape_and_one_pileup_line(canvas):
    """With the toggle on, escapes and summed pile-up must be single curves."""
    r, bk, esc = _fake(canvas)
    canvas.set_artefact_curves(True)
    e, p = _lines(canvas, 'escapes'), _lines(canvas, 'pile-up')
    assert len(e) == 1 and len(p) == 1, (e, p)
    np.testing.assert_allclose(p[0].get_ydata() - bk,
                               r.profiles[:, 2:] @ r.areas[2:])
    # the individual pileup component is not drawn on top of the sum
    assert not _lines(canvas, 'pileup')
    texts = _legend(canvas)
    assert any(t.startswith('escapes') for t in texts), texts
    assert any(t.startswith('pile-up') for t in texts), texts


@pytest.mark.xfail(strict=True, reason=BUG)
def test_artefact_curves_survive_components_off(canvas):
    """The curves are independent of the per-element component toggle."""
    _fake(canvas, comps=False)
    canvas.set_artefact_curves(True)
    assert len(_lines(canvas, 'escapes')) == 1
    assert len(_lines(canvas, 'pile-up')) == 1


@pytest.mark.xfail(strict=True, reason=BUG)
def test_artefact_curves_default_off_keeps_component_line(canvas):
    """Off by default: no curves, and the per-component pileup line stays."""
    _fake(canvas)
    assert getattr(canvas, '_artefact_curves', False) is False
    assert not _lines(canvas, 'escapes') and not _lines(canvas, 'pile-up')
    assert len(_lines(canvas, 'pileup')) >= 1
    canvas.set_artefact_curves(False)           # the API must exist
    assert not _lines(canvas, 'escapes')


# ---------------------------------------------------------------- C: the GUI

@pytest.mark.xfail(strict=True, reason=BUG)
def test_main_window_has_unchecked_curves_checkbox(qtbot):
    """MainWindow needs chk_curves, off by default, wired to the canvas."""
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        assert not w.chk_curves.isChecked()
        w.chk_curves.setChecked(True)
        assert w.spec._artefact_curves is True
    finally:
        w.close()
