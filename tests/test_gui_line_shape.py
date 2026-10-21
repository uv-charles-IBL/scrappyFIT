"""GUI: what the next fit starts from after a completed fit.

Real MainWindow, offscreen Qt, synthetic spectrum. Session.run_fit is
replaced by a recorder, so no fit runs.
"""
import types

import pytest

from scrappyfit import config
from tests import labdata

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

QtCore = pytest.importorskip('PyQt5.QtCore')


@pytest.fixture
def window(qtbot):
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


CONVERGED = object()      # stands in for the ShapePars of a finished fit


def _fake_result():
    return types.SimpleNamespace(
        pars=CONVERGED, reduced_chi2=2.4, names=['Si'], areas=[1000.0],
        errors=[10.0])


def _second_fit_start(qtbot, w, monkeypatch, hold):
    """Complete one fit (fake), then press Fit again. Returns the
    session.start_pars that run_fit saw on the second call."""
    assert w.open_path(str(labdata.SYNTHETIC))
    qtbot.mouseClick(w.elements.table.buttons['Si'], QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                    timeout=60000)
    w.chk_freeze.setChecked(hold)
    seen = []

    def run_fit(self, els, **kw):
        seen.append(self.start_pars)
        return _fake_result()

    monkeypatch.setattr(type(w.session), 'run_fit', run_fit)
    w.on_fit()
    qtbot.waitUntil(lambda: not w.jobs.busy, timeout=10000)
    assert len(seen) == 1
    w.on_fit()
    qtbot.waitUntil(lambda: not w.jobs.busy, timeout=10000)
    assert len(seen) == 2
    return seen[1]


def test_next_fit_starts_fresh_when_line_shape_is_not_held(
        qtbot, window, monkeypatch):
    """A fit's converged shape must not seed the next fit unless 'Hold line shape' is ticked."""
    start = _second_fit_start(qtbot, window, monkeypatch, hold=False)
    assert start is None


def test_next_fit_holds_converged_shape_when_ticked(
        qtbot, window, monkeypatch):
    """With 'Hold line shape' ticked the next fit must start from the last converged shape."""
    start = _second_fit_start(qtbot, window, monkeypatch, hold=True)
    assert start is CONVERGED


def test_reset_line_shape_forgets_the_held_shape(qtbot, window, monkeypatch):
    """Reset line shape: a held fit refines from the detector again, once."""
    w = window
    _second_fit_start(qtbot, w, monkeypatch, hold=True)
    w.on_reset_shape()
    assert w.last_shape is None
    assert w.session.start_pars is None
    # nothing to hold, so the next fit is free, not frozen at the options
    assert w.session.options.freeze_nonlinear is False


def test_hold_with_no_fit_yet_refines_freely(qtbot, window, monkeypatch):
    """Ticked before any fit: the message says it refines freely, and it does."""
    w = window
    assert w.open_path(str(labdata.SYNTHETIC))
    w.chk_freeze.setChecked(True)
    seen = []

    def run_fit(self, els, **kw):
        seen.append((self.start_pars, self.options.freeze_nonlinear))
        return _fake_result()

    monkeypatch.setattr(type(w.session), 'run_fit', run_fit)
    qtbot.mouseClick(w.elements.table.buttons['Si'], QtCore.Qt.LeftButton)
    w.on_fit()
    qtbot.waitUntil(lambda: not w.jobs.busy, timeout=10000)
    assert seen == [(None, False)]


@pytest.mark.parametrize('carried, frozen, refines', [
    (False, False, False), (True, False, True), (True, True, False)])
def test_run_fit_refines_from_a_carried_start(monkeypatch, carried, frozen,
                                               refines):
    """Session API: a start_pars without the hold is a starting point, so
    the first pass refines from it; held, or with no start, it does not."""
    from scrappyfit.fitting import fit as fitmod
    from scrappyfit.fitting.peakshape import ShapePars
    from scrappyfit.session import Session

    class _Captured(Exception):
        pass

    s = Session()
    s.load(str(labdata.SYNTHETIC))
    o = s.options
    if carried:
        s.start_pars = ShapePars(o.noise, o.fano, 100.0, 500.0,
                                 e_low=o.e_low, e_high=o.e_high)
    o.freeze_nonlinear = frozen
    seen = {}

    def fake(*a, **kw):
        seen['refine_from_start'] = kw['refine_from_start']
        raise _Captured

    monkeypatch.setattr(fitmod, 'fit_spectrum', fake)
    with pytest.raises(_Captured):
        s.run_fit(['Si'])
    assert seen['refine_from_start'] is refines
