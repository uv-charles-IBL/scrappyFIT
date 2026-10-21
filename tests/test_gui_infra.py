"""The pieces under the GUI: cancel, the live estimate, the action log, jobs.

Fast; the full journeys are in test_gui_journeys.py.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

from scrappyfit import config

QtCore = pytest.importorskip('PyQt5.QtCore')

from labdata import SYNTHETIC, lab_spectrum

DATA = pathlib.Path(__file__).parent / 'data'
# lab spectra when present (tests/labdata.py), else the synthetic one
QUARTZ = lab_spectrum('quartz.txt') or SYNTHETIC
SIC = lab_spectrum('404004_sic.txt') or SYNTHETIC
need_db = pytest.mark.skipif(config.database_path(required=False) is None,
                             reason='no GeoPIXE database available')
QUARTZ_ELS = ['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si']


def _quartz():
    from scrappyfit.session import Session
    s = Session()
    s.load(str(QUARTZ))
    s.set_calibration(0.0016984, -0.37225)
    return s


def _records(folder):
    out = []
    for f in pathlib.Path(folder).glob('*.log'):
        out += [json.loads(x) for x in f.read_text(encoding='utf-8').splitlines()]
    return out


@pytest.fixture
def logs(tmp_path):
    from scrappyfit.gui import observe
    observe.install(log_dir=str(tmp_path))
    yield tmp_path
    observe.uninstall()


# ------------------------------------------------------------ library

@need_db
def test_run_fit_cancel_raises_and_keeps_no_result():
    from scrappyfit.fitting.fit import FitCancelled
    s = _quartz()
    with pytest.raises(FitCancelled):
        s.run_fit(QUARTZ_ELS, should_stop=lambda: True)
    assert s.fit is None


@need_db
def test_run_fit_cancel_mid_refinement():
    """Stop from inside the LM residual, not only between passes."""
    from scrappyfit.fitting.fit import FitCancelled
    s = _quartz()
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3          # pass 1 has started; stop inside it
    with pytest.raises(FitCancelled):
        s.run_fit(QUARTZ_ELS, should_stop=stop)
    assert len(calls) == 4


@need_db
def test_preview_follows_pixe_initial():
    s = _quartz()
    p = s.preview(QUARTZ_ELS)
    assert p.is_preview and s.fit is None
    assert list(p.names) == QUARTZ_ELS
    assert np.all(p.areas >= 3.0)                       # pixe_initial floor
    np.testing.assert_array_equal(p.background, s.background)
    np.testing.assert_allclose(p.model, p.profiles @ p.areas + p.background)


@need_db
@pytest.mark.slow
def test_preview_majors_near_the_fit():
    """The two majors, from their peak heights alone, within 30 % of a fit."""
    s = _quartz()
    p = s.preview(QUARTZ_ELS)
    est = dict(zip(p.names, p.areas))
    r = s.run_fit(QUARTZ_ELS)
    got = dict(zip(r.names, r.areas))
    for el in ('O', 'Si'):
        assert est[el] == pytest.approx(got[el], rel=0.3)


@need_db
def test_preview_recovers_a_known_peak_as_pixe_initial_would():
    """A synthetic Si peak of known area. pixe_initial.pro's estimate is the
    mean over +-FWHM/2 channels read as a Gaussian's height, so it comes back
    low by exactly that window's mean of a unit-height Gaussian. That factor
    is computed here from the Gaussian alone, not from the code under test."""
    from scrappyfit.session import Session
    s = Session()
    s.options.tail_amp = 0.0
    cal = (0.0016984, -0.37225)
    s.load_spectrum(np.full(4096, 20.0), cal=cal)
    p0 = s.preview(['Si'])
    s.load_spectrum(20.0 + 5e5 * p0.profiles[:, 0], cal=cal)
    p = s.preview(['Si'])
    E = s.db.line_list(14, 1)[0][0]
    x, d = float(p.pars.centroid(E)), float(p.pars.fwhm_channels(E))
    j = max(int(d / 2), 1)
    i = np.arange(int(x - j), int(x + j) + 1)
    window_mean = np.exp(-4 * np.log(2) * (i - x) ** 2 / d ** 2).sum() / (2 * j + 1)
    assert p.areas[0] / 5e5 == pytest.approx(window_mean, rel=0.01)


@need_db
def test_preview_leaves_the_fit_and_quantify_state_alone():
    s = _quartz()
    s._meta = [('sentinel', 1)]
    s.preview(['O', 'Si'])
    assert s._meta == [('sentinel', 1)] and s.fit is None


@need_db
@pytest.mark.slow
def test_batch_cancel_keeps_done_runs(tmp_path):
    from scrappyfit import batch
    seen = []
    res = batch.run_batch([QUARTZ] * 3, ['O', 'Si'], tmp_path,
                          calibration=(0.0016984, -0.37225),
                          progress=lambda i, n, p: seen.append(i),
                          should_stop=lambda: len(seen) >= 2)  # file 1 started
    assert len(res) == 1 and res[0].ok
    rows = (tmp_path / 'batch_summary.csv').read_text().strip().splitlines()
    assert len(rows) == 2


# ------------------------------------------------------------ action log

class _Win:
    busy = False

    def _log_context(self):
        return {'state': 'x'}

    def _refuse_while_busy(self, name):
        return self.busy

    def plain(self):
        return 'plain'

    def takes(self, a, b=2):
        return a + b


def test_user_action_drops_qt_extra_args_and_logs(logs):
    from scrappyfit.gui.observe import user_action
    W = type('W', (_Win,), {'plain': user_action(_Win.plain),
                            'takes': user_action(_Win.takes)})
    w = W()
    assert w.plain(False) == 'plain'          # clicked(bool) passes checked
    assert w.takes(1, 5, 'extra') == 6
    acts = [r for r in _records(logs) if r['event'] == 'action']
    assert [a['action'] for a in acts] == ['plain', 'takes']
    assert acts[1]['args'] == {'a': 1, 'b': 5}
    assert acts[1]['ctx'] == {'state': 'x'} and acts[1]['ms'] >= 0
    w.busy = True
    assert w.plain() is None
    assert _records(logs)[-1]['event'] == 'action refused'


def test_excepthook_logs_and_does_not_exit(logs):
    from scrappyfit.gui import observe
    try:
        raise ValueError('boom')
    except ValueError:
        sys.excepthook(*sys.exc_info())
    err = [r for r in _records(logs) if r['level'] == 'ERROR']
    assert err and err[0]['type'] == 'ValueError' and 'boom' in err[0]['traceback']
    assert observe.handler_count() == 1
    observe.install(log_dir=str(logs))          # idempotent
    assert observe.handler_count() == 1


def test_excepthook_after_a_widget_does_not_crash(qtbot, logs):
    """With a QApplication up (any widget test before this one), the error
    dialog used to be shown with QMessageBox, which crashes the interpreter on
    the offscreen platform. Now it is logged as not shown."""
    from PyQt5 import QtWidgets
    from scrappyfit.gui import observe
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    w.show()
    observe.uninstall()
    observe.install(log_dir=str(logs))           # notifier now exists
    before = observe._state.get('dialogs_suppressed', 0)
    try:
        raise ValueError('boom after widget')
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert observe._state.get('dialogs_suppressed', 0) == before + 1
    recs = _records(logs)
    assert any(r['event'] == 'error dialog not shown' for r in recs)
    assert any(r['level'] == 'ERROR' and 'boom after widget' in
               r.get('traceback', '') for r in recs)


# ------------------------------------------------------------ job runner

def test_jobrunner_coalesces_previews_and_cancels(qtbot, logs):
    import threading
    from scrappyfit.fitting.fit import FitCancelled
    from scrappyfit.gui.worker import Job, JobRunner
    jr = JobRunner()
    gate = threading.Event()
    ran, got = [], []

    def slow(tag):
        def fn(ctx):
            ran.append(tag)
            gate.wait(5)
            if ctx.should_stop():
                raise FitCancelled()
            return tag
        return fn

    jr.submit(Job('preview', slow('p1'), on_done=got.append))
    for t in ('p2', 'p3', 'p4'):
        jr.submit(Job('preview', slow(t), on_done=got.append))
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert ran == ['p1', 'p4'] and got == ['p4']   # p1 stale, p2/p3 replaced

    gate.clear()
    cancelled = []
    jr.submit(Job('fit', slow('f'), on_done=got.append,
                  on_cancel=lambda: cancelled.append(1)))
    assert jr.running_heavy()
    jr.cancel()
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert cancelled == [1] and got == ['p4']

    fails = []
    jr.submit(Job('fit', lambda ctx: 1 / 0, on_fail=fails.append))
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert 'ZeroDivisionError' in fails[0]
    assert any(r['event'] == 'job failed' and r['level'] == 'ERROR'
               for r in _records(logs))
    jr.shutdown()
    assert not jr.thread_running()
    assert jr.submit(Job('fit', lambda ctx: 1)) is None    # closed


def test_jobrunner_cancel_of_a_queued_fit_reaches_its_caller(qtbot, logs):
    """Cancel pressed while the fit still waits behind an estimate: the fit
    never runs, and its on_cancel must still fire, or the window stays
    locked for good."""
    import threading
    from scrappyfit.gui.worker import Job, JobRunner
    jr = JobRunner()
    gate = threading.Event()
    jr.submit(Job('preview', lambda ctx: gate.wait(5)))
    hit, ran = [], []
    jr.submit(Job('fit', lambda ctx: ran.append(1),
                  on_cancel=lambda: hit.append('cancel')))
    assert jr.running_heavy()
    jr.cancel()
    assert hit == ['cancel'] and not jr.running_heavy()
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert ran == []
    jr.shutdown()


def test_jobrunner_callback_can_submit(qtbot, logs):
    """A finishing job's callback runs with the runner free, so what it
    submits (a redraw scheduling an estimate) is started, not dropped."""
    from scrappyfit.gui.worker import Job, JobRunner
    jr = JobRunner()
    got = []
    jr.submit(Job('fit', lambda ctx: 'fit', on_done=lambda r: jr.submit(
        Job('preview', lambda ctx: 'est', on_done=got.append))))
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert got == ['est']
    jr.shutdown()


# ------------------------------------------------------------ the window

@pytest.fixture
def window(qtbot, logs):
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


@need_db
def test_range_edit_applies_before_a_detector_loads(window, monkeypatch):
    """E low takes effect on leaving the box, so the .detector width that
    follows is converted at the typed E low (CHANGELOG, Unreleased)."""
    from PyQt5 import QtWidgets
    w = window
    w.open_path(str(SIC))
    w.ed_elo.setText('0.17')
    w.ed_elo.editingFinished.emit()
    assert w.session.options.e_low == pytest.approx(0.17)
    det = DATA / 'Amptek_25_C1_deflector.detector'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (str(det), '')))
    w.on_load_detector()
    s = w.session
    a = s.cal[0]
    fano = s.detector['w1'] ** 0.5 / a
    assert s.options.fano == pytest.approx(fano)
    assert s.options.noise == pytest.approx(
        (s.detector['w0'] / a ** 2 + fano ** 2 * 0.17) ** 0.5)


@need_db
@pytest.mark.slow
def test_fit_uses_the_applied_calibration_not_the_typed_one(window, qtbot):
    """As before the worker: a gain typed but not applied (no Enter, no
    focus change) does not reach the fit, even though locking the panel
    moves focus out of the box."""
    w = window
    w.open_path(str(QUARTZ))
    w.ed_gain.setText('0.0016984')
    w.ed_off.setText('-0.37225')
    w.ed_gain.editingFinished.emit()
    applied = w.session.cal
    w.elements.set_selection([(8, 1), (14, 1)])
    qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                    timeout=20000)
    w.show()
    w.ed_gain.setFocus()
    w.ed_gain.setText('0.0018000')
    w.on_fit()
    assert w.jobs.running('fit')
    assert w.session.cal == applied
    w.on_cancel()
    qtbot.waitUntil(lambda: not w.jobs.busy, timeout=60000)
    assert w.session.cal == applied and w.btn_fit.isEnabled()
