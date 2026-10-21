"""GUI journeys: drive the real MainWindow the way a user does.

Each step is timed from the user's action to the window being settled (no
job running, the plot redrawn). Every step checks three things: nothing was
logged at error level, the plotted data changed when it should, and the step
took no longer than its budget in tests/gui_budgets.json.

Run alone:   python -m pytest -q -m gui
Record:      SCRAPPYFIT_TIMINGS_OUT=path.json python -m pytest -q -m gui
             writes each step's measured ms instead of only checking it.
Slower box:  SCRAPPYFIT_BUDGET_SCALE=2 multiplies every budget.
"""
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import pytest

from scrappyfit import config

from labdata import (SYNTHETIC, geopixe_file, lab_data_required,
                     lab_spectrum)

pytestmark = [
    pytest.mark.gui, pytest.mark.slow,
    pytest.mark.skipif(config.database_path(required=False) is None,
                       reason='no GeoPIXE database available'),
]

QtCore = pytest.importorskip('PyQt5.QtCore')
QtWidgets = pytest.importorskip('PyQt5.QtWidgets')

DATA = pathlib.Path(__file__).parent / 'data'
DET = DATA / 'Amptek_25_C1_deflector.detector'
BUDGETS = pathlib.Path(__file__).parent / 'gui_budgets.json'

# The lab spectra are not in the public repository (tests/labdata.py). Without
# them every journey runs on the synthetic quartz-like spectrum instead, so
# CI and a fresh clone still drive load, fit, detector, quantify and batch.
QUARTZ_CAL = (0.0016984, -0.37225)
QUARTZ_ELS = ['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si']
SIC_ELS = ['C', 'N', 'O', 'Al', 'Si']
_LAB = [lab_spectrum(n) for n in
        ('quartz.txt', '404004_sic.txt', '404003_ga2o3.txt')]
if all(_LAB):
    QUARTZ, SIC, _GA2O3 = _LAB
    SIC_CAL = (0.0019756398864463256, -0.4522001604198639)
    BATCH = [QUARTZ, SIC, _GA2O3]
elif lab_data_required():
    raise RuntimeError('lab spectra missing and SCRAPPYFIT_REQUIRE_LAB_DATA '
                       'is set (tests/labdata.py)')
else:
    QUARTZ = SIC = SYNTHETIC
    SIC_CAL = QUARTZ_CAL
    BATCH = [SYNTHETIC] * 3
# J6 needs list mode; the .lmf is 36 MB and lives outside the repo.
LMF_287424 = pathlib.Path(
    os.environ.get('SCRAPPYFIT_J6_LMF')
    or geopixe_file('data/lmf/287424.lmf') or 'absent/287424.lmf')
LMF_CAL = (0.0016984, -0.37225)          # golden recipe for 287424
LMF_ELS = ['O', 'Na', 'Si', 'Cl']

_TIMINGS = {}


def _budgets():
    try:
        return json.loads(BUDGETS.read_text())['budget_ms']
    except (OSError, KeyError, ValueError):
        return {}


def teardown_module(module):
    out = os.environ.get('SCRAPPYFIT_TIMINGS_OUT')
    if out and _TIMINGS:
        p = pathlib.Path(out)
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(_TIMINGS)
        p.write_text(json.dumps(old, indent=1, sort_keys=True))


def plotted(win):
    """Digest of what is actually drawn on the spectrum axes."""
    h = hashlib.sha1()
    for ln in win.spec.ax.get_lines():
        h.update(np.asarray(ln.get_ydata(), float).tobytes())
    return h.hexdigest()


class Driver:
    """A MainWindow plus the bookkeeping every journey step needs."""

    def __init__(self, qtbot, journey, log_dir):
        from scrappyfit.gui import app as gui_app
        self.qtbot = qtbot
        self.journey = journey
        self.log_dir = log_dir
        self.win = gui_app.MainWindow()
        qtbot.addWidget(self.win)
        self.win.show()
        self._errors_seen = 0

    # -- waiting and checking --------------------------------------------

    def settle(self, timeout=180000):
        w = self.win
        self.qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                             timeout=timeout)
        QtWidgets.QApplication.processEvents()

    def errors(self):
        """Error-level records in the action log since the window opened."""
        out = []
        for f in pathlib.Path(self.log_dir).glob('*.log'):
            for line in f.read_text(encoding='utf-8').splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get('level') in ('ERROR', 'CRITICAL'):
                    out.append(rec)
        return out

    def step(self, name, action, changes=True):
        """Run one user action, wait for it to settle, check the result."""
        before = plotted(self.win)
        t0 = time.perf_counter()
        action()
        self.settle()
        ms = 1e3 * (time.perf_counter() - t0)
        key = '%s.%s' % (self.journey, name)
        _TIMINGS[key] = round(ms, 1)

        errs = self.errors()
        assert not errs, '%s logged errors: %s' % (key, errs)
        after = plotted(self.win)
        if changes is not None:
            assert (after != before) == changes, (
                '%s: plot %s' % (key, 'did not change' if changes
                                 else 'changed unexpectedly'))
        budget = _budgets().get(key)
        if budget is not None and not os.environ.get('SCRAPPYFIT_TIMINGS_OUT'):
            scale = float(os.environ.get('SCRAPPYFIT_BUDGET_SCALE', '1'))
            assert ms <= budget * scale, '%s took %.0f ms, budget %.0f ms' % (
                key, ms, budget * scale)
        return ms

    # -- user actions ----------------------------------------------------

    def open(self, path):
        assert self.win.open_path(str(path))

    def calibrate(self, gain, offset):
        w = self.win
        w.ed_gain.setText('%.10g' % gain)
        w.ed_off.setText('%.10g' % offset)
        w.ed_gain.editingFinished.emit()

    def set_range(self, lo, hi):
        w = self.win
        w.ed_elo.setText('%g' % lo)
        w.ed_ehi.setText('%g' % hi)
        w.ed_ehi.editingFinished.emit()

    def click_element(self, sym):
        b = self.win.elements.table.buttons[sym]
        assert b.isEnabled(), '%s is not available in the range' % sym
        self.qtbot.mouseClick(b, QtCore.Qt.LeftButton)

    def pick(self, syms):
        for s in syms:
            self.click_element(s)

    def fit(self):
        self.qtbot.mouseClick(self.win.btn_fit, QtCore.Qt.LeftButton)

    def close(self):
        self.win.close()
        QtWidgets.QApplication.processEvents()


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    d = tmp_path / 'logs'
    monkeypatch.setenv('SCRAPPYFIT_LOG_DIR', str(d))
    from scrappyfit.gui import observe
    observe.install(log_dir=str(d))
    yield d
    observe.uninstall()


@pytest.fixture
def no_dialogs(monkeypatch, tmp_path):
    """Answer file dialogs and questions as a user would, without a screen."""
    answers = {}
    QFD = QtWidgets.QFileDialog
    monkeypatch.setattr(QFD, 'getOpenFileName', staticmethod(
        lambda *a, **k: (answers['open'], '')))
    monkeypatch.setattr(QFD, 'getOpenFileNames', staticmethod(
        lambda *a, **k: (answers['open_many'], '')))
    monkeypatch.setattr(QFD, 'getSaveFileName', staticmethod(
        lambda *a, **k: (answers['save'], '')))
    monkeypatch.setattr(QFD, 'getExistingDirectory', staticmethod(
        lambda *a, **k: answers['dir']))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'question', staticmethod(
        lambda *a, **k: QtWidgets.QMessageBox.Yes))
    return answers


def _fit_names(win):
    f = win.session.fit
    return set() if f is None else set(f.names)


# --------------------------------------------------------------- journeys

def test_j1_load_calibrate_pick_preview_fit(qtbot, log_dir):
    d = Driver(qtbot, 'J1', log_dir)
    d.step('load', lambda: d.open(QUARTZ))
    d.step('calibrate', lambda: d.calibrate(*QUARTZ_CAL), changes=None)
    d.step('pick', lambda: d.pick(QUARTZ_ELS))
    # the live overlay: an initial estimate is on screen before any fit
    p = d.win.preview
    assert p is not None and p.is_preview
    assert sum(a > 0 for a in p.areas) >= 3

    # while the fit runs: the window answers, and what would change the
    # session under it is locked
    w, ticks, seen = d.win, [], {}
    t = QtCore.QTimer()
    t.timeout.connect(lambda: ticks.append(1))

    def fit_and_watch():
        d.fit()
        seen['locked'] = (not w.btn_fit.isEnabled() and not w.ed_gain.isEnabled()
                          and not w.elements.table.isEnabled()
                          and w.btn_cancel.isVisible())
        t.start(50)
    d.step('fit', fit_and_watch)
    t.stop()
    assert seen['locked'], 'controls stayed live during the fit'
    assert len(ticks) >= 20, 'event loop starved during the fit (%d ticks)' % len(ticks)
    assert w.btn_fit.isEnabled() and w.elements.table.isEnabled()
    assert d.win.session.fit is not None
    assert {'O', 'Si'} <= _fit_names(d.win)


def test_j2_change_after_fit_refit(qtbot, log_dir):
    d = Driver(qtbot, 'J2', log_dir)
    d.open(QUARTZ)
    d.calibrate(*QUARTZ_CAL)
    d.pick(QUARTZ_ELS)
    d.settle()
    d.step('fit', d.fit)
    chi_a = d.win.session.fit.reduced_chi2

    # drop fluorine: the plot must follow at once (preview) and after refit
    d.step('drop_F', lambda: d.pick(['F']))
    d.step('refit_noF', d.fit)
    assert 'F' not in _fit_names(d.win)

    d.step('range', lambda: d.set_range(0.2, 2.5))
    d.step('refit_range', d.fit)
    assert d.win.session.options.e_high == pytest.approx(2.5)
    assert d.win.session.fit.reduced_chi2 != chi_a

    # five quick clicks, one estimate: the overlay is coalesced
    n0 = d.win.preview_runs
    d.step('rapid_clicks', lambda: [d.click_element(s) for s in
                                    ('Na', 'Na', 'Mg', 'Mg', 'Al')])
    assert d.win.preview_runs - n0 <= 2


def test_j3_detector_quantify_export(qtbot, log_dir, no_dialogs, tmp_path):
    d = Driver(qtbot, 'J3', log_dir)
    w = d.win
    d.open(SIC)
    d.calibrate(*SIC_CAL)
    d.set_range(0.17, 6.0)
    d.settle()
    no_dialogs['open'] = str(DET)
    d.step('detector', w.on_load_detector, changes=None)
    assert w.session.detector
    idx = [i for i in range(w.cmb_eff.count())
           if 'deflector' in w.cmb_eff.itemText(i).lower()][0]
    d.step('efficiency', lambda: w.cmb_eff.setCurrentIndex(idx), changes=None)
    d.pick(SIC_ELS)
    d.settle()
    d.step('fit', d.fit)
    d.step('quantify', w.on_quantify, changes=False)
    assert w.session.concentration_table()

    no_dialogs['dir'] = str(tmp_path / 'export')
    d.step('export', w.on_export, changes=False)
    assert any((tmp_path / 'export').iterdir())
    no_dialogs['save'] = str(tmp_path / 'sic.yield')
    d.step('export_yield', w.on_export_yield, changes=False)
    assert (tmp_path / 'sic.yield').exists()


def test_j4_batch_cancel_rerun(qtbot, log_dir, no_dialogs, tmp_path):
    d = Driver(qtbot, 'J4', log_dir)
    w = d.win
    d.open(QUARTZ)
    d.calibrate(*QUARTZ_CAL)
    d.pick(QUARTZ_ELS)
    d.settle()
    no_dialogs['open_many'] = [str(p) for p in BATCH]

    no_dialogs['dir'] = str(tmp_path / 'b1')
    w.on_batch()
    assert w.jobs.running('batch') and not w.btn_fit.isEnabled()
    # cancel as soon as the first file is done
    qtbot.waitUntil(lambda: w.batch_done >= 1, timeout=120000)
    t0 = time.perf_counter()
    w.on_cancel()
    d.settle()
    lat = 1e3 * (time.perf_counter() - t0)
    _TIMINGS['J4.cancel_latency'] = round(lat, 1)
    budget = _budgets().get('J4.cancel_latency')
    if budget and not os.environ.get('SCRAPPYFIT_TIMINGS_OUT'):
        scale = float(os.environ.get('SCRAPPYFIT_BUDGET_SCALE', '1'))
        assert lat <= budget * scale, 'cancel took %.0f ms' % lat
    assert w.last_batch is not None and len(w.last_batch) < len(BATCH)
    assert not d.errors()
    # nothing left disabled behind it
    assert w.btn_fit.isEnabled()

    no_dialogs['dir'] = str(tmp_path / 'b2')
    d.step('batch', w.on_batch, changes=False)
    summary = (tmp_path / 'b2' / 'batch_summary.csv').read_text()
    assert summary.count('\n') == len(BATCH) + 1


def test_j5_open_fit_close_twice(qtbot, log_dir):
    chis = []
    for k in (1, 2):
        d = Driver(qtbot, 'J5', log_dir)
        d.step('open%d' % k, lambda: d.open(QUARTZ))
        d.calibrate(*QUARTZ_CAL)
        d.pick(QUARTZ_ELS)
        d.settle()
        d.step('fit%d' % k, d.fit)
        chis.append(d.win.session.fit.reduced_chi2)
        jobs = d.win.jobs
        d.close()
        assert not jobs.thread_running(), 'worker thread outlived close'
    # a second window in the same process gives the same answer
    assert chis[0] == chis[1]
    from scrappyfit.gui import observe
    assert observe.handler_count() == 1, 'log handlers leak per window'


@pytest.mark.skipif(not LMF_287424.exists(),
                    reason='287424.lmf not present (SCRAPPYFIT_J6_LMF or '
                           'SCRAPPYFIT_GEOPIXE_DATA)')
def test_j6_map_mask_grow_shrink_reset_refit(qtbot, log_dir):
    """Maps: show a map (twice: it must not move), mask the top decile,
    grow, shrink, zoom and reset, and refit the region."""
    d = Driver(qtbot, 'J6', log_dir)
    w = d.win
    d.step('open', lambda: d.open(LMF_287424))
    assert w.session.events is not None
    d.calibrate(*LMF_CAL)
    d.set_range(0.2, 6.5)
    d.pick(LMF_ELS)
    d.settle()
    d.step('fit_field', d.fit)
    field_chi = w.session.fit.reduced_chi2

    w.spin_bin.setValue(4)              # displayed() assumes 64 x 64
    w.cmb_mapel.setCurrentText('Si')
    d.step('show_map', w.on_map, changes=False)
    assert w.map._data.shape == (64, 64)
    w.map.draw()
    pos = w.map.ax.get_position().bounds
    d.step('show_map_again', w.on_map, changes=False)
    w.map.draw()
    assert np.allclose(w.map.ax.get_position().bounds, pos)

    def displayed():
        return int(w.session.mask.reshape(64, 4, 64, 4).any(axis=(1, 3)).sum())

    w.spin_thr.setValue(90)
    d.step('mask', w.on_threshold)
    n0 = displayed()
    assert 0 < n0 < 64 * 64
    d.step('grow', lambda: w.on_morph(1))
    n1 = displayed()
    assert n1 > n0
    d.step('shrink', lambda: w.on_morph(-1))
    assert displayed() < n1

    w.map.ax.set_xlim(10, 20)
    w.map.ax.set_ylim(10, 20)
    d.step('reset', w.btn_map_reset.click, changes=False)
    assert np.allclose(w.map.ax.get_xlim(), (-0.5, 63.5))

    d.step('refit_region', d.fit)
    s = w.session
    assert s.fit is not None and s.mask is not None
    assert {'O', 'Si'} <= _fit_names(w)
    # the fit read the region, not the field
    assert s.spectrum.sum() < s.full_spectrum.sum()
    assert s.fit.reduced_chi2 != field_chi


def test_j7_misfit_then_refit_returns(qtbot, log_dir):
    """A fit with the wrong elements must not spoil the next one.

    BUGS.md "A misfit's line shape poisons every later fit": every fit used
    to start from the last one's converged shape, so on 287424 fit / drop Al
    Si C / add back went 2.382, 359.4, 8.411. Unless the shape is held each
    fit now starts from the detector, and the refit is the first fit again.
    """
    d = Driver(qtbot, 'J7', log_dir)
    d.open(QUARTZ)
    d.calibrate(*QUARTZ_CAL)
    d.pick(QUARTZ_ELS)
    d.settle()
    sel = d.win.selected_elements()
    d.step('fit', d.fit)
    first = d.win.session.fit
    chi_a, pars_a = first.reduced_chi2, first.pars.a.copy()

    d.pick(['Al', 'Si', 'C'])
    d.settle()
    d.step('misfit', d.fit)
    assert not {'Al', 'Si', 'C'} & _fit_names(d.win)

    d.pick(['Al', 'Si', 'C'])
    d.settle()
    assert d.win.selected_elements() == sel
    d.step('refit', d.fit)
    r = d.win.session.fit
    assert r.reduced_chi2 == pytest.approx(chi_a, rel=1e-9)
    np.testing.assert_allclose(r.pars.a, pars_a, rtol=1e-9, atol=1e-12)

    # held: the converged shape, areas only, the same fit
    d.win.chk_freeze.setChecked(True)
    d.settle()
    d.step('held', d.fit, changes=None)
    assert d.win.session.fit.reduced_chi2 == pytest.approx(chi_a, rel=1e-6)
    d.close()
