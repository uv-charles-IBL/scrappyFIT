"""GUI spectrum legend: modes, and the legend of a selected region.

Synthetic data only; offscreen Qt (conftest sets QT_QPA_PLATFORM).
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt5.QtWidgets')

N = 1024


@pytest.fixture
def canvas(qtbot):
    """A SpectrumCanvas in a QApplication, closed at teardown."""
    from scrappyfit.gui.canvases import SpectrumCanvas
    c = SpectrumCanvas()
    qtbot.addWidget(c)
    return c


@pytest.fixture
def window(qtbot):
    """A MainWindow, closed at teardown so its worker thread stops."""
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def _peak(c, s=8.0):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _drawn(canvas):
    """Show a spectrum with a fake fit: Si, O, Ca elements plus pileup."""
    E = 0.01 * np.arange(N)
    bk = np.full(N, 20.0)
    names = ['Si', 'O', 'Ca', 'pileup']
    prof = np.stack([_peak(c) / _peak(c).sum() for c in (170, 52, 370, 340)],
                    axis=1)
    areas = np.array([20000.0, 10000.0, 5000.0, 3000.0])
    model = bk + (prof * areas).sum(axis=1)
    r = SimpleNamespace(names=names, areas=areas, profiles=prof, model=model,
                        reduced_chi2=1.0)
    canvas.show(E, model.copy(), fit=r, background=bk)
    return [t.get_text() for t in canvas.ax.get_legend().get_texts()]


def _starts(texts, name):
    return any(t.startswith(name) for t in texts)


def test_legend_mode_all_lists_every_component(canvas):
    """Legend 'all' must give one entry per drawn component."""
    assert canvas.LEGEND_MODES == ('compact', 'all', 'off')
    _drawn(canvas)
    canvas.set_legend('all')
    texts = [t.get_text() for t in canvas.ax.get_legend().get_texts()]
    for nm in ('data', 'background', 'Si', 'O', 'Ca', 'pileup', 'fit'):
        assert _starts(texts, nm), (nm, texts)


def test_legend_default_is_compact_without_elements(canvas):
    """Compact legend must hold data, fit, background, artefacts only."""
    assert canvas.legend_mode == 'compact'
    texts = _drawn(canvas)
    for nm in ('data', 'fit', 'background', 'pileup'):
        assert _starts(texts, nm), (nm, texts)
    for nm in ('Si', 'O', 'Ca'):
        assert not _starts(texts, nm), (nm, texts)


def test_legend_off_removes_legend(canvas):
    """Legend 'off' must leave the axes with no legend at all."""
    _drawn(canvas)
    canvas.set_legend('off')
    assert canvas.ax.get_legend() is None


def test_reference_spectrum_is_drawn_with_its_label(canvas):
    """show(reference=...) must draw an extra line carrying its label."""
    E = 0.01 * np.arange(N)
    canvas.show(E, np.full(N, 50.0), data_label='region x',
                reference=(np.full(N, 90.0), 'whole field'))
    texts = [t.get_text() for t in canvas.ax.get_legend().get_texts()]
    assert 'region x' in texts and 'whole field' in texts, texts


def _window_with_mask(w):
    from test_gui_maps import _fake_listmode
    _fake_listmode(w.session)
    w.session.label = 'sample.lmf'
    w.session.set_calibration(0.01, 0.0)
    m = np.zeros((256, 256), bool)
    m[:128, :128] = True
    w.session.set_mask(m, 'roi1')
    w.sessionChanged.emit()
    return m


def _legend_texts(w):
    return [t.get_text() for t in w.spec.ax.get_legend().get_texts()]


def test_region_legend_names_region_pixels_counts(window):
    """With a mask the data entry must read 'region <name> <n> px <n> counts'."""
    w = window
    m = _window_with_mask(w)
    assert w.cmb_legend.currentText() == 'compact'
    w.cmb_legend.setCurrentText('all')
    data = _legend_texts(w)[0]
    npx = int(w.session.mask.sum())
    cnt = int(round(w.session.spectrum.sum()))
    assert data.startswith('region ') and 'roi1' in data, data
    assert '%d raster px' % npx in data and '%d counts' % cnt in data, data


def test_whole_field_overlay_only_with_mask(window):
    """chk_field must add a 'whole field' entry with a mask, none without."""
    w = window
    from test_gui_maps import _fake_listmode
    _fake_listmode(w.session)
    w.session.label = 'sample.lmf'
    w.session.set_calibration(0.01, 0.0)
    w.sessionChanged.emit()
    assert not w.chk_field.isChecked()
    w.chk_field.setChecked(True)
    assert not _starts(_legend_texts(w), 'whole field')
    _window_with_mask(w)
    assert _starts(_legend_texts(w), 'whole field'), _legend_texts(w)


# ------------------------------------------------------- journey on real data
#
# The legend on a region of 287424.lmf, driven like J6. It lives here, not in
# test_gui_journeys.py, and reuses that file's Driver and data paths; its
# steps have no budgets.

@pytest.mark.gui
@pytest.mark.slow
def test_jl_region_legend_on_287424(qtbot, tmp_path, monkeypatch):
    from scrappyfit import config
    if config.database_path(required=False) is None:
        pytest.skip('no GeoPIXE database available')
    import test_gui_journeys as J
    if not J.LMF_287424.exists():
        pytest.skip('%s not found' % J.LMF_287424)
    d = tmp_path / 'logs'
    monkeypatch.setenv('SCRAPPYFIT_LOG_DIR', str(d))
    from scrappyfit.gui import observe
    observe.install(log_dir=str(d))
    try:
        _legend_journey(J.Driver(qtbot, 'JL', d), J)
    finally:
        observe.uninstall()


def _legend_journey(d, J):
    w = d.win
    d.step('open', lambda: d.open(J.LMF_287424))
    d.calibrate(*J.LMF_CAL)
    d.set_range(0.2, 6.5)
    d.pick(J.LMF_ELS)
    d.settle()
    d.step('fit_field', d.fit)
    assert _legend_texts(w)[0] == 'data'      # no mask: plain data entry

    w.spin_bin.setValue(4)
    w.cmb_mapel.setCurrentText('Si')
    d.step('show_map', w.on_map, changes=False)
    w.spin_thr.setValue(90)
    d.step('mask', w.on_threshold)
    d.step('refit_region', d.fit)
    s = w.session

    # legend order is drawing order: whole field (if shown), then data
    def legend():
        lg = w.spec.ax.get_legend()
        return [] if lg is None else [t.get_text() for t in lg.get_texts()]
    compact = legend()
    assert compact[0].startswith('region %s: ' % s.mask_name), compact
    assert '%d raster px' % s.mask.sum() in compact[0], compact
    assert '%d counts' % round(s.spectrum.sum()) in compact[0], compact
    assert not any(t.split(' ')[0] in J.LMF_ELS for t in compact), compact
    d.step('legend_all', lambda: w.cmb_legend.setCurrentText('all'),
           changes=None)
    every = legend()
    assert {'O', 'Si'} <= {t.split(' ')[0] for t in every}, every
    assert len(every) > len(compact)
    d.step('legend_off', lambda: w.cmb_legend.setCurrentText('off'),
           changes=None)
    assert legend() == []
    w.cmb_legend.setCurrentText('compact')
    d.step('field_on', lambda: w.chk_field.setChecked(True))
    assert legend()[0].startswith('whole field'), legend()
    d.step('field_off', lambda: w.chk_field.setChecked(False))
    assert legend() == compact
