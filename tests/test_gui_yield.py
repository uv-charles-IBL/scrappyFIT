"""GUI: the sample's .yield is loaded by default in the Layer stack dialog.

Synthetic spectrum (O and Si Gaussians on a flat background), real
MainWindow, offscreen Qt (conftest sets QT_QPA_PLATFORM).
"""
import os

import numpy as np
import pytest

from scrappyfit import config

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

QtCore = pytest.importorskip('PyQt5.QtCore')
QtWidgets = pytest.importorskip('PyQt5.QtWidgets')

GAIN, OFFSET = 0.01, 0.0          # keV/channel; O Ka 0.525, Si Ka 1.740


def _write_spectrum(path):
    x = np.arange(1024)
    E = GAIN * x + OFFSET
    y = np.full(x.size, 20.0)
    for e, area in ((0.525, 20000.0), (1.740, 30000.0)):
        s = (0.06 + 0.0 * e) / GAIN
        c = (e - OFFSET) / GAIN
        y += area * np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    y = np.random.default_rng(1).poisson(y)
    path.write_text('\n'.join('%d %d' % (i, v) for i, v in enumerate(y)))


@pytest.fixture
def window(qtbot):
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w.close()


def _fit_and_export(qtbot, w, monkeypatch, tmp_path):
    """Open a synthetic spectrum, fit O and Si, File > Export yield.

    Returns (spectrum path, exported .yield path).
    """
    spec = tmp_path / 'synth.txt'
    _write_spectrum(spec)
    out = str(tmp_path / 'sample.yield')
    QFD = QtWidgets.QFileDialog
    monkeypatch.setattr(QFD, 'getSaveFileName', staticmethod(
        lambda *a, **k: (out, '')))

    assert w.open_path(str(spec))
    w.ed_gain.setText('%.10g' % GAIN)
    w.ed_off.setText('%.10g' % OFFSET)
    w.ed_gain.editingFinished.emit()
    for sym in ('O', 'Si'):
        qtbot.mouseClick(w.elements.table.buttons[sym], QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                    timeout=60000)
    qtbot.mouseClick(w.btn_fit, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                    timeout=60000)
    assert w.session.fit is not None

    w.on_export_yield()
    assert os.path.exists(out), 'setup: yield export did not write the file'
    return spec, out


def test_layer_stack_has_sample_yield_loaded_after_export(
        qtbot, window, monkeypatch, tmp_path):
    """Layer stack must open with the exported .yield already loaded.

    BUGS.md 2026-10-20: "Yield file not loaded by default in the sample's
    dialogs".
    """
    w = window
    _, out = _fit_and_export(qtbot, w, monkeypatch, tmp_path)

    w.on_layers()
    assert w.session.yield_path == out
    assert w.layerview.layers
    assert os.path.basename(out) in w.layerview.lbl_yield.text()
    assert 'exported' in w.layerview.lbl_yield.text()


def test_sample_model_shows_yield_and_copies_it(
        qtbot, window, monkeypatch, tmp_path):
    w = window
    _, out = _fit_and_export(qtbot, w, monkeypatch, tmp_path)
    assert os.path.basename(out) in w.sample.lbl_yield.text()
    assert w.sample.btn_ycopy.isEnabled()
    # QUANTIFY reads the sample model, not the file: showing it changes
    # nothing until the user copies it in
    assert w.sample.matrix() is None
    w.sample.btn_ycopy.click()
    from scrappyfit.io.gpyield import read_yield
    L = read_yield(out)[0]['layers'][0]
    m = w.sample.matrix()
    assert sorted(m) == sorted(L['Z'])
    assert np.allclose([m[z] for z in L['Z']], L['W'], rtol=1e-3)
    assert w.sample.thickness() is None          # 1e4 mg/cm2 = thick target


def test_reopening_the_data_finds_its_yield_and_dialogs_follow(
        qtbot, window, monkeypatch, tmp_path):
    w = window
    spec, out = _fit_and_export(qtbot, w, monkeypatch, tmp_path)
    w.on_layers()
    other = tmp_path / 'other.txt'
    _write_spectrum(other)
    assert w.open_path(str(other))
    assert w.session.yield_path is None
    assert 'none' in w.sample.lbl_yield.text()

    # sample.yield's provenance sidecar names synth.txt as its source
    assert w.open_path(str(spec))
    assert w.session.yield_path == out
    assert w.session.yield_source == 'found beside the data'
    assert 'found beside the data' in w.layerview.lbl_yield.text()
    assert os.path.basename(out) in w.sample.lbl_yield.text()


def test_loading_a_yield_in_the_layer_dialog_sets_it_for_the_session(
        qtbot, window, monkeypatch, tmp_path):
    w = window
    _, out = _fit_and_export(qtbot, w, monkeypatch, tmp_path)
    copy = tmp_path / 'elsewhere' / 'picked.yield'
    copy.parent.mkdir()
    copy.write_bytes(open(out, 'rb').read())
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (str(copy), '')))
    w.on_layers()
    w.layerview.on_load_yield()
    assert w.session.yield_path == str(copy)
    assert w.session.yield_source == 'loaded'
    assert 'picked.yield' in w.layerview.lbl_yield.text()
    assert 'picked.yield' in w.sample.lbl_yield.text()

    # another sample, then the same file picked again from Sample model:
    # the layer dialog must say so, not keep its "none" label
    other = tmp_path / 'other.txt'
    _write_spectrum(other)
    assert w.open_path(str(other))
    assert 'none' in w.layerview.lbl_yield.text()
    w.sample._pick_yield()
    assert w.session.yield_path == str(copy)
    assert w.layerview.lbl_yield.text().startswith('yield file: picked.yield')


def test_find_yield_beside_rules(tmp_path):
    from scrappyfit.session import find_yield_beside
    d = tmp_path
    data = d / 'run7.txt'
    data.write_text('0 1')
    (d / 'stranger.yield').write_bytes(b'')
    assert find_yield_beside(data) is None       # no name or sidecar match
    (d / 'run7.yield').write_bytes(b'')
    assert find_yield_beside(data) == str(d / 'run7.yield')
    (d / 'run7.yield').unlink()
    import json
    for name in ('a', 'b'):
        (d / (name + '.yield')).write_bytes(b'')
        (d / (name + '.provenance.json')).write_text(
            json.dumps(dict(source_spectrum=str(data))))
    assert find_yield_beside(data) is None       # two claim it: ambiguous
    (d / 'b.provenance.json').unlink()
    assert find_yield_beside(data) == str(d / 'a.yield')
    # a malformed sidecar is skipped, never raised: it must not stop a load
    (d / 'b.provenance.json').write_text(json.dumps(dict(source_spectrum=7)))
    (d / 'c.yield').write_bytes(b'')
    (d / 'c.provenance.json').write_text('{not json')
    assert find_yield_beside(data) == str(d / 'a.yield')


def test_copy_takes_the_unknown_layer_of_a_multilayer_file(qtbot, tmp_path):
    """UNKNOWN is 1-based; a two-layer file copies that layer and says the
    other was dropped. Weight fractions go in, not the file's atom F."""
    from scrappyfit.gui.dialogs import SampleModelDialog
    from scrappyfit.io.gpyield import read_yield
    from scrappyfit.io.gpyield_write import write_yield
    from scrappyfit.physics.gpdb import Database
    db = Database()
    top = dict(zlist=[6], afrac=[1.0], thick=0.5, formula='C', name='coat')
    sub = dict(zlist=[8, 14], afrac=[2 / 3, 1 / 3], thick=1e4,
               formula='SiO2', name='quartz')
    path = write_yield(tmp_path / 'two.yield', db, [(14, 1)], [top, sub],
                       np.array([[1.0], [1.0]]), unknown=2, theta=150.0)
    dlg = SampleModelDialog(db)
    qtbot.addWidget(dlg)
    dlg.show_yield(path, 'loaded')
    what = dlg.copy_from_yield()
    W = read_yield(path)[0]['layers'][1]['W']
    m = dlg.matrix()
    assert sorted(m) == [8, 14]
    assert np.allclose([m[8], m[14]], W, rtol=1e-3)
    assert abs(m[14] - 0.4674) < 2e-3            # Si weight fraction of SiO2
    assert dlg.thickness() is None
    assert dlg.theta() == 150.0
    assert 'layer 2 of 2' in what and 'NOT copied' in what
    # the thin coat as UNKNOWN: its thickness comes across
    path1 = write_yield(tmp_path / 'coat.yield', db, [(14, 1)], [top, sub],
                        np.array([[1.0], [1.0]]), unknown=1)
    dlg.show_yield(path1, 'loaded')
    dlg.copy_from_yield()
    assert sorted(dlg.matrix()) == [6]
    assert abs(dlg.thickness() - 0.5) < 1e-6
