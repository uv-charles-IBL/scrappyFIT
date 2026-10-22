"""Journey JM: the Maps tab tools of fix/maps-phase3 on 287424.lmf.

Driven like J6 and reusing test_gui_journeys.py's Driver and data paths
(as test_gui_legend.py's JL does). Its steps have no budgets yet: each
step's time is checked only against nothing being logged at error level and
the window settling. Covers: window map with each background, the kind
stated on the map, a binning-3 rectangle mask, the periodic-table picker,
an association plot selecting a mask, and DA maps after loading a matrix.
"""
import numpy as np
import pytest

pytest.importorskip('PyQt5.QtWidgets')


@pytest.mark.gui
@pytest.mark.slow
def test_jm_map_kinds_picker_associations_on_287424(qtbot, tmp_path,
                                                    monkeypatch):
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
        _journey(J.Driver(qtbot, 'JM', d), J, monkeypatch)
    finally:
        observe.uninstall()


def _journey(d, J, monkeypatch):
    from PyQt5 import QtWidgets
    from scrappyfit.gui import maplines
    w = d.win
    s = w.session
    d.step('open', lambda: d.open(J.LMF_287424))
    d.calibrate(*J.LMF_CAL)
    d.set_range(0.2, 6.5)
    d.settle()

    # window maps: the kind and background are stated on the map
    w.spin_bin.setValue(4)
    w.cmb_mapel.setCurrentText('Si')
    d.step('show_map', w.on_map, changes=False)
    assert 'WINDOW, net counts' in w.map.ax.get_title()
    assert 'linear between flanks' in w.lbl_mapkind.text()
    net = w.map._data.copy()
    w.cmb_bg.setCurrentIndex(w.cmb_bg.findData('none'))
    d.step('show_map_raw', w.on_map, changes=False)
    assert 'WINDOW, raw counts' in w.map.ax.get_title()
    assert (w.map._data >= net - 1e-9).all()          # raw >= net
    # the field edge is not a dark rim any more (12df313)
    m = w.map._data
    edge = np.concatenate([m[0, 1:-1], m[-1, 1:-1], m[1:-1, 0], m[1:-1, -1]])
    assert edge.mean() > 0.8 * m[3:-3, 3:-3].mean()

    # a binning that does not divide 256: a rectangle is a mask (495194f)
    w.spin_bin.setValue(3)
    d.step('show_map_bin3', w.on_map, changes=False)
    assert w.map._data.shape == (86, 86)
    r = np.zeros((86, 86), bool)
    r[70:, 70:] = True
    d.step('mask_bin3', lambda: w.on_region(r))
    assert s.mask is not None and s.mask[255, 255] and s.mask.sum() == 46 * 46
    w.on_clear_mask()

    # the periodic table picks a line and maps it
    w.spin_bin.setValue(4)
    monkeypatch.setattr(maplines.MapLinePicker, 'exec_',
                        lambda self: self.pick('Ag L'))
    d.step('pick_line', w.on_map_pick, changes=False)
    assert w.cmb_mapel.currentText() == 'Ag L'
    assert w.map.ax.get_title().startswith('Ag L')

    # association plot: Si against O, the Si-rich cluster as the mask
    w.cmb_bg.setCurrentIndex(w.cmb_bg.findData('linear'))
    d.step('associations', w.on_associations, changes=False)
    a = w.assocview
    a.cmb_x.setCurrentText('Si')
    a.cmb_y.setCurrentText('O')
    d.step('assoc_plot', a.plot, changes=False)
    assert a.a.shape == (64, 64)
    top = np.percentile(a.a, 75)
    poly = [(top, -1e6), (1e6, -1e6), (1e6, 1e6), (top, 1e6)]
    d.step('assoc_select', lambda: a.select(poly))
    assert s.mask is not None and 'assoc Si-O' in s.mask_name
    frac = s.mask.mean()
    assert 0.15 < frac < 0.35                         # about a quarter
    a.close()

    # a DA matrix: the tab switches to DA maps, stated as wt%
    dam = J.geopixe_file('data/VOx_20240426/20240426/SioX.dam')
    if dam is None:
        return
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        lambda *a, **k: (str(dam), ''))
    d.step('load_dam', w.on_load_dam, changes=False)
    assert w.cmb_kind.currentText() == 'DA matrix'
    w.cmb_mapel.setCurrentText('Si')
    d.step('show_da_map', w.on_map, changes=False)
    assert 'DA, wt%' in w.map.ax.get_title()
    assert w.map._cbar.ax.get_ylabel() == 'wt%'
    assert 'SioX.dam' in w.lbl_mapkind.text()
