"""Association (correlation) plots of two maps, and a region drawn on one
selecting pixels on the map (BUGS.md: "Correlation (association) maps like
GeoPIXE")."""
import numpy as np
import pytest

from scrappyfit.analysis import association as AS


def _two_phases():
    """Left half phase 1 (A high, B low), right half phase 2 (A low, B high)
    with a little noise."""
    rng = np.random.default_rng(3)
    a = np.where(np.arange(32)[None, :] < 16, 10.0, 2.0) * np.ones((32, 32))
    b = np.where(np.arange(32)[None, :] < 16, 1.0, 8.0) * np.ones((32, 32))
    return a + rng.normal(0, 0.3, a.shape), b + rng.normal(0, 0.3, b.shape)


def test_histogram_counts_every_pixel_and_finds_two_clusters():
    a, b = _two_phases()
    H, xe, ye, lims = AS.histogram(a, b, bins=32)
    assert H.sum() == a.size
    assert lims[0][0] == 0 and lims[0][1] == pytest.approx(1.05 * a[1:-1, 1:-1].max(), rel=0.05)
    # two clusters, one per half of the map: (10, 1) and (2, 8)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    hi_a = H[xc > 6][:, yc < 4.5].sum()
    hi_b = H[xc < 6][:, yc > 4.5].sum()
    assert hi_a == hi_b == a.size // 2


def test_polygon_on_the_plot_selects_that_phase_on_the_map():
    a, b = _two_phases()
    m = AS.select(a, b, [(7, -1), (13, -1), (13, 4), (7, 4)])
    assert m.shape == a.shape
    assert m[:, :16].all() and not m[:, 16:].any()


def test_log_axes_and_negatives():
    a, b = _two_phases()
    a[0, 0] = -5.0                                  # a net map can go negative
    H, xe, ye, lims = AS.histogram(a, b, log=(True, False))
    assert xe[0] == pytest.approx(np.log10(lims[0][0]))
    assert H.sum() == a.size                        # clipped, not dropped
    m = AS.select(a, b, [(np.log10(7), -1), (np.log10(13), -1),
                         (np.log10(13), 4), (np.log10(7), 4)],
                  log=(True, False))
    assert m[1:, :16].all() and not m[:, 16:].any() and not m[0, 0]


def test_within_and_pearson():
    a, b = _two_phases()
    left = np.zeros(a.shape, bool)
    left[:, :16] = True
    H, *_ = AS.histogram(a, b, within=left)
    assert H.sum() == left.sum()
    assert AS.pearson(a, b) < -0.9                  # anticorrelated phases
    assert AS.pearson(a, a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        AS.histogram(a, b[:10])


def test_dialog_plots_and_a_polygon_emits_the_mask(qtbot):
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.association import AssociationDialog
    a, b = _two_phases()
    maps = {'A': a, 'B': b}
    d = AssociationDialog(['A', 'B'], lambda n: (maps[n], n, 'counts / pixel', 4))
    qtbot.addWidget(d)
    assert (d.cmb_x.currentText(), d.cmb_y.currentText()) == ('A', 'B')
    assert d.plot()
    assert 'Pearson r = -0.' in d.info.text()
    got = []
    d.maskSelected.connect(
        lambda m, name, op, b: got.append((m, name, op, b)))
    d.cmb_op.setCurrentIndex(d.cmb_op.findData('and'))
    m = d.select([(7, -1), (13, -1), (13, 4), (7, 4)])
    assert m[:, :16].all() and not m[:, 16:].any()
    assert got[0][1:] == ('assoc A-B', 'and', 4)
    d.cmb_x.setCurrentText('nothing')
    assert not d.plot() and 'Cannot plot' in d.info.text()


def _window(qtbot):
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    e = np.full(x.size, 7)
    s = w.session
    s.events = (x.ravel(), y.ravel(), e)
    s.full_spectrum = np.bincount(e, minlength=16).astype(float)
    s.spectrum = s.full_spectrum.copy()
    return w


@pytest.mark.parametrize('op, expect', [
    ('replace', 64 * 64), ('add', 2 * 64 * 64 - 32 * 32),
    ('and', 32 * 32), ('subtract', 64 * 64 - 32 * 32)])
def test_association_mask_ops_on_the_session(qtbot, monkeypatch, op, expect):
    """Include / refine / reject as GeoPIXE's (corr.pro:566-572), on the
    session mask, at the binning the plotted maps were made at."""
    pytest.importorskip('PyQt5.QtWidgets')
    w = _window(qtbot)
    try:
        said = []
        monkeypatch.setattr(w, 'say', lambda t: said.append(t))
        s = w.session
        cur = np.zeros((256, 256), bool)
        cur[:64, :64] = True                    # the current mask
        s.set_mask(cur, 'grain')
        w.spin_bin.setValue(3)                  # the Maps tab moved on
        m = np.zeros((64, 64), bool)
        m[8:24, 8:24] = True                    # raster 32-95 both ways
        w.on_assoc_mask(m, 'assoc A-B', op, 4)   # binning of the plot
        assert int(s.mask.sum()) == expect, said
        assert s.spectrum.sum() == expect
        assert ('assoc A-B' in s.mask_name)
    finally:
        w.close()


def test_maps_tab_opens_the_plot_on_maps_made_as_the_tab_makes_them(
        qtbot, monkeypatch):
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit import config
    if config.database_path(required=False) is None:
        pytest.skip('no GeoPIXE database available')
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        s = w.session
        rng = np.random.default_rng(0)
        e = rng.integers(305, 4096, 100000)
        s.cal = (0.0016984, -0.37225)
        s.events = (rng.integers(0, 256, e.size),
                    rng.integers(0, 256, e.size), e)
        s.full_spectrum = np.bincount(e, minlength=4096).astype(float)
        s.spectrum = s.full_spectrum.copy()
        w._window_lines = ['O', 'Si']
        w.cmb_mapel.clear()
        w.cmb_mapel.addItems(w._window_lines)
        w.spin_bin.setValue(8)
        w.cmb_bg.setCurrentIndex(w.cmb_bg.findData('none'))
        w.on_associations()
        d = w.assocview
        assert (d.cmb_x.currentText(), d.cmb_y.currentText()) == ('O', 'Si')
        assert d.plot()
        assert d.a.shape == (32, 32)
        assert 'raw counts' in d.ax.get_xlabel()
        assert 'counts / pixel' in d.ax.get_xlabel()
        # everything inside a huge polygon: the whole field
        d.cmb_op.setCurrentIndex(d.cmb_op.findData('replace'))
        d.select([(-1, -1), (1e6, -1), (1e6, 1e6), (-1, 1e6)])
        assert s.mask is not None and s.mask.all()
    finally:
        w.close()


@pytest.mark.parametrize('op, expect', [
    ('replace', 4096), ('add', 4096), ('and', 4096),
    ('subtract', 65536 - 4096)])
def test_association_ops_with_no_mask_set(qtbot, monkeypatch, op, expect):
    """No mask: add and refine make the region itself; subtract takes it
    from the whole field (the Maps tab's own ops do the same)."""
    pytest.importorskip('PyQt5.QtWidgets')
    w = _window(qtbot)
    try:
        monkeypatch.setattr(w, 'say', lambda t: None)
        m = np.zeros((64, 64), bool)
        m[8:24, 8:24] = True
        w.on_assoc_mask(m, 'assoc A-B', op, 4)
        assert int(w.session.mask.sum()) == expect
    finally:
        w.close()


def test_a_hot_border_pixel_is_on_the_plot():
    a, b = _two_phases()
    a[0, 0] = 1e6                               # in the 2 % border
    H, xe, ye, lims = AS.histogram(a, b)
    assert H.sum() == a.size
    assert lims[0][1] < 20                      # the range ignores it
