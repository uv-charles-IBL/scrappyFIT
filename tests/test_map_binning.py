"""Maps and masks at a binning that does not divide 256 (binning 3: 256 // 3 = 85)."""
import numpy as np
import pytest

from scrappyfit.analysis import maps
from scrappyfit.session import Session

BIN = 3
NPIX = -(-256 // BIN)          # ceil(256 / 3) = 86: every event has a pixel


def _events():
    """One event per raster pixel, all in channel 10: x, y cover 0..255."""
    y, x = np.divmod(np.arange(256 * 256), 256)
    return x, y, np.full(x.size, 10)



def test_map_at_binning_3_does_not_wrap_the_last_column():
    """x = 255 // 3 = 85 must get its own column, not spill into column 0 of the next row."""
    x = np.array([255, 255, 0])
    y = np.array([0, 0, 0])
    e = np.full(3, 10)
    m = maps.element_map(x, y, e, 5, 15, binning=BIN)
    assert m.shape == (NPIX, NPIX)
    assert m[0, 85] == 2 and m[1, 0] == 0, (m[0, 85], m[1, 0])



def test_map_at_binning_3_counts_every_event():
    """Total counts of a binning-3 map must equal the number of events selected."""
    x, y, e = _events()
    m = maps.element_map(x, y, e, 5, 15, binning=BIN)
    assert m.shape == (NPIX, NPIX)
    assert m.sum() == x.size



def test_set_mask_from_a_binning_3_map_selects_its_region():
    """A mask the shape of a binning-3 map must not raise IndexError and must select its events."""
    s = Session()
    x, y, e = _events()
    s.events = (x, y, e)
    s.full_spectrum = np.bincount(e, minlength=1024).astype(float)
    s.spectrum = s.full_spectrum.copy()
    m = np.zeros((NPIX, NPIX), bool)
    m[10:20, 30:40] = True                       # map pixels, rows 10-19, columns 30-39
    n = s.set_mask(m, 'roi')
    assert n == 10 * BIN * 10 * BIN, n           # each map pixel is 3x3 raster pixels


def test_bin_factor_and_to_raster_round_trip():
    from scrappyfit.analysis import masking as MSK
    assert [MSK.bin_factor(n) for n in (256, 128, 86, 64, 52)] == \
        [1, 2, 3, 4, 5]
    assert MSK.bin_factor(8, binning=32) == 32      # 32-36 all make 8 px
    with pytest.raises(ValueError):
        MSK.bin_factor(85)                 # what nbin // 3 used to give
    m = np.zeros((86, 86), bool)
    m[85, 85] = True                       # the partial corner pixel
    r = MSK.to_raster(m)
    assert r.shape == (256, 256)
    assert r.sum() == 1 and r[255, 255]
    f = MSK.map_fraction(r, 86)
    assert f.shape == (86, 86) and f[85, 85] == 1.0 and f.sum() == 1.0


def test_da_maps_at_binning_3_keep_a_flat_concentration_flat():
    """Each DA pixel is divided by the charge of its own raster pixels, so
    the partial edge pixels of an 86-px map read like the rest."""
    from scrappyfit.analysis.damaps import da_maps
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    x, y = x.ravel(), y.ravel()
    e = np.full(x.size, 5)
    dam = {'size': 10, 'el': ['Si'], 'charge': 2.0,
           'matrix': [np.ones(10)]}
    m3 = da_maps(dam, x, y, e, binning=3)['Si']
    m1 = da_maps(dam, x, y, e, binning=1)['Si']
    assert m3.shape == (86, 86)
    assert np.allclose(m3, m1[0, 0])
    m4 = da_maps(dam, x, y, e, binning=4)['Si']
    assert np.allclose(m4, m1[0, 0])


def test_mask_overlay_spans_the_raster_on_a_binning_3_map(qtbot):
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.canvases import MapCanvas
    c = MapCanvas()
    qtbot.addWidget(c)
    c.show_map(np.ones((86, 86)), binning=3)
    assert c.binning() == 3
    c.set_mask(np.ones((256, 256), bool))
    x0, x1, y0, y1 = c._mask_art[0].get_extent()
    assert (x0, x1) == pytest.approx((-0.5, 256 / 3 - 0.5))


def test_gui_rectangle_and_add_at_binning_3(qtbot, monkeypatch):
    """The Maps tab path: a rectangle on an 86-px map becomes a mask, and
    'add' combines a second one with it, instead of MASK FAILED."""
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        s = w.session
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        e = np.full(x.size, 7)
        s.events = (x.ravel(), y.ravel(), e)
        s.full_spectrum = np.bincount(e, minlength=16).astype(float)
        s.spectrum = s.full_spectrum.copy()
        said = []
        monkeypatch.setattr(w, 'say', lambda t: said.append(t))
        w.map.show_map(np.ones((86, 86)), binning=3)
        a = np.zeros((86, 86), bool)
        a[80:, 80:] = True                 # reaches the partial last pixel
        w.on_region(a)
        assert s.mask is not None and s.mask.shape == (256, 256)
        assert s.spectrum.sum() == 16 * 16    # raster 240..255 both ways
        w.cmb_op.setCurrentText('add')
        b = np.zeros((86, 86), bool)
        b[:2, :2] = True
        w.on_region(b)
        assert s.spectrum.sum() == 16 * 16 + 6 * 6
        assert not any('FAILED' in t for t in said), said
    finally:
        w.close()


def test_binning_17_is_not_taken_for_16():
    """16 and 17 both make a 16-px map. Inferring the factor from the size
    put a binning-17 mask up to 15 raster pixels off with no error."""
    from scrappyfit.analysis import masking as MSK
    with pytest.raises(ValueError, match='16 or 17'):
        MSK.bin_factor(16)
    assert MSK.bin_factor(16, binning=17) == 17
    with pytest.raises(ValueError):
        MSK.bin_factor(16, binning=3)
    m = np.zeros((16, 16), bool)
    m[15, 15] = True
    r = MSK.to_raster(m, binning=17)
    ys, xs = np.nonzero(r)
    assert ys.min() == xs.min() == 15 * 17 and r.sum() == 1   # 255 only
    s = Session()
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    e = np.zeros(x.size, int)
    s.events = (x.ravel(), y.ravel(), e)
    s.full_spectrum = np.bincount(e, minlength=4).astype(float)
    s.spectrum = s.full_spectrum.copy()
    m = np.zeros((16, 16), bool)
    m[0, 0] = True
    assert s.set_mask(m, 'b17', binning=17) == 17 * 17
    with pytest.raises(ValueError):
        s.set_mask(m, 'ambiguous')
