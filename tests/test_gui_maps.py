"""GUI map view: MapCanvas behaviour under repeated use.

Synthetic data only (no database, no data files); offscreen Qt (conftest sets
QT_QPA_PLATFORM).
"""
import gc
import os
import weakref

import numpy as np
import pytest

from scrappyfit import config

pytest.importorskip('PyQt5.QtWidgets')


@pytest.fixture
def canvas(qtbot):
    """A MapCanvas inside a QApplication, closed at teardown."""
    from scrappyfit.gui.canvases import MapCanvas
    c = MapCanvas()
    qtbot.addWidget(c)
    return c


@pytest.fixture
def window(qtbot):
    """A MainWindow, closed at teardown so its worker thread stops. Left to
    the garbage collector, a window whose QThread still runs takes the whole
    test process down."""
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def _fake_listmode(session, n=5000, seed=1):
    """Synthetic list-mode events on the 256x256 raster, enough for masks."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, n)
    y = rng.integers(0, 256, n)
    e = rng.integers(0, 1024, n)
    session.events = (x, y, e)
    session.full_spectrum = np.bincount(e, minlength=1024).astype(float)
    session.spectrum = session.full_spectrum.copy()


@pytest.fixture
def map_data():
    """A seeded synthetic 64x64 elemental map."""
    return np.random.default_rng(0).gamma(2.0, 10.0, size=(64, 64))


def test_repeated_show_map_keeps_axes_position(canvas, map_data):
    """Calling show_map repeatedly must not move or shrink the map axes, nor
    leave extra colorbar axes behind."""
    canvas.show_map(map_data)
    canvas.draw()
    first = canvas.ax.get_position().bounds
    for i in range(4):
        canvas.show_map(map_data)
        canvas.draw()
        now = canvas.ax.get_position().bounds
        assert np.allclose(now, first), \
            'call %d: axes bounds %s != first %s' % (i + 2, now, first)
    assert len(canvas.fig.axes) == 2


def test_repeated_show_map_frees_old_maps(canvas, map_data):
    """Each Show map must let the previous image (and its array) go: a
    colour bar that wraps the last one's axes locator keeps them all alive."""
    canvas.show_map(map_data)
    first = weakref.ref(canvas._img)
    for _ in range(3):
        canvas.show_map(map_data)
        canvas.draw()
    gc.collect()
    assert first() is None


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_grow_shrink_act_on_the_current_rectangle_mask(window, map_data):
    """Grow / shrink must change the session mask by one displayed pixel,
    whichever tool (here a rectangle) made it, not only after a flood fill."""
    win = window
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.integers(0, 256, n)
    y = rng.integers(0, 256, n)
    e = rng.integers(0, 1024, n)
    s = win.session
    s.events = (x, y, e)
    s.full_spectrum = np.bincount(e, minlength=1024).astype(float)
    s.spectrum = s.full_spectrum.copy()
    win.map.show_map(map_data)

    def shown():
        """Session mask binned back onto the displayed 64x64 grid."""
        m = s.mask
        assert m is not None and m.shape == (256, 256)
        return int(m.reshape(64, 4, 64, 4).any(axis=(1, 3)).sum())

    rect = np.zeros((64, 64), bool)
    rect[20:30, 20:30] = True
    win.on_region(rect)
    assert shown() == 100
    win.on_morph(1)
    assert shown() == 144
    win.on_morph(1)
    assert shown() == 196
    win.on_morph(-1)
    win.on_morph(-1)
    assert shown() == 100


def test_masking_grow_shrink_square_kernel():
    """connectivity=8 grows a square by a square ring and shrinks it back;
    the default 4 rounds the corners; anything else is refused."""
    from scrappyfit.analysis import masking as MSK
    b = np.zeros((20, 20), bool)
    b[8:12, 8:12] = True
    assert MSK.grow(b, 2, connectivity=8).sum() == 64
    assert (MSK.shrink(MSK.grow(b, 2, connectivity=8), 2,
                       connectivity=8) == b).all()
    assert MSK.grow(b, 2).sum() == 64 - 4 * 3
    with pytest.raises(ValueError):
        MSK.grow(b, 1, connectivity=6)


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_reset_view_button_restores_extent_and_contrast(qtbot, window,
                                                         map_data):
    """After zoom/pan and a contrast change, a Reset view button must restore
    the full extent and the default colour limits; a map toolbar must exist."""
    from PyQt5 import QtCore
    win = window
    win.map.show_map(map_data)
    default_clim = win.map._img.get_clim()
    win.map.ax.set_xlim(10, 20)
    win.map.ax.set_ylim(30, 40)
    win.map._img.set_clim(0, 1)
    qtbot.mouseClick(win.btn_map_reset, QtCore.Qt.LeftButton)
    assert np.allclose(win.map.ax.get_xlim(), (-0.5, 63.5))
    assert np.allclose(win.map.ax.get_ylim(), (-0.5, 63.5))
    assert np.allclose(win.map._img.get_clim(), default_clim)
    assert hasattr(win, 'map_toolbar')


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_flood_click_ignored_while_zooming(window, map_data):
    """A click on the map while the toolbar zooms must not seed a flood fill."""
    from types import SimpleNamespace
    win = window
    win.map.show_map(map_data)
    win.cmb_mode.setCurrentText('flood fill')
    win.map_toolbar.zoom()
    seen = []
    win._apply_mask = lambda m, name: seen.append(name)
    win.on_map_click(SimpleNamespace(inaxes=win.map.ax, xdata=30.0, ydata=30.0))
    assert seen == []


def test_display_scale_cmap_and_percentiles(canvas, map_data):
    """Colour table, log / sqrt scale and percentile limits apply to the
    shown map at once and survive the next Show map."""
    from matplotlib import colors as mcolors
    canvas.show_map(map_data)
    canvas.set_display(cmap='viridis', scale='log', pct=(5, 95))
    canvas.draw()
    assert canvas._img.get_cmap().name == 'viridis'
    assert isinstance(canvas._img.norm, mcolors.LogNorm)
    assert np.allclose(canvas._img.get_clim(),
                       np.percentile(map_data, [5, 95]))
    canvas.show_map(map_data * 2)              # the next element
    assert isinstance(canvas._img.norm, mcolors.LogNorm)
    assert canvas._img.get_cmap().name == 'viridis'
    assert np.allclose(canvas._img.get_clim(),
                       np.percentile(map_data * 2, [5, 95]))
    canvas.set_display(scale='sqrt')
    canvas.draw()
    assert isinstance(canvas._img.norm, mcolors.PowerNorm)
    canvas.reset_view()                        # default contrast, same scale
    canvas.draw()
    assert canvas.pct == (2.0, 99.0)
    assert isinstance(canvas._img.norm, mcolors.PowerNorm)
    assert np.allclose(canvas._img.get_clim(),
                       np.percentile(map_data * 2, [2, 99]))
    with pytest.raises(ValueError):
        canvas.set_display(pct=(50, 50))


def test_log_scale_on_a_net_map_with_negatives(canvas):
    """A flanking-subtracted map goes negative; log must still draw."""
    d = np.random.default_rng(3).normal(0.0, 5.0, (64, 64))
    canvas.show_map(d)
    canvas.set_display(scale='log')
    lo, hi = canvas._img.get_clim()
    assert 0 < lo < hi
    canvas.draw()


@pytest.mark.parametrize('kind', ['sparse', 'all_negative', 'high_below_zero'])
def test_log_scale_edge_maps_draw(canvas, kind):
    """Log must draw on a sparse count map (99th percentile ~0), an
    all-negative net map, and with the high percentile below zero."""
    rng = np.random.default_rng(4)
    pct = (2, 99)
    if kind == 'sparse':
        d = rng.poisson(0.015, (64, 64)).astype(float)
    elif kind == 'all_negative':
        d = -rng.gamma(2.0, 3.0, (64, 64))
    else:
        d = rng.normal(0.0, 5.0, (64, 64))
        pct = (2, 30)
    canvas.show_map(d)
    canvas.set_display(scale='log', pct=pct)
    canvas.draw()
    lo, hi = canvas._img.get_clim()
    assert lo < hi


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_display_controls_drive_the_map(window, map_data):
    from matplotlib import colors as mcolors
    win = window
    win.map.show_map(map_data)
    win.cmb_scale.setCurrentText('sqrt')
    win.cmb_cmap.setCurrentText('gray')
    win.spin_phi.setValue(90.0)
    assert isinstance(win.map._img.norm, mcolors.PowerNorm)
    assert win.map._img.get_cmap().name == 'gray'
    assert np.isclose(win.map._img.get_clim()[1],
                      np.percentile(map_data, 90))
    win.btn_map_reset.click()
    assert win.spin_phi.value() == 99.0
    assert np.isclose(win.map._img.get_clim()[1],
                      np.percentile(map_data, 99))


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_prev_next_step_through_the_element_list_and_wrap(window):
    win = window
    shown = []
    win.on_map = lambda: shown.append(win.cmb_mapel.currentText()) or True
    win.cmb_mapel.clear()
    win.cmb_mapel.addItems(['O', 'Si', 'Ca'])
    win.cmb_mapel.setCurrentIndex(0)
    win.btn_map_next.click()
    win.btn_map_next.click()
    win.btn_map_next.click()                   # wraps to the first
    win.btn_map_prev.click()                   # and back to the last
    assert shown == ['Si', 'Ca', 'O', 'Ca']


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_export_map_csv_with_mask(window, map_data, tmp_path):
    """The shown map round-trips through CSV (row 0 = bottom), and the mask
    is written beside it as a per-pixel fraction on the map's grid."""
    from scrappyfit.gui.app import export_map_csv
    win = window
    _fake_listmode(win.session)
    win.map.show_map(map_data)
    rect = np.zeros((64, 64), bool)
    rect[20:30, 10:15] = True
    win.on_region(rect)
    meta = dict(element='Si', binning=4, kind='net', label='run A',
                cal=(0.002, -0.4), stem='runA')
    out = export_map_csv(str(tmp_path / 'm.csv'), map_data, win.session, meta)
    assert [os.path.basename(p) for p in out] == ['m.csv', 'm_mask.csv']
    win.session.label, win.session.cal = 'run B', (0.003, 0.0)
    out = export_map_csv(str(tmp_path / 'm.csv'), map_data, win.session, meta)
    head = open(out[0], encoding='utf-8').readline()
    assert 'run A' in head and '0.002 keV/ch' in head   # what made the map
    back = np.loadtxt(out[0], delimiter=',')
    assert back.shape == (64, 64)
    assert np.allclose(back, map_data, rtol=1e-5)
    frac = np.loadtxt(out[1], delimiter=',')
    assert (frac == rect.astype(float)).all()
    win.on_clear_mask()
    assert export_map_csv(str(tmp_path / 'n.csv'), map_data, win.session,
                          meta) == [str(tmp_path / 'n.csv')]


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_step_keeps_the_list_on_what_is_shown_when_mapping_fails(window):
    """No list-mode data: the map fails, so the element list must not move
    off the element the canvas still shows. Case-insensitive match."""
    win = window
    win.cmb_mapel.clear()
    win.cmb_mapel.addItems(['O', 'Si', 'Ca'])
    win.cmb_mapel.setCurrentText('si')
    win.btn_map_next.click()                   # session has no events
    assert win.cmb_mapel.currentText() == 'si'
    shown = []
    win.on_map = lambda: shown.append(win.cmb_mapel.currentText()) or True
    win.btn_map_next.click()
    assert shown == ['Ca']                     # stepped from Si, not O


def test_polygon_selects_the_pixels_whose_centres_it_contains(canvas, map_data,
                                                             qtbot):
    """Pixel i is centred on i: a square from 9.5 to 19.5 holds pixels
    10..19, 100 of them, nothing outside."""
    canvas.show_map(map_data)
    with qtbot.waitSignal(canvas.polygonSelected, timeout=1000) as sig:
        canvas._on_poly([(9.5, 9.5), (19.5, 9.5), (19.5, 19.5), (9.5, 19.5)])
    m = sig.args[0]
    want = np.zeros((64, 64), bool)
    want[10:20, 10:20] = True
    assert (m == want).all()


def test_rectangle_rounds_to_the_pixel_under_the_cursor(canvas, map_data,
                                                        qtbot):
    """A drag from 9.7 starts in pixel 10 (9.5..10.5), not pixel 9."""
    from types import SimpleNamespace as NS
    canvas.show_map(map_data)
    with qtbot.waitSignal(canvas.regionSelected, timeout=1000) as sig:
        canvas._on_rect(NS(xdata=9.7, ydata=9.7), NS(xdata=12.4, ydata=12.4))
    m = sig.args[0]
    assert m.sum() == 9 and m[10:13, 10:13].all()


def test_tools_survive_show_map(canvas, map_data):
    """Each Show map rebuilds the tools on the cleared axes and keeps the
    chosen one active."""
    canvas.show_map(map_data)
    canvas.set_tool('polygon')
    canvas.show_map(map_data)
    assert canvas._poly.active and not canvas._sel.active
    assert canvas._poly.ax is canvas.ax
    canvas.set_tool('flood fill')
    assert not canvas._poly.active and not canvas._sel.active


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_add_and_subtract_combine_with_the_current_mask(window, map_data):
    """GeoPIXE's include / exclude: a shape with 'add' joins the mask, with
    'subtract' cuts it out; grow ignores the op; an empty result is refused."""
    win = window
    _fake_listmode(win.session)
    win.map.show_map(map_data)

    def shown():
        return win.session.mask.reshape(64, 4, 64, 4).any(axis=(1, 3))

    a = np.zeros((64, 64), bool)
    a[10:20, 10:20] = True
    b = np.zeros((64, 64), bool)
    b[30:35, 30:35] = True
    win.on_region(a)
    win.cmb_op.setCurrentText('add')
    win.on_polygon(b)
    assert (shown() == (a | b)).all()
    assert win.session.mask_name == 'region+polygon'
    win.cmb_op.setCurrentText('subtract')
    win.on_region(a)
    assert (shown() == b).all()
    win.on_morph(1)                            # grow is not a subtract
    assert shown().sum() == 7 * 7
    before = win.session.mask.copy()
    full = np.ones((64, 64), bool)
    win.on_region(full)                        # would empty it: refused
    assert (win.session.mask == before).all()


@pytest.mark.skipif(config.database_path(required=False) is None,
                    reason='no GeoPIXE database available')
def test_subtract_without_a_mask_cuts_from_the_whole_field(window, map_data):
    """Never the excluded shape itself; and a repeat of the same shape is a
    no-op that leaves the name alone."""
    win = window
    _fake_listmode(win.session)
    win.map.show_map(map_data)
    crack = np.zeros((64, 64), bool)
    crack[5:10, :] = True
    win.cmb_op.setCurrentText('subtract')
    win.on_region(crack)
    shown = win.session.mask.reshape(64, 4, 64, 4).any(axis=(1, 3))
    assert (shown == ~crack).all()
    name = win.session.mask_name
    win.on_region(crack)
    assert win.session.mask_name == name


def test_switching_tool_drops_a_half_drawn_polygon(canvas, map_data):
    canvas.show_map(map_data)
    canvas.set_tool('polygon')
    canvas._poly._xys[:0] = [(1.0, 1.0), (5.0, 1.0)]
    canvas.set_tool('rectangle')
    canvas.set_tool('polygon')
    assert len(canvas._poly.verts) == 0


@pytest.mark.parametrize('size', [(16, 5), (5, 5), (4, 5), (3, 6)])
def test_colour_bar_sits_beside_the_map(canvas, map_data, size):
    """The map keeps its aspect; the bar must follow its right edge and
    height, not a fixed fraction of a wide figure, and never cover it in a
    narrow one."""
    canvas.fig.set_size_inches(*size)
    canvas._reserve_bar_room()
    canvas.show_map(map_data)
    canvas.draw()
    m = canvas.ax.get_position()
    c = canvas.cax.get_position()
    w = canvas.fig.get_size_inches()[0] * canvas.fig.dpi
    assert 9 < (c.x0 - m.x1) * w < 11            # 10 px gap, never on it
    assert c.x1 * w <= w - 60                    # room for tick labels
    assert abs(c.y0 - m.y0) < 1e-6 and abs(c.height - m.height) < 1e-6


def test_colour_bar_follows_display_changes_after_later_maps(canvas, map_data):
    """The colour bar must track the shown map's scale, limits and colour map
    on every map, not only the first. One colour bar serves every Show map,
    so each new image has to be wired to it, or the bar can show a linear
    scale over a log map."""
    canvas.show_map(map_data)
    canvas.show_map(map_data * 2.0)
    canvas.set_display(scale='log', cmap='viridis')
    assert canvas._cbar.norm is canvas._img.norm
    assert canvas._cbar.cmap.name == 'viridis'
    canvas.reset_view()
    assert canvas._cbar.norm is canvas._img.norm
    assert canvas._cbar.norm.vmin == pytest.approx(canvas._img.norm.vmin)
    assert canvas._cbar.norm.vmax == pytest.approx(canvas._img.norm.vmax)
