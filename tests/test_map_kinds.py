"""Map kinds and backgrounds on the Maps tab (BUGS.md: "Map background
options and imaging tools like GeoPIXE"): GeoPIXE's three window background
types, DA maps beside window maps, and every map saying which it is."""
import numpy as np
import pytest

from scrappyfit import config
from scrappyfit.session import Session

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

CAL = (0.0016984, -0.37225)
NCH = 4096


def _flat(s, ch_min=305, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.integers(ch_min, NCH, n)
    s.cal = CAL
    s.events = (rng.integers(0, 256, n), rng.integers(0, 256, n), e)
    s.full_spectrum = np.bincount(e, minlength=NCH).astype(float)
    s.spectrum = s.full_spectrum.copy()
    return s


def _dam(names=('Back', 'Si', 'VL', 'sum')):
    """A tiny DA matrix: every row weight 1 on every channel."""
    return {'el': list(names), 'size': NCH, 'charge': 1.0,
            'matrix': [np.ones(NCH) for _ in names], 'file': 'test.dam'}


def test_window_backgrounds_as_geopixe_cut_types():
    s = _flat(Session())
    s.element_map('Si', background='linear')
    assert s.map_info['background'] == 'linear between flanks'
    lo, hi = s.map_info['window']
    s.element_map('Si', background='low')
    assert s.map_info['background'] == 'flat, low flank only'
    assert s.map_info['flanks'][0][1] < lo
    s.element_map('Si', background='high')
    assert s.map_info['background'] == 'flat, high flank only'
    assert s.map_info['flanks'][0][0] > hi
    raw = s.element_map('Si', background='none')
    assert s.map_info['background'] == 'none'
    assert s.map_info['units'] == 'counts / pixel'
    assert raw.min() >= 0
    # subtract=False still means raw, as before
    s.element_map('Si', subtract=False)
    assert s.map_info['background'] == 'none'
    with pytest.raises(ValueError):
        s.element_map('Si', background='quadratic')


def test_a_one_flank_background_off_the_spectrum_is_refused():
    s = _flat(Session())
    with pytest.raises(ValueError, match='low flank'):
        s.element_map('C', background='low')     # below the first count
    s.element_map('C', background='high')
    assert s.map_info['background'] == 'flat, high flank only'


def test_da_map_names_and_one_da_map():
    s = _flat(Session())
    s.dam = _dam()
    assert s.da_map_names() == [('Si', 'Si'), ('V L', 'VL')]
    m = s.da_map('V L', binning=4)
    assert m.shape == (64, 64)
    info = s.map_info
    assert (info['kind'], info['units'], info['dam_name'],
            info['matrix']) == ('DA', 'wt%', 'VL', 'test.dam')
    assert s.da_map('VL', binning=4).shape == (64, 64)    # GeoPIXE's name
    with pytest.raises(ValueError, match='not in the DA matrix'):
        s.da_map('Ca')


def test_describe_map_says_window_or_da():
    from scrappyfit.gui.canvases import describe_map, map_kind_short
    s = _flat(Session())
    s.element_map('Si')
    assert map_kind_short(s.map_info) == 'WINDOW, net counts'
    t = describe_map(s.map_info)
    assert t.startswith('WINDOW map of Si') and 'linear between flanks' in t
    s.element_map('Si', background='none')
    assert map_kind_short(s.map_info) == 'WINDOW, raw counts'
    assert 'raw counts' in describe_map(s.map_info)
    s.dam = _dam()
    s.da_map('Si')
    assert map_kind_short(s.map_info) == 'DA, wt%'
    assert describe_map(s.map_info).startswith('DA map of Si')


@pytest.fixture
def window(qtbot):
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def test_maps_tab_states_the_kind_and_offers_da_maps(window, monkeypatch):
    w = window
    s = _flat(w.session)
    said = []
    monkeypatch.setattr(w, 'say', lambda t: said.append(t))
    w.cmb_mapel.setCurrentText('Si')
    w.cmb_bg.setCurrentIndex(w.cmb_bg.findData('none'))
    assert w.on_map()
    assert 'WINDOW, raw counts' in w.map.ax.get_title()
    assert w.lbl_mapkind.text().startswith('WINDOW map of Si')
    assert w.map._cbar.ax.get_ylabel() == 'counts / pixel'
    # no DA matrix: DA is not on offer
    w.cmb_kind.setCurrentIndex(1)
    assert w.cmb_kind.currentIndex() == 0
    # a DA matrix: the tab lists its maps and draws them as DA, wt%
    s.dam = _dam()
    w._sync_map_kind()
    w.cmb_kind.setCurrentIndex(1)
    items = [w.cmb_mapel.itemText(i) for i in range(w.cmb_mapel.count())]
    assert items == ['Si', 'V L']
    assert not w.cmb_bg.isEnabled()
    w.cmb_mapel.setCurrentText('V L')
    assert w.on_map()
    assert 'DA, wt%' in w.map.ax.get_title()
    assert w.map._cbar.ax.get_ylabel() == 'wt%'
    assert 'test.dam' in w.lbl_mapkind.text()
    assert w._map_meta['kind'] == 'DA, wt%'
    # a second matrix while on DA maps: its own elements are listed
    s.dam = _dam(('Back', 'Al', 'Fe'))
    w._sync_map_kind(prefer_da=True)
    items = [w.cmb_mapel.itemText(i) for i in range(w.cmb_mapel.count())]
    assert items == ['Al', 'Fe']
    # the matrix goes (another file opened): back to window maps
    s.dam = None
    w.cmb_mapel.setCurrentText('Si')
    assert w.on_map()
    assert w.cmb_kind.currentIndex() == 0 and w.cmb_bg.isEnabled()
    assert not any('FAILED' in t for t in said), said
