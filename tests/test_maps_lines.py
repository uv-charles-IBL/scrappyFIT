"""Maps tab: the element box and Session.element_map must map lines that are
in the fit range, and a degenerate threshold must not become a mask."""
import numpy as np
import pytest

from scrappyfit import config

pytest.importorskip('PyQt5.QtWidgets')

pytestmark = pytest.mark.skipif(config.database_path(required=False) is None,
                                reason='no GeoPIXE database available')

CAL = (0.0016984, -0.37225)        # 287424: 4096 ch, 0.2-6.5 keV


@pytest.fixture
def window(qtbot):
    """A MainWindow, closed at teardown so its worker thread stops."""
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def _events(session, n=20000, seed=1):
    """Synthetic list-mode events over 4096 channels at the 287424 calibration."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, n)
    y = rng.integers(0, 256, n)
    e = rng.integers(0, 4096, n)
    session.events = (x, y, e)
    session.full_spectrum = np.bincount(e, minlength=4096).astype(float)
    session.spectrum = session.full_spectrum.copy()
    session.cal = CAL



def test_element_map_accepts_a_line_spec(window):
    """'Ag L' / 'AgL' / 'Ag:L' / 'Pb M' map the L / M line."""
    s = window.session
    _events(s)
    x, y, e = s.events
    ch = lambda E: int(round((E - CAL[1]) / CAL[0]))
    # 400 events at Ag La (2.98 keV) in pixel (40, 40); at Pb Ma (2.35 keV) in (200, 200)
    ex = np.r_[x, np.full(400, 40), np.full(400, 200)]
    ey = np.r_[y, np.full(400, 40), np.full(400, 200)]
    ee = np.r_[e, np.full(400, ch(2.984)), np.full(400, ch(2.346))]
    s.events = (ex, ey, ee)
    for spec in ('Ag L', 'AgL', 'Ag:L'):
        m = s.element_map(spec)
        assert np.unravel_index(np.argmax(m), m.shape) == (10, 10), spec
    m = s.element_map('Pb M')
    assert np.unravel_index(np.argmax(m), m.shape) == (50, 50)



def test_element_map_refuses_a_line_outside_the_spectrum(window):
    """Bare 'Ag' (K at 22 keV) on a 0.2-6.5 keV spectrum raises, not a zero map."""
    s = window.session
    _events(s)
    with pytest.raises(ValueError):
        s.element_map('Ag')



def test_every_maps_box_entry_maps_without_error(window):
    """Each entry of the Maps element box is a line inside the fit range."""
    w = window
    s = w.session
    _events(s)
    s.options.e_low, s.options.e_high = 0.2, 6.5
    w.rebuild_element_lists()
    items = [w.cmb_mapel.itemText(i) for i in range(w.cmb_mapel.count())]
    assert items
    assert 'Si' in items and 'Ag L' in items and 'Pb M' in items
    for it in items:
        s.element_map(it)



def test_threshold_refuses_a_constant_map(window):
    """'Mask above percentile' on a constant map must not select every pixel."""
    w = window
    s = w.session
    _events(s)
    w.map.show_map(np.zeros((64, 64)))
    w.on_threshold()
    assert s.mask is None


def test_parse_line_forms(window):
    """Separated shells in any case; run-together symbols stay elements."""
    p = window.session.parse_line
    assert p('Ag L') == p('AgL') == p('Ag:L') == p('ag l') == (47, 2)
    assert p('Pb M') == p('PbM') == (82, 3)
    assert p('Si') == p('Si K') == p('si') == (14, 1)
    assert p('Cl') == p('CL') == (17, 1)        # chlorine, as before
    assert p('Al') == (13, 1) and p('Tl') == (81, 1)


def test_stepping_the_maps_box_leaves_the_session_alone(window):
    """Map browsing must not change the spectrum, the mask or the fit start."""
    w = window
    s = w.session
    _events(s)
    s.options.e_low, s.options.e_high = 0.2, 6.5
    w.rebuild_element_lists()
    spec, start = s.spectrum.copy(), s.start_pars
    for _ in range(w.cmb_mapel.count() + 2):
        w.on_map_step(1)
    assert np.array_equal(s.spectrum, spec)
    assert s.mask is None and s.mask_name == ''
    assert s.start_pars is start
