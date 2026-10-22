"""Periodic-table line picker for maps (BUGS.md: "Periodic-table line
picker for maps")."""
import numpy as np
import pytest

from scrappyfit import config

pytest.importorskip('PyQt5.QtWidgets')
pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')


def _db():
    from scrappyfit.session import Session
    return Session().db


def test_split_line():
    from scrappyfit.gui.maplines import split_line
    assert split_line('Si') == ('Si', 1)
    assert split_line('Ag L') == ('Ag', 2)
    assert split_line('Pb M') == ('Pb', 3)
    assert split_line('ag l') == ('Ag', 2)
    for bad in ('1.74', '1.74 keV', 'Ag La', 'Back1', 'VL', ''):
        assert split_line(bad) is None, bad


def test_table_offers_only_the_listed_lines(qtbot):
    from scrappyfit.gui.maplines import MapLinePicker
    d = MapLinePicker(_db(), ['Si', 'Ag L', 'Pb M', 'Fe', 'Fe L'], 'Si')
    qtbot.addWidget(d)
    b = d.buttons
    assert b['Si'].available == (1,) and b['Si'].isEnabled()
    assert b['Ag'].available == (2,)
    assert b['Fe'].available == (1, 2)
    assert not b['Ca'].isEnabled()            # not in the list
    assert b['Si'].shells == (1,)             # the line now shown is marked
    assert b['Fe'].text() == 'Fe\nKL'
    assert 'Ag L' in b['Ag'].toolTip()


@pytest.mark.parametrize('current', ['ag l', '1.74 keV', 'Ag La', 'fe l'])
def test_any_typed_current_text_opens_the_table(qtbot, current):
    from scrappyfit.gui.maplines import MapLinePicker
    d = MapLinePicker(_db(), ['Si', 'Ag L', 'Fe', 'Fe L'], current)
    qtbot.addWidget(d)
    marked = {s: b.shells for s, b in d.buttons.items() if b.shells}
    assert marked == {'ag l': {'Ag': (2,)}, 'fe l': {'Fe': (2,)}}.get(
        current, {})


def test_names_off_the_table_get_their_own_buttons(qtbot):
    """A DA matrix row that is not an element line is still offered."""
    from scrappyfit.gui.maplines import MapLinePicker
    d = MapLinePicker(_db(), ['Si', 'V L', 'Back1', 'Np'])
    qtbot.addWidget(d)
    assert set(d.other_buttons) == {'Back1', 'Np'}
    got = []
    d.linePicked.connect(got.append)
    d.other_buttons['Back1'].click()
    assert got == ['Back1']


def test_one_shell_maps_at_once_two_ask(qtbot):
    from scrappyfit.gui.maplines import MapLinePicker
    d = MapLinePicker(_db(), ['Si', 'Ag L', 'Fe', 'Fe L'])
    qtbot.addWidget(d)
    got = []
    d.linePicked.connect(got.append)
    d.buttons['Ag'].click()
    assert got == ['Ag L']
    d2 = MapLinePicker(_db(), ['Si', 'Ag L', 'Fe', 'Fe L'])
    qtbot.addWidget(d2)
    d2.linePicked.connect(got.append)
    d2.buttons['Fe'].click()
    acts = [a.text() for a in d2._menu.actions()]
    assert acts == ['Fe  (K)', 'Fe L  (L)']
    d2._menu.actions()[1].trigger()
    assert got == ['Ag L', 'Fe L']
    with pytest.raises(ValueError):
        d2.pick('Ca')


def test_maps_tab_table_button_maps_the_picked_line(qtbot, monkeypatch):
    from scrappyfit.gui.app import MainWindow
    from scrappyfit.gui import maplines
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
        w._window_lines = ['O', 'Si', 'Ag L']
        w.cmb_mapel.clear()
        w.cmb_mapel.addItems(w._window_lines)
        shown = []
        monkeypatch.setattr(w, 'on_map', lambda: shown.append(
            w.cmb_mapel.currentText()) or True)
        # the dialog is modal in use; here pick straight away
        monkeypatch.setattr(maplines.MapLinePicker, 'exec_',
                            lambda self: self.pick('Ag L'))
        w.on_map_pick()
        assert w.cmb_mapel.currentText() == 'Ag L'
        assert shown == ['Ag L']
    finally:
        w.close()
