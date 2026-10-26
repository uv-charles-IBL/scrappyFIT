"""Element table: M shells below Cd (Z < 48) must be selectable.

Offscreen Qt (conftest sets QT_QPA_PLATFORM). Synthetic/vendored data only.
"""
import pytest

pytest.importorskip('PyQt5.QtWidgets')

from scrappyfit import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

M = 3  # shell index of M in the table


@pytest.fixture
def window(qtbot):
    """A MainWindow, closed at teardown so its worker thread stops."""
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def _window_range(win, lo, hi):
    win.session.options.e_low = lo
    win.session.options.e_high = hi
    win.rebuild_element_lists()
    return win.elements


@pytest.mark.parametrize('sym', ['Ag', 'Pd'])
def test_m_shell_is_available_for_ag_pd_in_the_default_window(window, sym):
    """Ag/Pd M lines (0.3-0.6 keV) lie in 0.2-6.5 keV, so shell M must be offered."""
    dlg = _window_range(window, 0.2, 6.5)
    assert M in dlg.table.buttons[sym].available


def test_ag_m_selection_survives_the_range_update(window):
    """AgM is silently dropped from the selection when the range is applied."""
    dlg = _window_range(window, 0.2, 6.5)
    dlg.set_selection([(47, 2), (47, 3), (49, 3), (82, 3)])
    dlg.set_range(0.2, 6.5)
    assert (47, 3) in dlg.table.selection()


def test_m_availability_still_follows_the_window(window):
    """A fix must not enable M everywhere: Ag M is out of a 1-6.5 keV window."""
    dlg = _window_range(window, 0.2, 6.5)
    assert M in dlg.table.buttons['Cd'].available
    assert M in dlg.table.buttons['Pb'].available
    dlg = _window_range(window, 1.0, 6.5)
    assert M not in dlg.table.buttons['Ag'].available


@pytest.mark.parametrize('split', [True, False])
@pytest.mark.parametrize('lo,hi', [(0.2, 6.5), (1.0, 6.5), (0.5, 12.0)])
def test_m_offered_exactly_when_the_fit_builds_it(split, lo, hi):
    """The table offers M from Session.shell_lines; that must agree with
    build_components, or a selection is silently dropped (or does nothing)."""
    from scrappyfit.gui.periodic import PeriodicTable
    from scrappyfit.session import Session
    s = Session()
    s.options.split_m_subshells = split
    s.options.e_low, s.options.e_high = lo, hi
    table = PeriodicTable(s.db, lines_fn=s.shell_lines)
    table.set_range(lo, hi)
    for b in table.buttons.values():
        if b.Z < 20:
            continue
        built = bool(s.build_components([(b.Z, M)]))
        assert (M in b.available) == built, (b.symbol, split, lo, hi)


def test_line_viewer_names_ag_m_subshells(window):
    """Split M lines are listed per subshell, not as identical 'M line' rows."""
    dlg = _window_range(window, 0.2, 6.5)
    dlg.lines.show_element(47)
    t = dlg.lines.table
    rows = [(t.item(r, 0).text(), t.item(r, 1).text())
            for r in range(t.rowCount())]
    m_groups = {g for sh, g in rows if sh == 'M'}
    assert {'M4', 'M5', 'M3'} <= m_groups
    assert 'M line' not in m_groups
