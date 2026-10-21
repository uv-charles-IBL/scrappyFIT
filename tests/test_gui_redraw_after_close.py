"""Canvases: a draw queued by draw_idle must not touch a canvas that is gone.

Synthetic data only; offscreen Qt (conftest sets QT_QPA_PLATFORM). Named to
run after test_gui_infra: a widget test before its excepthook test crashes
the interpreter (BUGS.md, 2026-10-21).
"""
import pytest

pytest.importorskip('PyQt5.QtWidgets')

BUG = 'BUGS.md: A deferred draw_idle fires after a window closes'


@pytest.mark.xfail(strict=True, reason=BUG)
@pytest.mark.parametrize('kind', ['SpectrumCanvas', 'MapCanvas'])
def test_queued_draw_on_deleted_canvas_is_a_noop(qtbot, capfd, kind):
    """A pending _draw_idle on a deleted canvas must not raise 'has been deleted'."""
    from PyQt5 import QtWidgets, sip
    from matplotlib.backend_tools import Cursors
    from scrappyfit.gui import canvases
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    canvas = getattr(canvases, kind)(parent)
    sip.delete(canvas)                    # keep the Python wrapper
    canvas._draw_pending = True
    canvas.set_cursor(Cursors.WAIT)       # what the draw's wait-cursor does
    canvas._draw_idle()
    out = capfd.readouterr()
    assert 'has been deleted' not in out.out + out.err


@pytest.mark.xfail(strict=True, reason=BUG)
def test_closing_main_window_cancels_pending_draws(qtbot):
    """MainWindow.close() must leave no draw pending on its spectrum and map canvases."""
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.spec.draw_idle()
    w.map.draw_idle()
    assert w.spec._draw_pending and w.map._draw_pending
    w.close()
    assert not w.spec._draw_pending
    assert not w.map._draw_pending
