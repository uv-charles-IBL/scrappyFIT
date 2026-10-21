"""GUI layout: nothing in the left panel may be cut off.

Fast; offscreen Qt (conftest sets QT_QPA_PLATFORM).
"""
import pytest

QtWidgets = pytest.importorskip('PyQt5.QtWidgets')
from PyQt5 import QtCore, QtGui  # noqa: E402


def clipped_widgets(win):
    """Descriptions of everything in the left panel that is cut off."""
    scroll = win._left
    vp = scroll.viewport()
    bad = []
    if scroll.horizontalScrollBar().isVisible():
        bad.append('horizontal scrollbar visible (max=%d)'
                   % scroll.horizontalScrollBar().maximum())
    for w in scroll.widget().findChildren(QtWidgets.QWidget):
        if not isinstance(w, (QtWidgets.QLabel, QtWidgets.QCheckBox,
                              QtWidgets.QGroupBox)):
            continue
        if not w.isVisibleTo(scroll.widget()) or not w.isVisible():
            continue
        tl = w.mapTo(vp, QtCore.QPoint(0, 0))
        full = QtCore.QRect(tl, w.size())
        # The panel scrolls vertically by design, so only the horizontal
        # extent counts as visible area here.
        vis = full.intersected(QtCore.QRect(vp.rect().left(), full.top(),
                                            vp.rect().width(), full.height()))
        name = '%s %r' % (type(w).__name__,
                          (w.title() if isinstance(w, QtWidgets.QGroupBox)
                           else w.text())[:40])
        if isinstance(w, QtWidgets.QGroupBox):
            opt = QtWidgets.QStyleOptionGroupBox()
            w.initStyleOption(opt)
            r = w.style().subControlRect(
                QtWidgets.QStyle.CC_GroupBox, opt,
                QtWidgets.QStyle.SC_GroupBoxLabel, w)
            r = r.translated(tl)
            if not vis.contains(r) or full.right() > vp.rect().right():
                bad.append('%s title/edge cut (title %s, box right %d, vp %d)'
                           % (name, r.getRect(), full.right(),
                              vp.rect().right()))
        elif isinstance(w, QtWidgets.QLabel) and w.wordWrap():
            if full.left() < 0 or full.right() > vp.rect().right():
                bad.append('%s wrapped label outside viewport (%d..%d of %d)'
                           % (name, full.left(), full.right(),
                              vp.rect().right()))
            elif w.heightForWidth(w.width()) > w.height():
                bad.append('%s wrapped label too short (%d < %d)'
                           % (name, w.height(), w.heightForWidth(w.width())))
        else:
            if w.sizeHint().width() > vis.width():
                bad.append('%s needs %d px, %d visible'
                           % (name, w.sizeHint().width(), vis.width()))
    return bad


@pytest.mark.parametrize('scale', [1.0, 1.25, 1.5])
def test_left_panel_text_is_not_clipped(qtbot, scale):
    """At 1500x950 and 100-150 % font scaling no left-panel label, checkbox or
    group box is cut off or needs a horizontal scrollbar."""
    from scrappyfit.gui.app import MainWindow
    app = QtWidgets.QApplication.instance()
    orig = app.font()
    assert orig.pointSizeF() > 0, 'app font is in px; scaling would do nothing'
    f = QtGui.QFont(orig)
    f.setPointSizeF(orig.pointSizeF() * scale)
    app.setFont(f)
    try:
        win = MainWindow()
        qtbot.addWidget(win)
        win.chk_preview.setChecked(False)
        win.resize(1500, 950)
        win.show()
        qtbot.waitExposed(win)
        for _ in range(3):
            app.processEvents()
        bad = clipped_widgets(win)
        assert not bad, '\n'.join(bad)
    finally:
        app.setFont(orig)
