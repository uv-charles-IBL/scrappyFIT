"""Spectrum legend with many element groups: per-group display selection.

BUGS.md: 'Choose which lines are drawn and listed in the spectrum legend'.
Synthetic data only; offscreen Qt (conftest sets QT_QPA_PLATFORM).
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt5.QtWidgets')

N = 1024
K_SYMS = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
# 18 K groups + Ag L, Ag M, Pb L, Pb M = 22 groups
NAMES = (K_SYMS + ['FeKb', 'AgL', 'AgM5', 'AgM4', 'PbL', 'PbM5', 'PbM4',
                   'pileup'])
GROUPS = [s + ' K' for s in K_SYMS] + ['Ag L', 'Ag M', 'Pb L', 'Pb M']


@pytest.fixture
def canvas(qtbot):
    """A SpectrumCanvas in a QApplication, closed at teardown."""
    from scrappyfit.gui.canvases import SpectrumCanvas
    c = SpectrumCanvas()
    qtbot.addWidget(c)
    return c


def _peak(c, s=3.0):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _show(canvas):
    """Show a fake fit with 22 element groups plus pile-up."""
    E = 0.01 * np.arange(N)
    bk = np.full(N, 20.0)
    centres = np.linspace(40, 980, len(NAMES))
    prof = np.stack([_peak(c) / _peak(c).sum() for c in centres], axis=1)
    areas = np.full(len(NAMES), 5000.0)
    model = bk + (prof * areas).sum(axis=1)
    r = SimpleNamespace(names=list(NAMES), areas=areas, profiles=prof,
                        model=model, reduced_chi2=1.0)
    canvas.show(E, model.copy(), fit=r, background=bk)
    return r


def _tokens(canvas):
    return [l.get_label().split(' ')[0] for l in canvas.ax.get_lines()]


def _legend_tokens(canvas):
    return [t.get_text().split(' ')[0]
            for t in canvas.ax.get_legend().get_texts()]


HIDDEN = {'Fe K', 'Ag M'}
HIDDEN_COMPS = {'Fe', 'FeKb', 'AgM5', 'AgM4'}


def test_element_groups_lists_groups_in_first_appearance_order(canvas):
    """element_groups() must name '<Sym> <shell>' groups, no artefacts."""
    _show(canvas)
    assert canvas.element_groups() == GROUPS


def test_hidden_groups_are_not_drawn_or_listed(canvas):
    """set_hidden() must drop the group's lines and legend entries."""
    _show(canvas)
    canvas.set_legend('all')
    assert HIDDEN_COMPS <= set(_tokens(canvas))      # drawn before hiding
    canvas.set_hidden(HIDDEN)
    assert canvas.hidden == HIDDEN
    toks = _tokens(canvas)
    assert not HIDDEN_COMPS & set(toks), toks
    assert not HIDDEN_COMPS & set(_legend_tokens(canvas))
    assert 'Ca' in toks and 'Ca' in _legend_tokens(canvas)
    assert 'pileup' in toks


def test_hidden_groups_survive_a_new_show_and_ignore_unknown(canvas):
    """The hidden set must persist across show(); unknown names are ignored."""
    _show(canvas)
    canvas.set_legend('all')
    canvas.set_hidden(HIDDEN | {'Xx K'})
    _show(canvas)
    assert not HIDDEN_COMPS & set(_tokens(canvas))
    assert 'Ca' in _tokens(canvas)


def test_legend_all_with_22_groups_stays_beside_the_axes(canvas):
    """The 'all' legend must not cover the data or run off the canvas."""
    canvas.resize(1000, 600)
    canvas.fig.set_size_inches(10, 6)
    _show(canvas)
    canvas.set_legend('all')
    canvas.draw()
    rend = canvas.get_renderer()
    leg = canvas.ax.get_legend().get_window_extent(rend)
    axb = canvas.ax.get_window_extent(rend)
    fb = canvas.fig.bbox
    assert leg.x0 >= axb.x1 - 1, ('legend overlaps axes', leg, axb)
    assert leg.x0 >= fb.x0 - 1 and leg.x1 <= fb.x1 + 1, ('off figure', leg, fb)
    assert leg.y0 >= fb.y0 - 1 and leg.y1 <= fb.y1 + 1, ('off figure', leg, fb)


def test_compact_legend_has_no_elements_and_off_has_none(canvas):
    """Compact must list data/fit/background/pileup only; off, no legend."""
    _show(canvas)
    canvas.set_legend('compact')
    toks = _legend_tokens(canvas)
    for nm in ('data', 'fit', 'background', 'pileup'):
        assert nm in toks, toks
    assert not {'Fe', 'Ca', 'AgM5'} & set(toks), toks
    canvas.set_legend('off')
    assert canvas.ax.get_legend() is None


def test_lines_dialog_hides_unticked_groups_and_keeps_others(qtbot):
    """Lines...: unticking a group hides it; groups not in this fit keep
    their earlier choice; nothing to choose before anything is drawn."""
    from PyQt5 import QtCore
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        assert w.lines_dialog() is None
        _show(w.spec)
        w.spec.hidden = {'Xx K'}
        dlg, lst = w.lines_dialog()
        assert [lst.item(i).text() for i in range(lst.count())] == GROUPS
        for i in range(lst.count()):
            if lst.item(i).text() in HIDDEN:
                lst.item(i).setCheckState(QtCore.Qt.Unchecked)
        w.apply_lines(lst)
        assert w.spec.hidden == HIDDEN | {'Xx K'}
        assert not HIDDEN_COMPS & set(_tokens(w.spec))
        dlg.close()
    finally:
        w.close()


def test_small_canvas_folds_elements_into_more_and_off_restores_axes(canvas):
    """At 700x400 the 22 groups cannot fit in 40% of the width: the legend
    folds the rest into '+ n more', stays on the canvas and off the axes;
    'off' gives the axes their full width back."""
    from scrappyfit.gui.canvases import AX_RIGHT, LEGEND_MAX_FRAC
    canvas.resize(700, 400)
    canvas.fig.set_size_inches(7, 4)
    _show(canvas)
    canvas.set_legend('all')
    canvas.draw()
    rend = canvas.get_renderer()
    leg = canvas.ax.get_legend()
    texts = [t.get_text() for t in leg.get_texts()]
    assert texts[-1].startswith('+ ') and 'more' in texts[-1], texts
    for nm in ('data', 'background', 'pileup', 'fit'):
        assert any(t.startswith(nm) for t in texts), (nm, texts)
    lb = leg.get_window_extent(rend)
    assert lb.x0 >= canvas.ax.get_window_extent(rend).x1 - 1
    assert lb.x1 <= canvas.fig.bbox.x1 + 1 and lb.y0 >= canvas.fig.bbox.y0 - 1
    assert lb.width <= LEGEND_MAX_FRAC * canvas.fig.bbox.width + 1
    canvas.set_legend('off')
    assert canvas.fig.subplotpars.right == AX_RIGHT


def test_hiding_a_group_keeps_the_other_colours(canvas):
    """Hiding Fe must not recolour the elements drawn after it."""
    _show(canvas)
    canvas.set_legend('all')

    def colours():
        return {l.get_label().split(' ')[0]: l.get_color()
                for l in canvas.ax.get_lines()}
    before = colours()
    canvas.set_hidden({'Fe K'})
    after = colours()
    for nm in ('Co', 'Zn', 'AgL', 'PbM4'):
        assert after[nm] == before[nm], nm
