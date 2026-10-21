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


@pytest.mark.xfail(strict=True, reason='BUGS.md: Choose which lines are '
                   'drawn and listed in the spectrum legend')
def test_element_groups_lists_groups_in_first_appearance_order(canvas):
    """element_groups() must name '<Sym> <shell>' groups, no artefacts."""
    _show(canvas)
    assert canvas.element_groups() == GROUPS


@pytest.mark.xfail(strict=True, reason='BUGS.md: Choose which lines are '
                   'drawn and listed in the spectrum legend')
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


@pytest.mark.xfail(strict=True, reason='BUGS.md: Choose which lines are '
                   'drawn and listed in the spectrum legend')
def test_hidden_groups_survive_a_new_show_and_ignore_unknown(canvas):
    """The hidden set must persist across show(); unknown names are ignored."""
    _show(canvas)
    canvas.set_legend('all')
    canvas.set_hidden(HIDDEN | {'Xx K'})
    _show(canvas)
    assert not HIDDEN_COMPS & set(_tokens(canvas))
    assert 'Ca' in _tokens(canvas)


@pytest.mark.xfail(strict=True, reason='BUGS.md: Choose which lines are '
                   'drawn and listed in the spectrum legend')
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
