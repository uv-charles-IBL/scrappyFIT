"""Plot canvases: the spectrum view and the map view.

These are deliberately dumb. They own no analysis state - they are handed
arrays and asked to draw them, and they emit signals when the user interacts.
All decisions live in Session. That separation is what lets the same analysis
be driven from a script, a notebook, or this window without duplicating logic.

Two interactions matter enough to build in:

  drag on the spectrum   select an energy range - used for zooming and for
                         reading off what sits under a feature
  drag on a map          select a rectangular region, which becomes a spectrum
                         mask. Fitting a masked region is the single most
                         useful thing this program does that GeoPIXE does not
                         make easy: the bulk spectrum of a heterogeneous
                         sample describes nowhere in it.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, SpanSelector
from PyQt5 import QtCore, QtWidgets

INK = '#151A22'
DATA = '#151A22'
MODEL = '#A8323F'
BACK = '#0F766E'
COMP = '#3B82F6'


class SpectrumCanvas(FigureCanvasQTAgg):
    """Spectrum with the fitted model, components and a residual panel."""

    rangeSelected = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        self.ax = self.fig.add_subplot(gs[0])
        self.axr = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.11)
        self._log = True
        self._show_comps = True
        self._labels = []
        self._span = SpanSelector(
            self.ax, self._on_span, 'horizontal', useblit=True,
            props=dict(alpha=0.2, facecolor='#2B6CB0'), interactive=False)

    def _on_span(self, lo, hi):
        if hi > lo:
            self.rangeSelected.emit(float(lo), float(hi))

    def set_log(self, on):
        self._log = bool(on)
        self.redraw()

    def set_components(self, on):
        self._show_comps = bool(on)
        self.redraw()

    def show(self, energy, counts, fit=None, background=None, labels=(),
             title=''):
        """labels: sequence of (text, energy_keV) drawn as vertical guides."""
        self._state = dict(energy=energy, counts=counts, fit=fit,
                           background=background, labels=list(labels),
                           title=title)
        self.redraw()

    def redraw(self):
        st = getattr(self, '_state', None)
        if st is None:
            return
        E, y = st['energy'], st['counts']
        r, bk = st['fit'], st['background']
        self.ax.clear()
        self.axr.clear()

        self.ax.step(E, y, where='mid', lw=0.8, color=DATA, label='data')
        if bk is not None:
            n = min(len(E), len(bk))
            self.ax.plot(E[:n], bk[:n], lw=1.0, ls='--', color=BACK,
                         label='background')
        if r is not None:
            n = min(len(E), len(r.model))
            self.ax.plot(E[:n], r.model[:n], lw=1.3, color=MODEL,
                         label='fit  chi2r=%.2f' % r.reduced_chi2)
            if self._show_comps and bk is not None:
                base = bk[:n]
                for i, nm in enumerate(r.names):
                    if r.areas[i] <= 0:
                        continue
                    c = r.profiles[:n, i] * r.areas[i]
                    if c.max() < 0.003 * max(y.max(), 1):
                        continue
                    self.ax.plot(E[:n], c + base, lw=0.6, alpha=0.55,
                                 color=COMP)
            sg = np.sqrt(np.maximum(r.model[:n], 1.0))
            d = (y[:n] - r.model[:n]) / sg
            self.axr.fill_between(E[:n], 0, d, where=d > 0, step='mid',
                                  color=MODEL, alpha=0.45)
            self.axr.fill_between(E[:n], 0, d, where=d <= 0, step='mid',
                                  color='#2B6CB0', alpha=0.35)
            self.axr.step(E[:n], d, where='mid', lw=0.6, color=INK)
            self.axr.axhspan(-2, 2, color='0.9', zorder=0)
            self.axr.set_ylim(-10, 10)

        for text, en in st['labels']:
            self.ax.axvline(en, color='0.85', lw=0.7, zorder=0)
            self.ax.annotate(text, (en, 1.0), xycoords=('data', 'axes fraction'),
                             xytext=(0, 2), textcoords='offset points',
                             ha='center', fontsize=8, color=INK)

        if self._log:
            self.ax.set_yscale('log')
            pos = y[y > 0]
            self.ax.set_ylim(0.5, (pos.max() if len(pos) else 10) * 4)
        else:
            self.ax.set_yscale('linear')
            self.ax.set_ylim(0, max(y.max() * 1.1, 1))
        self.ax.set_ylabel('counts / channel')
        self.axr.set_ylabel('resid/sig')
        self.axr.set_xlabel('energy (keV)')
        self.ax.grid(alpha=0.2, lw=0.5)
        self.axr.grid(alpha=0.25, lw=0.5)
        self.ax.legend(fontsize=8, frameon=False, loc='upper right')
        if st['title']:
            self.ax.set_title(st['title'], fontsize=10, color=INK)
        self.ax.tick_params(labelbottom=False)
        self.draw_idle()

    def set_xlim(self, lo, hi):
        self.ax.set_xlim(lo, hi)
        self.draw_idle()


class MapCanvas(FigureCanvasQTAgg):
    """An elemental map, with rectangle selection that becomes a mask."""

    regionSelected = QtCore.pyqtSignal(object)     # bool array, map-shaped

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.02, right=0.88, top=0.93, bottom=0.03)
        self._img = None
        self._cbar = None
        self._data = None
        self._sel = RectangleSelector(
            self.ax, self._on_rect, useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords='data', interactive=True,
            props=dict(facecolor='none', edgecolor='#A8323F', lw=1.4))

    def _on_rect(self, press, release):
        if self._data is None:
            return
        x0, x1 = sorted((int(press.xdata), int(release.xdata)))
        y0, y1 = sorted((int(press.ydata), int(release.ydata)))
        m = np.zeros(self._data.shape, dtype=bool)
        m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        if m.any():
            self.regionSelected.emit(m)

    def show_map(self, data, title='', cmap='magma'):
        self._data = np.asarray(data, dtype=float)
        self.ax.clear()
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        lo, hi = np.percentile(self._data, 2), np.percentile(self._data, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(self._data)), float(np.nanmax(self._data)) + 1e-9
        self._img = self.ax.imshow(self._data, origin='lower', cmap=cmap,
                                   vmin=lo, vmax=hi, interpolation='nearest')
        self._cbar = self.fig.colorbar(self._img, ax=self.ax, fraction=0.046)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=10, color=INK)
        self.draw_idle()

    def threshold_mask(self, percentile):
        """Everything at or above a percentile of the displayed map. This is
        how you isolate a phase rather than a rectangle."""
        if self._data is None:
            return None
        thr = np.percentile(self._data, percentile)
        return self._data >= thr


def toolbar_for(canvas, parent):
    return NavigationToolbar2QT(canvas, parent)
