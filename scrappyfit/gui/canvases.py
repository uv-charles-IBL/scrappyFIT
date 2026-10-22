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

import math
import re

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.widgets import PolygonSelector, RectangleSelector, SpanSelector
from PyQt5 import QtCore, QtWidgets, sip

from ..analysis import masking as MSK

INK = '#151A22'
DATA = '#3C3C3C'          # data sits back so the model reads on top of it
MODEL = '#D01C1F'         # the fit, the line you check first
BACK = '#00857A'          # background
COMP = '#1F4FD8'          # individual components

#: Components that are never hidden by the small-component threshold, because
#: they explain features no element does. name -> (colour, style, width)
ARTEFACT_STYLE = {
    'pileup': ('#C4187A', '-', 1.6),
    'escape': ('#7B3FA8', '-', 1.4),
}

# Components are drawn in a rotating palette rather than one colour. With ten
# overlapping elements a single hue is an unreadable thicket, and telling
# which component is which is most of what the plot is for.
COMP_CYCLE = ['#1F4FD8', '#E07A00', '#7B3FA8', '#00873E', '#C4187A',
              '#0F8FA8', '#8A6D00', '#B03030']

REF = '#8C9AAE'           # the whole-field spectrum drawn behind a region's


def is_pileup(name):
    """The pile-up terms of a fit: 'pileup' (sum peaks), 'pileup flat' /
    'pileup ...' (plateau) and 'subthreshold pileup'."""
    return name.startswith('pileup') or name == 'subthreshold pileup'

#: Legend modes. 'compact' keeps data, fit, background and artefacts; 'all'
#: adds every drawn element component; 'off' draws none.
LEGEND_MODES = ('compact', 'all', 'off')

#: Figure fractions of the spectrum axes when no legend takes room.
AX_RIGHT, AX_TOP, AX_BOTTOM = 0.98, 0.94, 0.11
#: The legend sits right of the axes and may take at most this share of the
#: canvas width; entries past that are summarised in one '+ n more' line.
LEGEND_MAX_FRAC = 0.4

# '<symbol><shell>' as Session names components: 'Fe' (K), 'FeKb', 'FeL',
# 'AgM5', 'AgMz'. Pile-up, escape and background names are lower case.
_COMP_RE = re.compile(r'^([A-Z][a-z]?)(Kb|L|M[1-5z]?)?$')


def element_group(name):
    """'Fe K' for 'Fe' and 'FeKb', 'Ag M' for 'AgM5', None for anything
    that is not an element component (pile-up, escape, background)."""
    m = _COMP_RE.match(name)
    if m is None:
        return None
    return '%s %s' % (m.group(1), (m.group(2) or 'K')[0])


class _Canvas(FigureCanvasQTAgg):
    """A Qt canvas that is safe to draw on after its widget is gone.

    draw_idle queues _draw_idle on the event loop, and the toolbar sets a
    wait cursor around each draw (backend_bases _wait_cursor_for_draw_cm).
    With the C++ widget gone by then, that set_cursor printed 'wrapped
    C/C++ object of type SpectrumCanvas has been deleted' (fast suite,
    2026-10-21; how it went mid-draw was not reproduced).
    cancel_draws() drops a queued draw; drawing and the cursor are no-ops
    once the widget is deleted.
    """

    def cancel_draws(self):
        self._draw_pending = False

    def _draw_idle(self):
        if sip.isdeleted(self):
            self._draw_pending = False
            return
        super()._draw_idle()

    def set_cursor(self, cursor):
        if not sip.isdeleted(self):
            super().set_cursor(cursor)


class SpectrumCanvas(_Canvas):
    """Spectrum with the fitted model, components and a residual panel."""

    rangeSelected = QtCore.pyqtSignal(float, float)
    LEGEND_MODES = LEGEND_MODES

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
        self._artefact_curves = False   # summed escape, pile-up curves
        self.legend_mode = 'compact'
        self.hidden = set()     # element groups not drawn ('Fe K', 'Ag M')
        self._legend_layout = {}    # (size, entries) -> (kept, columns)
        self._labels = []
        self._markers = []      # identification markers, drawn over the data
        self._span = SpanSelector(
            self.ax, self._on_span, 'horizontal', useblit=True,
            props=dict(alpha=0.2, facecolor='#2B6CB0'), interactive=False)
        # the legend's columns depend on the canvas height, so a resize
        # lays it out again
        self.mpl_connect('resize_event', lambda ev: self._place_legend())

    def _on_span(self, lo, hi):
        if hi > lo:
            self.rangeSelected.emit(float(lo), float(hi))

    def set_log(self, on):
        self._log = bool(on)
        self.redraw()

    def set_components(self, on):
        self._show_comps = bool(on)
        self.redraw()

    def set_artefact_curves(self, on):
        """Draw the fit's escape lines and pile-up terms as two summed
        curves, 'escapes' and 'pile-up', as GeoPIXE overlays its 'Escapes'
        and 'Pileup' spectra (spectrum_display_eventcb.pro:3452, :3780).
        Drawn with or without the element components; display only."""
        self._artefact_curves = bool(on)
        self.redraw()

    def set_legend(self, mode):
        if mode not in LEGEND_MODES:
            raise ValueError('legend mode %r not in %s' % (mode, LEGEND_MODES))
        self.legend_mode = mode
        self.redraw()

    def set_hidden(self, groups):
        """Element groups ('Fe K', 'Ag M', as element_groups() names them)
        whose components are neither drawn nor listed. Kept across fits, so
        a refit keeps the selection; names not in the fit are ignored."""
        self.hidden = set(groups)
        self.redraw()

    def element_groups(self):
        """The current fit's element groups, in fit order."""
        r = (getattr(self, '_state', None) or {}).get('fit')
        out = []
        for nm in getattr(r, 'names', ()):
            g = element_group(nm)
            if g is not None and g not in out:
                out.append(g)
        return out

    def set_markers(self, markers):
        """Vertical guides at identified line energies.

        markers: [(label, energy_keV, colour)]. Drawn where the line WOULD
        fall given the current calibration, which is the point - if the guide
        misses the peak, either the identification is wrong or the calibration
        is, and both are worth knowing before fitting anything.
        """
        self._markers = list(markers or [])
        self.redraw()

    def clear_markers(self):
        self._markers = []
        self.redraw()

    def show(self, energy, counts, fit=None, background=None, labels=(),
             title='', xlim=None, data_label='data', reference=None):
        """labels: sequence of (text, energy_keV) drawn as vertical guides.

        data_label names the data in the legend (a region's name, pixels and
        counts when a mask is active). reference = (counts, label) draws a
        second spectrum behind the data, e.g. the whole field under a region.

        xlim fixes the visible range. Without it a guide for a line far
        outside the spectrum - a heavy element's L series at 30 keV, say -
        silently stretches the axis until the actual data is a sliver at the
        left edge.
        """
        self._state = dict(energy=energy, counts=counts, fit=fit,
                           background=background, labels=list(labels),
                           title=title, xlim=xlim, data_label=data_label,
                           reference=reference)
        self.redraw()

    def redraw(self):
        st = getattr(self, '_state', None)
        if st is None:
            return
        E, y = st['energy'], st['counts']
        r, bk = st['fit'], st['background']
        self.ax.clear()
        self.axr.clear()
        self._legend_items = None

        ref = st.get('reference')
        if ref is not None:
            yr = np.asarray(ref[0], float)[:len(E)]
            self.ax.step(E[:len(yr)], yr, where='mid', lw=0.8, color=REF,
                         alpha=0.8, zorder=1, label=ref[1])
        self.ax.step(E, y, where='mid', lw=0.9, color=DATA, alpha=0.85,
                     label=st.get('data_label') or 'data')
        if bk is not None:
            n = min(len(E), len(bk))
            self.ax.plot(E[:n], bk[:n], lw=1.4, ls='--', color=BACK,
                         label='background')
        if r is not None:
            n = min(len(E), len(r.model))
            if self._show_comps and bk is not None:
                base = bk[:n]
                k = 0
                for i, nm in enumerate(r.names):
                    if r.areas[i] <= 0:
                        continue
                    c = r.profiles[:n, i] * r.areas[i]
                    special = nm in ARTEFACT_STYLE
                    if self._artefact_curves and is_pileup(nm):
                        continue            # drawn summed, below
                    # Elements are hidden below 0.3% of the tallest peak, or
                    # a busy spectrum turns into spaghetti. Artefacts are NOT
                    # subject to that: pile-up on run 287427 is 58 counts
                    # against a 700,000-count silicon peak, so the threshold
                    # hid it completely - and pile-up is exactly the thing an
                    # operator needs to SEE, because it is what stops them
                    # assigning a real element to a sum peak. GeoPIXE shows
                    # its 'sum' pseudo-element for the same reason.
                    if not special and c.max() < 0.003 * max(y.max(), 1):
                        continue
                    grp = element_group(nm)
                    if grp in self.hidden:
                        # keep its palette slot, so hiding one element does
                        # not recolour the others
                        k += not special
                        continue
                    if special:
                        col, ls, lw = ARTEFACT_STYLE[nm]
                        self.ax.plot(E[:n], c + base, lw=lw, ls=ls,
                                     color=col, alpha=0.95, zorder=3,
                                     label='%s (%.0f counts)' % (nm,
                                                                 r.areas[i]))
                    else:
                        col = COMP_CYCLE[k % len(COMP_CYCLE)]
                        k += 1
                        lab = ('%s (%.0f counts)' % (nm, r.areas[i])
                               if self.legend_mode == 'all' else None)
                        ln, = self.ax.plot(E[:n], c + base, lw=1.0,
                                           alpha=0.75, color=col, label=lab)
                        # the legend may fold elements into '+ n more';
                        # other components (pile-up plateau) always show
                        ln._sf_element = grp is not None
            if self._artefact_curves:
                self._draw_artefact_curves(r, E, n, bk, st.get('xlim'))
            # the model is drawn LAST so it is never buried under a component
            self.ax.plot(E[:n], r.model[:n], lw=1.8, color=MODEL, zorder=4,
                         label='fit  chi2r=%.2f' % r.reduced_chi2)
            sg = np.sqrt(np.maximum(r.model[:n], 1.0))
            d = (y[:n] - r.model[:n]) / sg
            self.axr.fill_between(E[:n], 0, d, where=d > 0, step='mid',
                                  color=MODEL, alpha=0.55)
            self.axr.fill_between(E[:n], 0, d, where=d <= 0, step='mid',
                                  color='#1F4FD8', alpha=0.45)
            self.axr.step(E[:n], d, where='mid', lw=0.7, color=INK)
            self.axr.axhspan(-2, 2, color='#DCDCDC', zorder=0)
            self.axr.set_ylim(-10, 10)

        # decide the visible window FIRST, so that nothing drawn afterwards
        # can stretch the axis
        lo_x, hi_x = st.get('xlim') or (float(E[0]), float(E[-1]))
        self.ax.set_xlim(lo_x, hi_x)

        for text, en in st['labels']:
            if not (lo_x <= en <= hi_x):
                continue
            self.ax.axvline(en, color='0.85', lw=0.7, zorder=0)
            self.ax.annotate(text, (en, 1.0), xycoords=('data', 'axes fraction'),
                             xytext=(0, 2), textcoords='offset points',
                             ha='center', fontsize=8, color=INK)

        # Identification markers. Every marker gets its guide line, but a
        # label is only placed where there is room for it: text is stacked
        # into whichever row is free at that energy, and if all rows are
        # occupied the label is dropped rather than printed on top of another.
        # A crowded low-energy region otherwise turns into an unreadable
        # block, which is exactly where the labels are most wanted.
        span = max(hi_x - lo_x, 1e-6)
        min_gap = 0.055 * span            # roughly one label width
        rows = [-1e9] * 5                 # rightmost labelled energy per row
        for text, en, col in sorted(self._markers, key=lambda m: m[1]):
            if not (lo_x <= en <= hi_x):
                continue
            self.ax.axvline(en, color=col, lw=1.0, ls='--', alpha=0.8,
                            zorder=5)
            slot = next((i for i, x in enumerate(rows)
                         if en - x >= min_gap), None)
            if slot is None:
                continue
            rows[slot] = en
            self.ax.annotate(text, (en, 0.985 - 0.052 * slot),
                             xycoords=('data', 'axes fraction'),
                             xytext=(3, 0), textcoords='offset points',
                             ha='left', va='top', fontsize=7.5, color=col,
                             zorder=6, clip_on=False,
                             bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                       ec=col, lw=0.5, alpha=0.9))

        # the reference spectrum counts toward the y range, so an overlay
        # that sits above the region is not cut off at the top
        yy = y if ref is None else np.concatenate([y, yr])
        if self._log:
            self.ax.set_yscale('log')
            pos = yy[yy > 0]
            self.ax.set_ylim(0.5, (pos.max() if len(pos) else 10) * 4)
        else:
            self.ax.set_yscale('linear')
            self.ax.set_ylim(0, max(yy.max() * 1.1, 1))
        self.ax.set_ylabel('counts / channel')
        self.axr.set_ylabel('resid/sig')
        self.axr.set_xlabel('energy (keV)')
        self.ax.grid(alpha=0.2, lw=0.5)
        self.axr.grid(alpha=0.25, lw=0.5)
        if self.legend_mode != 'off':
            self._legend_items = self.ax.get_legend_handles_labels()
        self._place_legend(draw=False)
        if st['title']:
            self.ax.set_title(st['title'], fontsize=10, color=INK)
        self.ax.tick_params(labelbottom=False)
        self.draw_idle()

    def _draw_artefact_curves(self, r, E, n, bk, xlim):
        """Summed escape and pile-up spectra over the background. The
        counts in the legend are those inside xlim (the fit window), not
        the whole spectrum the curve is drawn over."""
        lo, hi = xlim or (float(E[0]), float(E[n - 1]))
        win = (E[:n] >= lo) & (E[:n] <= hi)
        base = bk[:n] if bk is not None else 0.0
        esc = getattr(r, 'escape_spectrum', None)
        esc = esc() if callable(esc) else None
        pile = [i for i, nm in enumerate(r.names)
                if is_pileup(nm) and r.areas[i] > 0]
        for curve, nm, what in (
                (esc[:n] if esc is not None else None, 'escape', 'escapes'),
                (sum(r.profiles[:n, i] * r.areas[i] for i in pile)
                 if pile else None, 'pileup', 'pile-up')):
            if curve is None:
                continue
            col, ls, lw = ARTEFACT_STYLE[nm]
            self.ax.plot(E[:n], curve + base, lw=lw + 0.2, ls=ls, color=col,
                         alpha=0.95, zorder=3,
                         label='%s (fit, %.0f counts)' % (what,
                                                          curve[win].sum()))

    def _place_legend(self, draw=True):
        """Put the legend right of the axes, never over the data.

        It runs from the top of the spectrum down beside the residuals, in
        as many columns as that height needs (22 element groups on 287424
        ran off the plot in one), and the axes give up exactly its width.
        Past LEGEND_MAX_FRAC of the canvas, the remaining element entries
        become one '+ n more' line; Lines... chooses what is listed.
        """
        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()
        items = getattr(self, '_legend_items', None)
        if not items or not items[0]:
            self.fig.subplots_adjust(right=AX_RIGHT)
            if draw:
                self.draw_idle()
            return
        handles, labels = list(items[0]), list(items[1])
        W, H = self.fig.bbox.width, self.fig.bbox.height
        if W < 2 or H < 2:                 # collapsed; laid out on resize
            return
        rend = self.get_renderer()
        fixed = [k for k, h in enumerate(handles)
                 if getattr(h, '_sf_element', False) is False]
        extra = [k for k in range(len(handles)) if k not in fixed]

        def make(keep, ncol=None):
            idx = fixed + extra[:keep]
            hs, ls = [handles[k] for k in idx], [labels[k] for k in idx]
            if keep < len(extra):
                hs.append(Line2D([], [], lw=0))
                ls.append('+ %d more (Lines...)' % (len(extra) - keep))
            if ncol is None:
                # one column first, to measure a row
                lg = self.ax.legend(hs, ls, fontsize=8, frameon=False)
                one = lg.get_window_extent(rend).height / max(len(hs), 1)
                rows = max(int((AX_TOP - AX_BOTTOM) * H / max(one, 1.0)), 1)
                lg.remove()
                ncol = math.ceil(len(hs) / rows)
            lg = self.ax.legend(hs, ls, fontsize=8, frameon=False,
                                loc='upper left', borderaxespad=0.0,
                                ncol=ncol, bbox_to_anchor=(1.0, AX_TOP),
                                bbox_transform=self.fig.transFigure)
            return lg, ncol

        # Measuring is most of a redraw's legend cost, so the layout found
        # for a canvas size and set of entries is reused.
        key = (W, H, tuple(labels), tuple(fixed))
        hit = self._legend_layout.get(key)
        if hit is not None:
            keep, ncol = hit
            lg, _ = make(keep, ncol)
            w = lg.get_window_extent(rend).width
        else:
            keep = len(extra)
            lg, ncol = make(keep)
            w = lg.get_window_extent(rend).width
            while w > LEGEND_MAX_FRAC * W and keep > 0:
                keep = keep * 2 // 3
                lg.remove()
                lg, ncol = make(keep)
                w = lg.get_window_extent(rend).width
            if len(self._legend_layout) > 32:
                self._legend_layout.clear()
            self._legend_layout[key] = (keep, ncol)
        pad = 8.0                          # px between the axes and legend
        right = max(1.0 - (w + 2 * pad) / W, 0.3)
        self.fig.subplots_adjust(right=right)
        lg.set_bbox_to_anchor((right + pad / W, AX_TOP),
                              transform=self.fig.transFigure)
        if draw:
            self.draw_idle()

    def set_xlim(self, lo, hi):
        self.ax.set_xlim(lo, hi)
        self.draw_idle()

    def save_image(self, path, dpi=200, title=None):
        """Write the current view to a file, at publication resolution.

        Whatever is on screen is what is written - markers, labels, zoom and
        all - so the figure in a report is the figure that was looked at.
        """
        if title:
            self.ax.set_title(title, fontsize=10, color=INK)
        self.fig.savefig(path, dpi=dpi, facecolor='white',
                         bbox_inches='tight')
        return path


#: Map colour tables offered in the Maps tab. GeoPIXE's menu has its own
#: default, inferno, viridis and grey (gimage.pro:414-452).
MAP_CMAPS = ('magma', 'viridis', 'inferno', 'cividis', 'gray', 'turbo')
#: Intensity scales, as GeoPIXE's L / G / S droplist (build_image_scale.pro).
MAP_SCALES = ('linear', 'log', 'sqrt')
#: Room kept right of the map for the colour bar and its labels (pixels).
MAP_BAR_PX = 90
#: Default colour limits, percentiles of the shown map.
MAP_PCT = (2.0, 99.0)


class MapCanvas(_Canvas):
    """An elemental map, with rectangle selection that becomes a mask."""

    regionSelected = QtCore.pyqtSignal(object)     # bool array, map-shaped
    polygonSelected = QtCore.pyqtSignal(object)    # bool array, map-shaped

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.02, top=0.93, bottom=0.03)
        self._reserve_bar_room()
        self.mpl_connect('resize_event', self._reserve_bar_room)
        # The colour bar gets its own axes, made once. fig.colorbar(ax=...)
        # steals room from the map axes on every call and remove() does not
        # give it back, so each Show map drew a smaller map further left.
        self.cax = self.fig.add_axes([0.88, 0.08, 0.035, 0.80])
        self._img = None
        self._cbar = None
        self._data = None
        self._binning = None
        self._mask_art = []
        # display settings survive Show map, so stepping through elements
        # keeps the scale the user chose
        self.cmap, self.scale, self.pct = 'magma', 'linear', MAP_PCT
        self._tool = 'rectangle'
        self._sel = self._poly = None
        self._make_selectors()

    def _make_selectors(self):
        """(Re)build the rectangle and polygon tools on self.ax. Show map
        clears the axes, which drops the tools' artists, so each map gets
        fresh ones; only the active tool listens."""
        for w in (self._sel, self._poly):
            if w is not None:
                w.set_active(False)
                w.disconnect_events()
        self._sel = RectangleSelector(
            self.ax, self._on_rect, useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords='data', interactive=True,
            props=dict(facecolor='none', edgecolor='#A8323F', lw=1.4))
        self._poly = PolygonSelector(
            self.ax, self._on_poly, useblit=True,
            props=dict(color='#A8323F', lw=1.4),
            handle_props=dict(markersize=4))
        self.set_tool(self._tool)

    def set_tool(self, tool):
        """'rectangle', 'polygon', or anything else (flood fill: clicks are
        handled by the window) to switch both drawing tools off."""
        if self._tool == 'polygon' and tool != 'polygon':
            self._poly.clear()           # drop a half-drawn polygon
        self._tool = tool
        self._sel.set_active(tool == 'rectangle')
        self._poly.set_active(tool == 'polygon')

    @staticmethod
    def _pix(v):
        """Data coordinate -> pixel index; pixel i spans i-0.5 .. i+0.5."""
        return int(np.floor(v + 0.5))

    def _on_rect(self, press, release):
        if self._data is None:
            return
        x0, x1 = sorted((self._pix(press.xdata), self._pix(release.xdata)))
        y0, y1 = sorted((self._pix(press.ydata), self._pix(release.ydata)))
        m = np.zeros(self._data.shape, dtype=bool)
        m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        if m.any():
            self.regionSelected.emit(m)

    def _on_poly(self, verts):
        """A closed polygon -> the pixels whose centres it contains."""
        if self._data is None or len(verts) < 3:
            return
        # masking.polygon tests pixel centres at index + 0.5; on this
        # axes pixel i is centred on i, so shift the vertices by half
        m = MSK.polygon(self._data.shape,
                        [(x + 0.5, y + 0.5) for x, y in verts])
        # start the next polygon fresh; not from inside the callback
        poly = self._poly
        QtCore.QTimer.singleShot(
            0, lambda: None if sip.isdeleted(self) else poly.clear())
        if m.any():
            self.polygonSelected.emit(m)

    def show_map(self, data, title='', cmap=None, binning=None):
        """binning: raster pixels per map pixel; inferred from the size if
        not given, which fails for sizes several binnings make."""
        self._data = np.asarray(data, dtype=float)
        self._binning = binning
        self.ax.clear()
        self._mask_art = []
        if cmap is not None:
            self.cmap = cmap
        self._img = self.ax.imshow(self._data, origin='lower',
                                   cmap=self.cmap, interpolation='nearest',
                                   norm=self._norm(*self.limits()))
        # One colour bar for the life of the canvas, pointed at each new map.
        # A new Colorbar per map leaks every old one: each wraps the axes
        # locator the last installed and, in matplotlib 3.11.2, cax.remove
        # too (colorbar.py:448, functools.partial(_remove_cbar_axes,
        # cbar=self); checked 2026-10-20), and each colour bar holds its map
        # image and data.
        if self._cbar is None:
            self.cax.set_axes_locator(self._beside_map)
            self._cbar = self.fig.colorbar(self._img, cax=self.cax)
        else:
            # Colorbar.__init__ is the only place matplotlib subscribes the
            # bar to its image's 'changed' signal, so wire each new image up
            # here, or later scale, limit and colour-map edits leave the bar
            # behind (a linear bar over a log map).
            self._img.colorbar = self._cbar
            # stored as Colorbar.__init__ does, so Colorbar.remove() can undo it
            self._img.colorbar_cid = self._img.callbacks.connect(
                'changed', self._cbar.update_normal)
            self._cbar.update_normal(self._img)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=10, color=INK)
        self._make_selectors()
        if self.toolbar is not None:
            self.toolbar.update()        # Home is this map, not the last one
        self.draw_idle()

    def _beside_map(self, cax, renderer):
        """Put the colour bar just right of the map as drawn. The map keeps
        its aspect, so in a wide window it sits in the middle of its axes
        box and a bar at a fixed figure fraction ends up far from it."""
        p = self.ax.get_position(original=False)       # aspect applied
        w = self.fig.get_size_inches()[0] * self.fig.dpi
        return Bbox.from_bounds(p.x1 + 10.0 / w, p.y0, 16.0 / w, p.height)

    def _reserve_bar_room(self, event=None):
        """Keep MAP_BAR_PX at the right for the bar and its tick labels, at
        any canvas width, so the bar never has to sit on the map."""
        w = self.fig.get_size_inches()[0] * self.fig.dpi
        self.fig.subplots_adjust(right=max(0.5, 1.0 - MAP_BAR_PX / w))

    def limits(self, pct=None):
        """Colour limits at percentiles of the map (default self.pct).

        Percentiles, not GeoPIXE's % of maximum (gimage.pro:1700-1713): one
        hot pixel sets the maximum, and then every slider position means
        something different on the next map."""
        d = self._data
        p0, p1 = self.pct if pct is None else pct
        lo, hi = np.nanpercentile(d, p0), np.nanpercentile(d, p1)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(d)), float(np.nanmax(d)) + 1e-9
        return float(lo), float(hi)

    def default_limits(self):
        """The colour limits Reset view returns to: 2nd-99th percentile."""
        return self.limits(MAP_PCT)

    def _norm(self, lo, hi):
        if self.scale == 'log':
            # Log needs 0 < lo < hi. The top falls back to the largest
            # positive pixel when the high percentile is not positive (a
            # sparse count map's 99th percentile can be 0; a net map can be
            # all negative). The floor is 5 decades below the top, as
            # GeoPIXE's log scale (build_image_scale.pro:39-42); zeros and
            # negatives draw in the bottom colour.
            if not hi > 0:
                pos = self._data[self._data > 0]
                if not pos.size:
                    return mcolors.Normalize(lo, hi, clip=False)
                hi = float(pos.max())
            floor = hi * 1e-5
            lo = min(max(lo, floor), hi / 10)
            return mcolors.LogNorm(lo, hi, clip=False)
        if self.scale == 'sqrt':
            return mcolors.PowerNorm(0.5, lo, hi, clip=False)
        return mcolors.Normalize(lo, hi, clip=False)

    def set_display(self, cmap=None, scale=None, pct=None):
        """Colour table, intensity scale and percentile limits; redraws the
        shown map at once and applies to every later Show map."""
        if cmap is not None:
            self.cmap = cmap
        if scale is not None:
            if scale not in MAP_SCALES:
                raise ValueError('scale must be one of %s' % (MAP_SCALES,))
            self.scale = scale
        if pct is not None:
            p0, p1 = float(pct[0]), float(pct[1])
            if not 0 <= p0 < p1 <= 100:
                raise ValueError('need 0 <= low < high <= 100, got %r' % (pct,))
            self.pct = (p0, p1)
        if self._img is None:
            return
        self._img.set_cmap(self.cmap)
        self._img.set_norm(self._norm(*self.limits()))
        self.draw_idle()

    def reset_view(self):
        """Back to the view Show map gave: whole field, default contrast.
        The mask is kept; Clear mask is its own action."""
        if self._data is None:
            return
        ny, nx = self._data.shape
        self.ax.set_xlim(-0.5, nx - 0.5)
        self.ax.set_ylim(-0.5, ny - 0.5)
        tb = self.toolbar
        if tb is not None:
            tb.update()                  # the Home view is this one again
            tb.push_current()
        # A new norm, not set_clim: set_clim moves vmin then vmax, and the
        # colour bar 'repairs' the moment the new vmin is above the old vmax
        # (a reset from contrast (0, 1) came back at 0.9, not the 2.09 asked
        # for: matplotlib 3.8.0, test_reset_view_button_restores_extent_and_
        # contrast, 2026-10-20).
        self.pct = MAP_PCT
        self._img.set_norm(self._norm(*self.limits()))
        self.draw_idle()

    def binning(self, raster=256):
        """Raster pixels per displayed pixel (1 before any map is shown)."""
        if self._data is None:
            return 1
        return MSK.bin_factor(self._data.shape[0], raster, self._binning)

    def set_mask(self, mask):
        """Outline the active mask over the map; None removes it.

        mask is in raster pixels (Session.mask), drawn on the binned map's
        grid, so a mask edge that moves by one displayed pixel is seen to.
        """
        for a in self._mask_art:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._mask_art = []
        if mask is not None and self._data is not None:
            m = np.asarray(mask, dtype=float)
            # raster pixels on the map's grid: map pixel i covers raster
            # pixels i*b .. i*b+b-1, so the raster's far edge is at
            # 256/b - 0.5, short of the last (partial) map pixel's edge
            # when b does not divide the raster
            b = self.binning(m.shape[0])
            r = m.shape[0] / float(b)
            ext = (-0.5, r - 0.5, -0.5, r - 0.5)
            tint = np.zeros(m.shape + (4,))
            tint[m > 0] = (0.35, 0.85, 1.0, 0.30)
            self._mask_art.append(self.ax.imshow(
                tint, origin='lower', extent=ext, interpolation='nearest',
                zorder=3))
            if 0 < m.sum() < m.size:        # a full field has no edge
                self._mask_art.append(self.ax.contour(
                    m, levels=[0.5], colors='#39D5FF', linewidths=1.2,
                    origin='lower', extent=ext, zorder=4))
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
