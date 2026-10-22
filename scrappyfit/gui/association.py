"""Association plot: one element map against another, and a region drawn on
the plot selecting those pixels on the map (as GeoPIXE's Element
Associations, corr.pro; geopixe-ref 2026-10-21).

The plot is a 2-D histogram of the two maps' pixel values, density as colour
on a log scale. Draw a polygon round a cluster and its pixels become the
mask, combined with the current one by the chosen op (GeoPIXE's include /
further refine / reject, corr.pro:566-572).
"""

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from PyQt5 import QtCore, QtWidgets

from ..analysis import association as AS
from .canvases import _Canvas

#: (label, masking.combine op or 'replace'), as GeoPIXE's Include / Further
#: refine / Reject (corr.pro:566-572)
ASSOC_OPS = (('replace the mask', 'replace'), ('add to the mask', 'add'),
             ('refine (keep only both)', 'and'),
             ('take from the mask', 'subtract'))


class AssociationDialog(QtWidgets.QDialog):
    """make_map(name) -> (map, title, units, binning) builds a map the way
    the Maps tab would (same kind, background and binning). maskSelected
    carries (map-shaped bool mask, region name, op, binning of the plotted
    maps)."""

    maskSelected = QtCore.pyqtSignal(object, str, str, int)

    def __init__(self, lines, make_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Element associations')
        self.make_map = make_map
        self.a = self.b = None
        self.binning = None
        self.names = ('', '')
        self.cmb_x, self.cmb_y = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for c in (self.cmb_x, self.cmb_y):
            c.setEditable(True)
        self.set_lines(lines)
        self.chk_lx = QtWidgets.QCheckBox('log')
        self.chk_ly = QtWidgets.QCheckBox('log')
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(16, 512)
        self.spin_bins.setValue(128)
        self.cmb_op = QtWidgets.QComboBox()
        for lab, key in ASSOC_OPS:
            self.cmb_op.addItem(lab, key)
        b_plot = QtWidgets.QPushButton('Plot')
        b_plot.clicked.connect(self.plot)
        top = QtWidgets.QHBoxLayout()
        for w, lab in ((self.cmb_x, 'x'), (self.chk_lx, None),
                       (self.cmb_y, 'y'), (self.chk_ly, None),
                       (self.spin_bins, 'bins'), (b_plot, None)):
            if lab:
                top.addWidget(QtWidgets.QLabel(lab))
            top.addWidget(w)
        top.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel('a region drawn on the plot will'))
        row2.addWidget(self.cmb_op)
        row2.addStretch(1)
        self.fig = Figure(figsize=(5.5, 5), dpi=100, facecolor='white')
        self.canvas = _Canvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.info = QtWidgets.QLabel('Choose two maps and press Plot. Then '
                                     'click round a cluster (click the first '
                                     'corner to close) to select its pixels.')
        self.info.setWordWrap(True)
        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(top)
        v.addLayout(row2)
        v.addWidget(self.canvas, 1)
        v.addWidget(self.info)
        self._poly = None
        self.resize(640, 640)

    def closeEvent(self, ev):
        # a draw queued now would run after the canvas is deleted (BUGS.md
        # "A deferred draw_idle fires after a window closes")
        self.canvas.cancel_draws()
        super().closeEvent(ev)

    def reset(self):
        """Forget the plotted maps (another file was opened): a region drawn
        on the old plot must not select the new file's pixels."""
        self.a = self.b = None
        self.binning = None
        if self._poly is not None:
            self._poly.disconnect_events()
            self._poly = None
        self.ax.clear()
        self.canvas.draw_idle()
        self.info.setText('New data: press Plot.')

    def set_lines(self, lines):
        for c, i in ((self.cmb_x, 0), (self.cmb_y, 1)):
            cur = c.currentText()
            c.clear()
            c.addItems(list(lines))
            if cur:
                c.setCurrentText(cur)
            elif c.count() > i:
                c.setCurrentIndex(i)

    def log(self):
        return (self.chk_lx.isChecked(), self.chk_ly.isChecked())

    def plot(self):
        nx, ny = self.cmb_x.currentText().strip(), self.cmb_y.currentText().strip()
        try:
            a, tx, ux, bx = self.make_map(nx)
            b, ty, uy, by = self.make_map(ny)
            if bx != by:
                raise ValueError('the two maps differ in binning')
            H, xe, ye, lims = AS.histogram(a, b, bins=self.spin_bins.value(),
                                           log=self.log())
        except Exception as ex:
            self.info.setText('Cannot plot: %s' % ex)
            return False
        # the binning travels with the maps it made, so a mask drawn later
        # lands where these pixels are whatever the Maps tab is set to now
        self.a, self.b, self.names, self.binning = a, b, (nx, ny), bx
        self.ax.clear()
        Hm = np.ma.masked_equal(H.T, 0)
        self.ax.imshow(Hm, origin='lower', aspect='auto',
                       extent=(xe[0], xe[-1], ye[0], ye[-1]),
                       cmap='viridis', interpolation='nearest',
                       norm=mcolors.LogNorm(1, max(H.max(), 2)))
        lx, ly = self.log()
        self.ax.set_xlabel('%s%s (%s)' % ('log10 ' if lx else '', tx, ux))
        self.ax.set_ylabel('%s%s (%s)' % ('log10 ' if ly else '', ty, uy))
        self.ax.set_title('pixels per bin (log colour); %d pixels'
                          % a.size, fontsize=9)
        if self._poly is not None:
            self._poly.disconnect_events()
        self._poly = PolygonSelector(self.ax, self._on_poly, useblit=True,
                                     props=dict(color='#A8323F', lw=1.4))
        self.canvas.draw_idle()
        self.info.setText('%s vs %s: Pearson r = %.3f over %d pixels (as '
                          'plotted: negatives at 0). Click round a cluster to '
                          'select its pixels.'
                          % (ny, nx, AS.pearson(a, b), a.size))
        return True

    def _on_poly(self, verts):
        self.select(verts)
        poly = self._poly
        QtCore.QTimer.singleShot(0, lambda: poly.clear())

    def select(self, verts):
        """Pixels whose point is inside verts (plot coordinates) -> mask."""
        if self.a is None or len(verts) < 3:
            return None
        m = AS.select(self.a, self.b, verts, log=self.log())
        if not m.any():
            self.info.setText('No pixel lies inside that region.')
            return m
        name = 'assoc %s-%s' % self.names
        self.info.setText('%d pixels selected (%s).'
                          % (int(m.sum()), self.cmb_op.currentText()))
        self.canvas.draw_idle()
        self.maskSelected.emit(m, name, self.cmb_op.currentData(),
                               self.binning)
        return m
