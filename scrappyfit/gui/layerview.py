"""Layer stack diagram, with an escape calculator.

Why this is worth a window
--------------------------
In a layered sample the question that decides everything is not "what is in
each layer" but "can I see it". An X-ray generated in a buried layer has to
climb back out through everything above it, and at light-element energies that
is usually hopeless: oxygen Ka through half a micron of silver is gone. A
composition table cannot show that. A picture of the stack, with a number
attached to each layer saying what fraction escapes, can.

The calculation is deliberately simple and exact:

    transmission = exp( -sum_over_layers_above( mu_i * rho_i * t_i ) / cos(theta) )

with mu from whichever attenuation dataset the session is using. No geometry
approximations beyond the take-off angle, because there are none to make - the
photon travels in a straight line and the layers are flat.

Also reported is the mean free path in the layer the photon was born in, since
that says how much of the layer you are actually sampling. When the MFP is far
shorter than the layer, a bulk concentration for that layer is being inferred
from its top surface.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

INK = '#151A22'
PALETTE = ['#1F4FD8', '#E07A00', '#00873E', '#7B3FA8', '#C4187A',
           '#0F8FA8', '#B03030', '#8A6D00']


class StackCanvas(FigureCanvasQTAgg):
    layerPicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4.4, 6.0), dpi=100, facecolor='white')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.03, right=0.72, top=0.94, bottom=0.05)
        self._rects = []
        self.mpl_connect('button_press_event', self._click)

    def _click(self, ev):
        if ev.inaxes is not self.ax or ev.ydata is None:
            return
        for i, (y0, y1) in enumerate(self._rects):
            if y0 <= ev.ydata <= y1:
                self.layerPicked.emit(i)
                return

    def draw_stack(self, layers, selected=None, escape=None):
        """layers: [dict(name, formula, thick, density, microns)] surface first.

        Thickness is drawn on a log scale. A stack that spans 0.002 to 2200 um
        - which the perovskite cell does - is unreadable linearly: the glass
        would be the whole picture and the absorber an invisible line.
        """
        self.ax.clear()
        self._rects = []
        if not layers:
            self.ax.text(0.5, 0.5, 'no layers', ha='center', transform=self.ax.transAxes)
            self.draw_idle()
            return
        t = np.array([max(float(L.get('thick', 1.0)), 1e-6) for L in layers])
        h = np.log10(t) - np.log10(t.min()) + 1.0
        h = h / h.sum()
        y = 1.0
        for i, (L, hh) in enumerate(zip(layers, h)):
            y0, y1 = y - hh, y
            self._rects.append((y0, y1))
            col = PALETTE[i % len(PALETTE)]
            self.ax.add_patch(plt_rect(0.02, y0, 0.96, hh, col,
                                       selected == i))
            unit = 'um' if L.get('microns') else 'mg/cm2'
            label = '%s\n%.4g %s' % (L.get('formula') or L.get('name', 'layer'),
                                     L.get('thick', 0), unit)
            self.ax.text(0.5, (y0 + y1) / 2, label, ha='center', va='center',
                         fontsize=8, color='white' if hh > 0.06 else col,
                         zorder=3)
            if escape is not None and i < len(escape) and escape[i] is not None:
                self.ax.text(1.03, (y0 + y1) / 2, '%.2g%%' % (100 * escape[i]),
                             ha='left', va='center', fontsize=9,
                             color=INK, fontweight='bold')
            y = y0
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.ax.set_title('surface at top     escape %', fontsize=9, color=INK)
        self.draw_idle()


def plt_rect(x, y, w, h, colour, highlight):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=colour,
                     edgecolor='#111' if highlight else 'white',
                     lw=2.5 if highlight else 1.0, alpha=0.9, zorder=2)


class LayerStackDialog(QtWidgets.QDialog):
    """The stack, plus what escapes from each layer at a chosen energy."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.layers = []
        self.selected = 0
        self.setWindowTitle('Layer stack and escape')
        self.setModal(False)
        self.resize(900, 640)

        h = QtWidgets.QHBoxLayout(self)
        self.canvas = StackCanvas()
        self.canvas.layerPicked.connect(self.on_pick)
        h.addWidget(self.canvas, 1)

        side = QtWidgets.QVBoxLayout()
        b = QtWidgets.QPushButton('Load a .yield file...')
        b.clicked.connect(self.on_load_yield)
        side.addWidget(b)
        b2 = QtWidgets.QPushButton('Use the sample model')
        b2.clicked.connect(self.from_sample_model)
        side.addWidget(b2)

        f = QtWidgets.QFormLayout()
        self.ed_line = QtWidgets.QLineEdit('O Ka')
        self.ed_line.setToolTip('An element and line ("Si Ka", "Pb Ma"), or '
                                'a bare energy in keV')
        self.ed_line.returnPressed.connect(self.recompute)
        self.ed_theta = QtWidgets.QLineEdit('135')
        self.ed_theta.returnPressed.connect(self.recompute)
        f.addRow('X-ray of interest', self.ed_line)
        f.addRow('take-off angle deg', self.ed_theta)
        side.addLayout(f)
        bc = QtWidgets.QPushButton('Compute escape')
        bc.clicked.connect(self.recompute)
        side.addWidget(bc)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumWidth(360)
        side.addWidget(self.out, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.hide)
        side.addWidget(bb)
        h.addLayout(side)

    # -- sources --------------------------------------------------------

    def on_load_yield(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open a GeoPIXE yield file', '', 'Yield (*.yield);;All (*)')
        if not path:
            return
        from ..io.gpyield import read_yield
        try:
            Y = read_yield(path)[0]
        except Exception as ex:
            self.out.setPlainText('could not read: %s' % ex)
            return
        layers = []
        for i, L in enumerate(Y['layers']):
            layers.append(dict(
                name=L.get('name', 'layer %d' % i),
                formula=(Y.get('formula') or [''] * len(Y['layers']))[i],
                thick=(Y.get('thick') or [L['thick']])[i],
                density=(Y.get('density') or [0])[i],
                microns=(Y.get('microns') or [0])[i],
                zlist=L['Z'], wfrac=L['F']))
        self.layers = layers
        self.selected = 0
        self.ed_theta.setText('%.0f' % Y.get('theta', 135.0))
        self.recompute()

    def from_sample_model(self):
        m = getattr(self.parent(), 'sample', None)
        if m is None:
            return
        mat = m.matrix()
        if not mat:
            self.out.setPlainText(
                'The sample model is set to bootstrap from the fit, which '
                'gives one layer with no thickness. Type a composition, or '
                'load a .yield file with a real stack.')
            return
        db = self.session.db
        self.layers = [dict(name='matrix',
                            formula=''.join('(%s)%.3g' % (db.sym[z], w)
                                            for z, w in mat.items()),
                            thick=m.thickness() or 1e4, density=0, microns=0,
                            zlist=list(mat), wfrac=list(mat.values()))]
        self.selected = 0
        self.ed_theta.setText('%.0f' % m.theta())
        self.recompute()

    # -- calculation ----------------------------------------------------

    def _energy(self):
        t = self.ed_line.text().strip()
        try:
            return float(t), t
        except ValueError:
            pass
        parts = t.replace(',', ' ').split()
        db = self.session.db
        Z = db.z.get(parts[0].lower()) if parts else None
        if Z is None:
            return None, t
        shell = 1
        if len(parts) > 1:
            u = parts[1].upper()
            shell = 2 if u.startswith('L') else (3 if u.startswith('M') else 1)
        e = db.line_energy(Z, shell)
        return (e if e > 0 else None), '%s %s' % (db.sym[Z],
                                                  {1: 'Ka', 2: 'La', 3: 'Ma'}[shell])

    def on_pick(self, i):
        self.selected = i
        self.recompute()

    def recompute(self):
        if not self.layers:
            self.out.setPlainText('Load a .yield file, or use the sample model.')
            return
        E, label = self._energy()
        if not E:
            self.out.setPlainText('Could not interpret "%s". Try "Si Ka" or a '
                                  'number in keV.' % self.ed_line.text())
            return
        try:
            theta = float(self.ed_theta.text())
        except ValueError:
            theta = 135.0
        cos_out = abs(np.cos(np.radians(180.0 - theta if theta > 90 else theta)))
        cos_out = max(cos_out, 1e-3)
        db = self.session.db
        mac = self.session.options.mac

        esc, rows = [], []
        rows.append('%s at %.4f keV, take-off %.0f deg' % (label, E, theta))
        rows.append('MAC dataset: %s' % mac)
        rows.append('')
        cum = 0.0
        per = []
        for L in self.layers:
            mu = db.mu_compound(L['zlist'], L['wfrac'], E, mac)
            if mu is None:
                per.append(None)
                continue
            areal = self._areal(L)
            per.append(mu * 1e-3 * areal)          # cm2/g * mg/cm2 -> unitless
        for i, L in enumerate(self.layers):
            above = [p for p in per[:i] if p is not None]
            if per[i] is None:
                esc.append(None)
                continue
            tau_above = sum(above) / cos_out
            # average over depth within the layer itself
            tau_self = per[i] / cos_out
            if tau_self > 1e-9:
                selfabs = (1.0 - np.exp(-tau_self)) / tau_self
            else:
                selfabs = 1.0
            esc.append(float(np.exp(-tau_above) * selfabs))
        for i, (L, e) in enumerate(zip(self.layers, esc)):
            mark = '>' if i == self.selected else ' '
            if e is None:
                rows.append('%s %-22s  MAC out of range' % (mark, L['formula'][:22]))
                continue
            mu = db.mu_compound(L['zlist'], L['wfrac'], E, mac) or 0.0
            mfp_mg = cos_out / (mu * 1e-3) if mu > 0 else float('inf')
            rows.append('%s %-22s escape %7.3f%%   absorbed %6.2f%%'
                        % (mark, (L['formula'] or L['name'])[:22],
                           100 * e, 100 * (1 - e)))
            rows.append('    MFP in this layer %.4g mg/cm2  vs layer %.4g'
                        % (mfp_mg, self._areal(L)))
            if self._areal(L) > 3 * mfp_mg:
                rows.append('    NOTE only the top of this layer is sampled - '
                            'a bulk number for it is inferred from its surface')
        rows.append('')
        rows.append('escape = exp(-sum of mu.rho.t above / cos theta), times')
        rows.append('the average over depth within the layer itself.')
        self.out.setPlainText('\n'.join(rows))
        self.canvas.draw_stack(self.layers, self.selected, esc)

    @staticmethod
    def _areal(L):
        """Areal density in mg/cm2, whatever the layer was specified in."""
        t = float(L.get('thick', 0.0))
        if L.get('microns'):
            return t * float(L.get('density', 0.0)) / 10.0
        return t
