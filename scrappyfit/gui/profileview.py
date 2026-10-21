"""Draw a traverse on an element map and read concentrations along it.

Drag on the map to place a line. The profile appears immediately, for every
selected element at once, because the interesting quantity is almost never
one element - it is how two of them move relative to each other across a
boundary.

Three modes:
    line      drag a traverse
    polyline  click vertices, double-click to finish, for a path that
              follows a feature rather than cutting straight across it
    radial    click a centre; mean in annuli about it

The width control matters more than it looks. A one-pixel traverse over a
DA map is dominated by counting noise; widening it to 5 or 9 pixels averages
across the path and turns an unreadable profile into a usable one, at the
cost of blurring anything not perpendicular to the traverse. The shaded band
on the plot is the spread ACROSS the width, so when it swamps the signal the
traverse is badly placed and the plot says so.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from ..analysis.profiles import (line_profile, polyline_profile,
                                 profile_table, radial_profile, region_stats,
                                 to_csv)

INK = '#151A22'
SERIES = ['#1F4FD8', '#E07A00', '#00873E', '#7B3FA8', '#C4187A',
          '#0F8FA8', '#B03030', '#8A6D00', '#3B5BA5', '#A8541F']


class ProfileDialog(QtWidgets.QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.maps = {}
        self.points = []
        self.mode = 'line'
        self._drag_from = None
        self.setWindowTitle('Concentration profiles')
        self.setModal(False)
        self.resize(1180, 720)

        root = QtWidgets.QHBoxLayout(self)

        # -- left: the map ------------------------------------------------
        left = QtWidgets.QVBoxLayout()
        self.fig_map = Figure(figsize=(5.2, 5.2), dpi=100, facecolor='white')
        self.cv_map = FigureCanvasQTAgg(self.fig_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.fig_map.subplots_adjust(0.06, 0.06, 0.98, 0.94)
        self.cv_map.mpl_connect('button_press_event', self._press)
        self.cv_map.mpl_connect('motion_notify_event', self._motion)
        self.cv_map.mpl_connect('button_release_event', self._release)
        left.addWidget(self.cv_map, 1)

        bar = QtWidgets.QHBoxLayout()
        self.cmb_show = QtWidgets.QComboBox()
        self.cmb_show.currentIndexChanged.connect(self.draw_map)
        bar.addWidget(QtWidgets.QLabel('show'))
        bar.addWidget(self.cmb_show, 1)
        for name in ('line', 'polyline', 'radial'):
            b = QtWidgets.QRadioButton(name)
            b.setChecked(name == 'line')
            b.toggled.connect(lambda on, n=name: on and self._set_mode(n))
            bar.addWidget(b)
        left.addLayout(bar)
        root.addLayout(left, 1)

        # -- right: the profile -------------------------------------------
        right = QtWidgets.QVBoxLayout()
        self.fig_pr = Figure(figsize=(6.0, 4.2), dpi=100, facecolor='white')
        self.cv_pr = FigureCanvasQTAgg(self.fig_pr)
        self.ax_pr = self.fig_pr.add_subplot(111)
        right.addWidget(self.cv_pr, 1)

        form = QtWidgets.QHBoxLayout()
        self.sp_width = QtWidgets.QSpinBox()
        self.sp_width.setRange(1, 199)
        self.sp_width.setValue(5)
        self.sp_width.setToolTip(
            'Pixels averaged ACROSS the traverse. 1 is a single-pixel cut and '
            'is usually too noisy on a DA map; widening trades spatial '
            'resolution for precision. The shaded band is the spread across '
            'this width - if it swamps the signal, the traverse is not '
            'perpendicular to the feature.')
        self.ed_px = QtWidgets.QLineEdit('1.0')
        self.ed_px.setToolTip('Microns per pixel, so distance comes out in '
                              'microns rather than pixels.')
        self.chk_med = QtWidgets.QCheckBox('median')
        self.chk_med.setToolTip('Median across the width instead of the mean. '
                                'Resists a hot or dead pixel.')
        for w in (self.sp_width, self.ed_px, self.chk_med):
            try:
                w.valueChanged.connect(self.recompute)
            except AttributeError:
                try:
                    w.textChanged.connect(self.recompute)
                except AttributeError:
                    w.toggled.connect(self.recompute)
        form.addWidget(QtWidgets.QLabel('width px'))
        form.addWidget(self.sp_width)
        form.addWidget(QtWidgets.QLabel('um/px'))
        form.addWidget(self.ed_px)
        form.addWidget(self.chk_med)
        form.addStretch(1)
        right.addLayout(form)

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumHeight(150)
        right.addWidget(self.out)

        btns = QtWidgets.QHBoxLayout()
        b_csv = QtWidgets.QPushButton('Export CSV...')
        b_csv.clicked.connect(self.on_csv)
        b_clear = QtWidgets.QPushButton('Clear')
        b_clear.clicked.connect(self.on_clear)
        b_close = QtWidgets.QPushButton('Close')
        b_close.clicked.connect(self.hide)
        btns.addWidget(b_csv)
        btns.addWidget(b_clear)
        btns.addStretch(1)
        btns.addWidget(b_close)
        right.addLayout(btns)
        root.addLayout(right, 1)

        self._last = None

    # -- data -------------------------------------------------------------

    def set_maps(self, maps):
        """maps: {element: 2-D array} in weight percent."""
        self.maps = {k: np.asarray(v, float) for k, v in maps.items()
                     if np.ndim(v) == 2}
        cur = self.cmb_show.currentText()
        self.cmb_show.blockSignals(True)
        self.cmb_show.clear()
        self.cmb_show.addItems(sorted(self.maps))
        if cur in self.maps:
            self.cmb_show.setCurrentText(cur)
        self.cmb_show.blockSignals(False)
        self.draw_map()
        self.recompute()

    def _set_mode(self, name):
        self.mode = name
        self.points = []
        self.draw_map()
        self.out.setPlainText({
            'line': 'Drag across the feature you want a traverse through.',
            'polyline': 'Click each vertex; double-click to finish.',
            'radial': 'Click the centre. Mean in annuli about it - this is '
                      'the one to use for anything concentric, since it '
                      'averages the whole ring instead of one cut.',
        }[name])

    # -- interaction -------------------------------------------------------

    def _xy(self, ev):
        return (float(ev.xdata), float(ev.ydata))

    def _press(self, ev):
        if ev.inaxes is not self.ax_map or ev.xdata is None:
            return
        p = self._xy(ev)
        if self.mode == 'line':
            self._drag_from = p
            self.points = [p, p]
        elif self.mode == 'radial':
            self.points = [p]
            self.recompute()
        else:
            if ev.dblclick:
                self.recompute()
                return
            self.points.append(p)
            self.recompute()
        self.draw_map()

    def _motion(self, ev):
        if (self.mode != 'line' or self._drag_from is None
                or ev.inaxes is not self.ax_map or ev.xdata is None):
            return
        self.points = [self._drag_from, self._xy(ev)]
        self.draw_map()

    def _release(self, ev):
        if self.mode != 'line' or self._drag_from is None:
            return
        if ev.inaxes is self.ax_map and ev.xdata is not None:
            self.points = [self._drag_from, self._xy(ev)]
        self._drag_from = None
        self.draw_map()
        self.recompute()

    def on_clear(self):
        self.points = []
        self._last = None
        self.ax_pr.clear()
        self.cv_pr.draw_idle()
        self.draw_map()

    # -- drawing -----------------------------------------------------------

    def draw_map(self):
        self.ax_map.clear()
        nm = self.cmb_show.currentText()
        if nm and nm in self.maps:
            m = self.maps[nm]
            # 1-99 percentile, so one hot pixel cannot flatten the image
            finite = m[np.isfinite(m)]
            if finite.size:
                lo, hi = np.percentile(finite, (1, 99))
                if hi <= lo:
                    lo, hi = finite.min(), max(finite.max(), finite.min() + 1e-9)
            else:
                lo, hi = 0, 1
            self.ax_map.imshow(m, origin='upper', cmap='viridis',
                               vmin=lo, vmax=hi, interpolation='nearest')
            self.ax_map.set_title('%s   (1-99%% scaling)' % nm, fontsize=10)
        if len(self.points) >= 2 and self.mode in ('line', 'polyline'):
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax_map.plot(xs, ys, '-o', color='#FF3B30', lw=1.6, ms=4)
        elif len(self.points) == 1:
            self.ax_map.plot(self.points[0][0], self.points[0][1], '+',
                             color='#FF3B30', ms=14, mew=2)
        self.cv_map.draw_idle()

    def recompute(self):
        if not self.maps or not self.points:
            return
        try:
            px = float(self.ed_px.text())
        except ValueError:
            px = 1.0
        unit = 'um' if px != 1.0 else 'pixels'
        w = self.sp_width.value()
        red = 'median' if self.chk_med.isChecked() else 'mean'
        names = sorted(self.maps)

        self.ax_pr.clear()
        rows = []
        try:
            if self.mode == 'radial' and len(self.points) == 1:
                series = {}
                dist = None
                for nm in names:
                    r, mean, std, cnt = radial_profile(
                        self.maps[nm], centre=self.points[0], pixel_size=px)
                    dist = r if dist is None else dist
                    series[nm] = mean
                rows.append('radial about (%.1f, %.1f)' % self.points[0])
            elif self.mode == 'polyline' and len(self.points) >= 2:
                series, dist = {}, None
                for nm in names:
                    d, v, _ = polyline_profile(
                        self.maps[nm], self.points, width=w, pixel_size=px,
                        reduce=red)
                    dist = d if dist is None else dist
                    series[nm] = v
                rows.append('polyline, %d vertices' % len(self.points))
            elif len(self.points) >= 2:
                dist, series = profile_table(
                    self.maps, self.points[0], self.points[1], width=w,
                    pixel_size=px, reduce=red)
                _, _, spread = line_profile(
                    self.maps[names[0]], self.points[0], self.points[1],
                    width=w, pixel_size=px, reduce=red)
                rows.append('line (%.1f, %.1f) -> (%.1f, %.1f), length %.1f %s'
                            % (self.points[0][0], self.points[0][1],
                               self.points[1][0], self.points[1][1],
                               dist[-1], unit))
            else:
                return
        except Exception as ex:
            self.out.setPlainText('profile failed: %s' % ex)
            return

        for i, nm in enumerate(names):
            if nm not in series:
                continue
            self.ax_pr.plot(dist, series[nm], lw=1.4,
                            color=SERIES[i % len(SERIES)], label=nm)
        self.ax_pr.set_xlabel('distance, %s' % unit)
        self.ax_pr.set_ylabel('concentration, wt%')
        self.ax_pr.legend(fontsize=7, ncol=4, frameon=False)
        self.ax_pr.grid(alpha=0.25)
        self.fig_pr.tight_layout()
        self.cv_pr.draw_idle()

        rows.append('width %d px, %s across the path' % (w, red))
        for nm in names:
            if nm in series and np.isfinite(series[nm]).any():
                v = series[nm]
                rows.append('   %-5s mean %8.3f  min %8.3f  max %8.3f wt%%'
                            % (nm, np.nanmean(v), np.nanmin(v), np.nanmax(v)))
        self.out.setPlainText('\n'.join(rows))
        self._last = (dist, series, unit)

    def on_csv(self):
        if not self._last:
            self.out.setPlainText('Nothing to export - draw a traverse first.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save profile', 'profile.csv', 'CSV (*.csv)')
        if not path:
            return
        dist, series, unit = self._last
        to_csv(path, dist, series, unit=unit)
        self.out.appendPlainText('written: %s' % path)
