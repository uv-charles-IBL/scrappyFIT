"""scrappyFIT main window.

The workflow this is built around, which is the one GeoPIXE supports plus the
two things it makes hard:

    1. open a file          LMF list mode, .dam, .spec text, or two columns
    2. set the calibration  gain and offset, checkable against known lines
    3. choose elements      K lines, and L or M where the element needs them
    4. fit                  peak shape + detector response + SNIP background
    5. look at the residual the panel under the spectrum is where lies show up
    6. map an element       needs list mode
    7. MASK A REGION        drag on the map, or threshold it, then refit -
                            this is the step that turns a bulk average into a
                            measurement of an actual phase
    8. export               one folder holding spectrum, model, components,
                            residual, areas, mask and every option used

Steps 6-7 are the reason this exists. A heterogeneous sample's bulk spectrum
describes nowhere in it, and the fitted numbers from one are an average over
regions that may share no mineral.
"""

import copy
import os
import pathlib
import re
import sys

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import __version__, config
from ..analysis import masking as MSK
from ..batch import run_batch, summarise
from ..io.gpda_write import from_session as write_dam_from_session
from ..io.gpyield_write import from_session as write_yield_from_session
from ..io.live import LiveLMF, refresh_session
from ..physics import artefacts, lineid
from ..session import FitOptions, Session
from .canvases import (LEGEND_MODES, MAP_CMAPS, MAP_PCT, MAP_SCALES,
                       MapCanvas, SpectrumCanvas, toolbar_for)
from .dialogs import ElementDialog, SampleModelDialog
from .layerview import LayerStackDialog
from .observe import log_event, user_action
from .profileview import ProfileDialog
from .worker import Job, JobRunner

# The element lists are not a fixed menu. They are rebuilt from the working
# energy range, so every element with a line you could actually detect is
# there and nothing you could detect is missing. A fixed list is worse than
# useless on an unknown sample: it silently rules out whatever the author did
# not anticipate.
SHELL_NAMES = {1: 'K', 2: 'L', 3: 'M'}

# Sensible starting selections, not limits on what is available.
PRESET_LIGHT = ['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                'K', 'Ca']
PRESET_SILICATE = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                   'K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe']


def elements_in_range(db, shell, e_low, e_high, min_intensity=0.05):
    """Every element whose given shell puts a usable line inside the window.

    'Usable' means a line carrying at least min_intensity of that shell's
    emission - an element whose only in-range line is a 1% satellite cannot
    be identified from it and would only add a free parameter.
    """
    out = []
    for Z in range(3, 93):
        try:
            lines = db.line_list(Z, shell)
        except Exception:
            continue
        if any(e_low <= e <= e_high and i >= min_intensity for e, i in lines):
            out.append((Z, db.sym[Z]))
    return out


# Handlers that change the Session. While a fit or batch owns the session they
# are refused (their controls are disabled too; this catches shortcuts).
MUTATING = {
    'on_open', 'open_path', 'on_adc_changed', 'on_set_db', 'on_cal_changed',
    'on_range_changed', 'on_fit', 'on_quantify', 'on_batch', 'on_region',
    'on_threshold', 'on_polygon', 'on_clear_mask', 'on_load_dam', 'on_eff_changed',
    'on_load_eff', 'on_load_detector', 'on_attach_live', 'on_morph',
    'on_lines_file', 'on_accept_suggested'}

# The live estimate waits this long after the last change before it runs, so
# a burst of clicks costs one estimate, not one each.
PREVIEW_DELAY_MS = 150

# The window's size when it opens; the left panel must fit at this size.
DEFAULT_SIZE = (1500, 950)


def export_map_csv(path, data, session, meta):
    """Write a shown map, and the session mask on its grid, as CSV.

    Row 0 is y = 0, the bottom row of the displayed map (origin='lower').
    The mask file holds the fraction of each displayed pixel's raster pixels
    inside the mask, so a mask made at another binning is not rounded.
    Returns the paths written.
    """
    a, b = meta['cal']
    head = ('scrappyFIT map: %s, element %s, %s, binning %d (%dx%d px); '
            'cal %.10g keV/ch, %.10g keV; row 0 = bottom (y = 0)'
            % (meta['label'], meta['element'], meta['kind'], meta['binning'],
               data.shape[1], data.shape[0], a, b))
    np.savetxt(path, data, delimiter=',', fmt='%.8g', header=head)
    out = [path]
    m = session.mask
    if m is not None:
        ny, nx = data.shape
        fy, fx = m.shape[0] // ny, m.shape[1] // nx
        frac = m[:ny * fy, :nx * fx].reshape(ny, fy, nx, fx).mean(axis=(1, 3))
        mpath = os.path.splitext(path)[0] + '_mask.csv'
        np.savetxt(mpath, frac, delimiter=',', fmt='%.4g',
                   header=head + '; mask %s, fraction of each pixel'
                   % (session.mask_name or 'mask'))
        out.append(mpath)
    return out


class _VScrollArea(QtWidgets.QScrollArea):
    """Scrolls vertically only, and is never narrower than what it holds.

    The left panel used a fixed 340 px minimum. The splitter held it there
    (the centre wants the rest of the window), the content needed more, and a
    horizontal scrollbar cut the right edge off every box and wrapped note;
    larger fonts (125-150 % Windows scaling) made it worse. Taking the minimum
    from the content means a bigger font widens the panel instead.
    """

    def __init__(self, content):
        super().__init__()
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.widget() and ev.type() == QtCore.QEvent.LayoutRequest:
            self.updateGeometry()        # content's minimum may have changed
        return super().eventFilter(obj, ev)

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        need = (self.widget().minimumSizeHint().width()
                + self.verticalScrollBar().sizeHint().width()
                + 2 * self.frameWidth())
        return QtCore.QSize(max(s.width(), need), s.height())


def _wrappable_path(p):
    """A path a word-wrapped QLabel can break after each separator: one long
    unbreakable path set the whole left panel's minimum width."""
    zwsp = chr(0x200B)                   # zero-width space
    return str(p).replace(chr(92), chr(92) + zwsp).replace('/', '/' + zwsp)


class MainWindow(QtWidgets.QMainWindow):
    """The scrappyFIT window.

    Drawing follows one rule: anything that changes what the plot should
    show calls mark_changed() (or emits sessionChanged for a pure view
    change), and refresh_spectrum() is connected to sessionChanged and
    previewUpdated and called from nowhere else. Slow work - fits, batches,
    the live estimate - runs on self.jobs, off the GUI thread.
    """

    sessionChanged = QtCore.pyqtSignal()
    previewUpdated = QtCore.pyqtSignal()
    yieldChanged = QtCore.pyqtSignal()   # session.yield_path changed

    def __init__(self):
        super().__init__()
        self.session = Session()
        self.jobs = JobRunner(self)
        self._gen = 0                  # bumped by every change to the session
        self.preview = None            # last live estimate (FitResult)
        self._preview_gen = -1         # the generation it was computed for
        self.preview_runs = 0
        self._preview_bk = None        # (inputs, SNIP curve) for estimates
        self._fit_sel = None           # selection the current fit was made with
        self.last_shape = None         # last fit's converged ShapePars, for the hold
        self.batch_done = 0
        self.last_batch = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._launch_preview)
        self.sessionChanged.connect(self._on_session_changed)
        self.previewUpdated.connect(self.refresh_spectrum)
        self.jobs.progress.connect(self._on_job_progress)
        self.setWindowTitle('scrappyFIT %s' % __version__)
        self.resize(*DEFAULT_SIZE)
        self.elements = ElementDialog(self.session.db, self)
        self.elements.selectionChanged.connect(self._on_elements_changed)
        self.sample = SampleModelDialog(self.session.db, self)
        self.sample.yieldPicked.connect(
            lambda path: self.set_session_yield(path, 'loaded'))
        self.sample.yieldSetPicked.connect(self.set_session_yield_set)
        self.sample.regionPicked.connect(self.set_session_yield_region)
        self.sample.searchPathChanged.connect(self._on_yield_search_changed)
        self.yieldChanged.connect(self._on_yield_changed)
        self._build()
        self._refresh_db_label()
        self.rebuild_element_lists()

    # ------------------------------------------------- change and redraw

    def mark_changed(self):
        """The session changed: redraw, and refresh the live estimate."""
        self._gen += 1
        self.sessionChanged.emit()

    def _on_session_changed(self):
        self.refresh_spectrum()
        self._schedule_preview()

    def set_session_yield(self, path, source):
        """Make path the sample's .yield everywhere in the window."""
        self.session.set_yield(path, source)
        self.yieldChanged.emit()

    def set_session_yield_set(self, paths, source='chosen'):
        """Make these .yield files the sample's yield set, and remember them
        for this data file."""
        from ..yieldset import YieldSet
        s = self.session
        ys = YieldSet(paths)
        s.set_yield_set(ys, s.yield_region, source)
        s.remember_yields()
        self.say('Yield files for %s: %s'
                 % (os.path.basename(s.path or '') or 'this sample',
                    ys.describe(s.yield_region)))
        self.yieldChanged.emit()

    def set_session_yield_region(self, region):
        s = self.session
        if s.yield_set is None:
            return
        s.set_yield_set(s.yield_set, region, getattr(
            s.yield_set, 'source', '') or 'chosen')
        s.remember_yields()
        self.yieldChanged.emit()

    def _on_yield_search_changed(self):
        """Search the new path now, unless the user already chose files."""
        from ..yieldset import find_yield_set
        s = self.session
        if not s.path or (s.yield_set is not None and getattr(
                s.yield_set, 'source', '') != 'found on the yield search '
                'path'):
            return
        ys, why = find_yield_set(s.path)
        if ys is not None:
            s.set_yield_set(ys, None, why)
        else:
            s.yield_search = why
        self.yieldChanged.emit()

    def _on_yield_changed(self):
        s = self.session
        self.sample.show_yield(s.yield_path, s.yield_source)
        self.sample.show_yield_set(
            s.yield_set, s.yield_region,
            getattr(s.yield_set, 'source', '') if s.yield_set else '',
            s.yield_search)
        if hasattr(self, 'lbl_sm'):
            self.lbl_sm.setText(self._sample_summary())

    @property
    def preview_pending(self):
        return self._preview_timer.isActive() or self.jobs.running('preview')

    def _fit_is_current(self):
        s = self.session
        return s.fit is not None and self._fit_sel == self.selected_elements()

    def _schedule_preview(self):
        s = self.session
        if not self.chk_preview.isChecked() or s.spectrum is None:
            return
        if not self.selected_elements():
            # nothing to estimate - and nothing stale left on screen
            if self.preview is not None:
                self.preview = None
                self.previewUpdated.emit()
            return
        if self._fit_is_current() or self._preview_gen == self._gen:
            return
        if self.jobs.running_heavy():
            return
        self._preview_timer.start(PREVIEW_DELAY_MS)

    def _launch_preview(self):
        """The GeoPIXE-style overlay: one model evaluation with peak-height
        areas (Session.preview), run on a copy of the session so the GUI can
        keep changing the original meanwhile."""
        s = self.session
        if s.spectrum is None or self.jobs.running_heavy():
            return
        gen, sel = self._gen, list(self.selected_elements())
        if not sel:
            return
        snap = copy.copy(s)
        snap.options = copy.deepcopy(s.options)
        # The estimate's SNIP curve is cached here, never written into the
        # session, so what a fit uses for its background is exactly what it
        # used before the live estimate existed.
        key = self._bk_key(snap)
        if s._bk is not None:
            snap._bk = s._bk
        elif self._preview_bk is not None and self._same_bk_inputs(
                self._preview_bk[0], key):
            snap._bk = self._preview_bk[1]
        self.preview_runs += 1

        def done(r):
            self._preview_bk = (key, r.background)
            if gen != self._gen:
                return                  # superseded; a newer one is queued
            self.preview, self._preview_gen = r, gen
            self.previewUpdated.emit()

        def fail(tb):
            self.say('live estimate failed (the fit itself is unaffected): '
                     + tb.strip().splitlines()[-1])

        self.jobs.submit(Job('preview', lambda ctx: snap.preview(
            sel, should_stop=ctx.should_stop), on_done=done, on_fail=fail,
            tag=gen))

    @staticmethod
    def _bk_key(s):
        """Everything Session.background depends on: the objects themselves
        (compared by identity, and held so their ids cannot be reused) and
        the option values."""
        o = s.options
        det = s.detector or {}
        layers = tuple((lay.get('Z'), tuple(lay.get('F') or ()),
                        lay.get('thick'))
                       for lay in det.get('external') or [])
        return ((s.spectrum, s.efficiency, s.detector, s.db),
                (tuple(s.cal), o.e_low, o.e_high, o.mac, o.snip_passes,
                 o.snip_boost, o.snip_trim_seb, o.snip_e_trans,
                 det.get('w0'), det.get('w1'),
                 # load_filter appends to the detector's list in place
                 repr(layers)))

    @staticmethod
    def _same_bk_inputs(k1, k2):
        return (all(x is y for x, y in zip(k1[0], k2[0]))
                and k1[1] == k2[1])

    # ------------------------------------------------- jobs and busy state

    def _set_busy(self, kind):
        """Lock out what would change the session under a running job."""
        on = kind is not None
        for w in (self._left, self._map_tab, self.elements.tabs.widget(0)):
            w.setEnabled(not on)
        for a in self._job_actions:
            a.setEnabled(not on)
        self.prog.setVisible(on)
        self.btn_cancel.setVisible(on)
        self.btn_cancel.setEnabled(on)
        if on:
            self.prog.setRange(0, 0)
            self.prog.setFormat(kind)

    def _on_job_progress(self, kind, text, done, total):
        if kind == 'preview':
            return
        if total:
            self.prog.setRange(0, total)
            self.prog.setValue(done)
            self.prog.setFormat('%s %d/%d' % (text, done, total))
        else:
            self.prog.setRange(0, 0)
        if kind == 'batch':
            self.batch_done = done
        self.status.showMessage('%s: %s' % (kind, text))

    @user_action
    def on_cancel(self):
        if self.jobs.running_heavy():
            self.btn_cancel.setEnabled(False)
            self.say('Cancelling - the fit stops at its next step.')
        self.jobs.cancel()

    def _refuse_while_busy(self, name):
        return name in MUTATING and self.jobs.running_heavy()

    def _log_context(self):
        s, o = self.session, self.session.options
        sel = self.selected_elements()
        return {'file': getattr(s, 'path', None),
                'cal': [float(x) for x in s.cal] if s.spectrum is not None
                else None,
                'range': [o.e_low, o.e_high],
                'elements': [s.db.sym[Z] + SHELL_NAMES[sh]
                             for Z, sh in sel],
                'mask': s.mask_name,
                'chi2r': round(float(s.fit.reduced_chi2), 4)
                if s.fit is not None else None,
                'job': self.jobs.current.kind if self.jobs.current else None}

    def closeEvent(self, ev):
        self._preview_timer.stop()
        if getattr(self, '_live_timer', None) is not None:
            self._live_timer.stop()
        self.jobs.shutdown()
        # a draw queued now would run after the canvases are deleted
        self.spec.cancel_draws()
        self.map.cancel_draws()
        log_event('window closed')
        super().closeEvent(ev)

    # ------------------------------------------------------------ layout

    def _build(self):
        self._build_menu()
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._left = self._left_panel()
        split.addWidget(self._left)
        split.addWidget(self._centre())
        split.setStretchFactor(1, 1)
        left_w = max(360, self._left.minimumSizeHint().width())
        split.setSizes([left_w, DEFAULT_SIZE[0] - left_w])
        self.setCentralWidget(split)
        self.status = self.statusBar()
        self.prog = QtWidgets.QProgressBar()
        self.prog.setMaximumWidth(260)
        self.prog.setTextVisible(True)
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.status.addPermanentWidget(self.prog)
        self.status.addPermanentWidget(self.btn_cancel)
        self.prog.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.status.showMessage('Open a spectrum or list-mode file to begin.')

    def _build_menu(self):
        m = self.menuBar()
        f = m.addMenu('&File')
        f.addAction('&Open...', self.on_open, 'Ctrl+O')
        f.addAction('Set &database folder...', self.on_set_db)
        f.addAction('Attach to &live file...', self.on_attach_live)
        f.addSeparator()
        f.addAction('&Export analysis...', self.on_export, 'Ctrl+E')
        f.addAction('Export &DA matrix (.dam)...', self.on_export_dam)
        f.addAction('Export &yield file (.yield)...', self.on_export_yield)
        f.addAction('Save spectrum &image...', self.on_save_image,
                    'Ctrl+P')
        f.addSeparator()
        quit_ = f.addAction('&Quit', self.close, 'Ctrl+Q')
        a = m.addMenu('&Analysis')
        a.addAction('&Fit', self.on_fit, 'Ctrl+F')
        a.addAction('&Quantify', self.on_quantify, 'Ctrl+Shift+Q')
        a.addAction('Sample &model...', self.on_sample_model)
        a.addAction('Sample &yield files...', self.on_sample_model)
        a.addAction('&Layer stack and escape...', self.on_layers,
                    'Ctrl+L')
        a.addAction('Load &DA matrix (.dam)...', self.on_load_dam)
        a.addAction('Concentration &profiles...', self.on_profiles,
                    'Ctrl+Shift+P')
        a.addAction('&Batch...', self.on_batch, 'Ctrl+B')
        a.addSeparator()
        a.addAction('&Identify peaks', self.on_identify, 'Ctrl+I')
        a.addAction('&Suggest elements', self.on_suggest, 'Ctrl+D')
        a.addAction('Clear &mask', self.on_clear_mask)
        a.addSeparator()
        a.addAction('Check &calibration against known lines', self.on_check_cal)
        h = m.addMenu('&Help')
        h.addAction('&About', self.on_about)
        # everything in File and Analysis except Quit and Save image waits
        # while a fit or batch runs
        self._job_actions = [x for menu in (f, a) for x in menu.actions()
                             if x is not quit_ and not x.isSeparator()
                             and x.text() != 'Save spectrum &image...']

    def _left_panel(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)

        # -- file
        g = QtWidgets.QGroupBox('Data')
        gl = QtWidgets.QFormLayout(g)
        self.lbl_file = QtWidgets.QLabel('(nothing loaded)')
        self.lbl_file.setWordWrap(True)
        btn = QtWidgets.QPushButton('Open...')
        btn.clicked.connect(self.on_open)
        self.spin_adc = QtWidgets.QSpinBox()
        self.spin_adc.setRange(0, 7)
        self.spin_adc.valueChanged.connect(self.on_adc_changed)
        gl.addRow(btn)
        gl.addRow('File', self.lbl_file)
        gl.addRow('ADC', self.spin_adc)
        self.lbl_db = QtWidgets.QLabel('')
        self.lbl_db.setWordWrap(True)
        self.lbl_db.setStyleSheet('color: #666; font-size: 7.5pt;')
        gl.addRow('Database', self.lbl_db)
        v.addWidget(g)

        # -- calibration
        # a group box title cannot wrap, so the formula goes in the box
        g = QtWidgets.QGroupBox('Calibration')
        gl = QtWidgets.QFormLayout(g)
        gl.addRow(QtWidgets.QLabel('E = offset + gain x channel'))
        self.ed_gain = QtWidgets.QLineEdit('0.0017000')
        self.ed_off = QtWidgets.QLineEdit('-0.37500')
        for e in (self.ed_gain, self.ed_off):
            e.editingFinished.connect(self.on_cal_changed)
        gl.addRow('gain keV/ch', self.ed_gain)
        gl.addRow('offset keV', self.ed_off)
        v.addWidget(g)

        # -- elements
        g = QtWidgets.QGroupBox('Elements')
        gl = QtWidgets.QVBoxLayout(g)
        b_el = QtWidgets.QPushButton('Choose elements...')
        b_el.setMinimumHeight(30)
        b_el.setToolTip('Periodic table. Click an element to cycle which of '
                        'its shells are fitted; a second tab shows any '
                        'element line table.')
        b_el.clicked.connect(self.on_choose_elements)
        gl.addWidget(b_el)
        self.lbl_els = QtWidgets.QLabel('(nothing selected)')
        self.lbl_els.setWordWrap(True)
        self.lbl_els.setStyleSheet('font-size:8.25pt; color:#0F766E;')
        gl.addWidget(self.lbl_els)
        row = QtWidgets.QHBoxLayout()
        b4 = QtWidgets.QPushButton('Suggest from spectrum')
        b4.setToolTip('Find peaks, then highlight the elements whose lines '
                      'explain them. Candidates, not conclusions.')
        b4.clicked.connect(self.on_suggest)
        b5 = QtWidgets.QPushButton('Accept suggested')
        b5.clicked.connect(self.on_accept_suggested)
        row.addWidget(b4)
        row.addWidget(b5)
        gl.addLayout(row)
        self.lbl_sugg = QtWidgets.QLabel('')
        self.lbl_sugg.setWordWrap(True)
        self.lbl_sugg.setStyleSheet('color:#0F766E; font-size:7.5pt;')
        gl.addWidget(self.lbl_sugg)
        v.addWidget(g)

        # -- sample model
        g = QtWidgets.QGroupBox('Sample model')
        gl = QtWidgets.QVBoxLayout(g)
        b_sm = QtWidgets.QPushButton('Matrix and thickness...')
        b_sm.clicked.connect(self.on_sample_model)
        gl.addWidget(b_sm)
        b_ly = QtWidgets.QPushButton('Layer stack and escape...')
        b_ly.setToolTip('Draw the stack and ask what fraction of a given '
                        'X-ray escapes from each layer.')
        b_ly.clicked.connect(self.on_layers)
        gl.addWidget(b_ly)
        self.lbl_sm = QtWidgets.QLabel('matrix bootstrapped from the fit, '
                                       'thick target, 1.0 MeV, 135 deg')
        self.lbl_sm.setWordWrap(True)
        self.lbl_sm.setStyleSheet('color:#666; font-size:7.5pt;')
        gl.addWidget(self.lbl_sm)
        v.addWidget(g)

        # -- detector efficiency
        g = QtWidgets.QGroupBox('Detector efficiency')
        gl = QtWidgets.QVBoxLayout(g)
        self.cmb_eff = QtWidgets.QComboBox()
        self.cmb_eff.addItem('(none - areas only)')
        for pth in self.session.builtin_efficiencies():
            self.cmb_eff.addItem(os.path.basename(pth), pth)
        self.cmb_eff.currentIndexChanged.connect(self.on_eff_changed)
        b_eff = QtWidgets.QPushButton('Load curve from file...')
        b_eff.clicked.connect(self.on_load_eff)
        gl.addWidget(self.cmb_eff)
        gl.addWidget(b_eff)
        # A .detector file carries more than the efficiency: the crystal, the
        # energy-dependent tail, w0/w1, AND the geometry. Loading it was
        # command-line only, so the solid angle had to be typed in by hand -
        # and then disagreed with the file everything else came from.
        b_det = QtWidgets.QPushButton('Load .detector file...')
        b_det.setToolTip('GeoPIXE .detector: crystal, tail, resolution, '
                         'absorbers, active area and distance.')
        b_det.clicked.connect(self.on_load_detector)
        gl.addWidget(b_det)
        self.lbl_det = QtWidgets.QLabel('detector file: -')
        self.lbl_det.setWordWrap(True)
        self.lbl_det.setStyleSheet('color:#555; font-size:7.5pt;')
        gl.addWidget(self.lbl_det)
        note = QtWidgets.QLabel('Needed for concentrations. At C Ka a C1 '
                                'window transmits about 3%.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666; font-size:7.5pt;')
        gl.addWidget(note)
        v.addWidget(g)

        # -- options
        g = QtWidgets.QGroupBox('Fit options')
        gl = QtWidgets.QFormLayout(g)
        o = self.session.options
        self.ed_elo = QtWidgets.QLineEdit(str(o.e_low))
        self.ed_ehi = QtWidgets.QLineEdit(str(o.e_high))
        for e in (self.ed_elo, self.ed_ehi):
            e.editingFinished.connect(self.on_range_changed)
        self.cmb_mac = QtWidgets.QComboBox()
        self.cmb_mac.addItems(['mixed', 'ffast', 'henke1993', 'xcom',
                               'sabbatuccisalvat2016'])
        self.cmb_lines = QtWidgets.QComboBox()
        self.cmb_lines.addItems(['xray_lines_rebuilt.txt', 'xray_lines.txt'])
        self.cmb_lines.currentTextChanged.connect(self.on_lines_file)
        self.chk_escape = QtWidgets.QCheckBox('Si LVV escape step')
        self.chk_escape.setChecked(o.use_escape_step)
        self.chk_contact = QtWidgets.QCheckBox('Al contact injection')
        self.chk_contact.setChecked(o.use_contact_step)
        self.chk_window = QtWidgets.QCheckBox('Si3N4 window (N K edge)')
        self.chk_window.setChecked(o.use_window_step)
        self.chk_pileup = QtWidgets.QCheckBox('Model sum-peak pile-up')
        self.chk_pileup.setChecked(o.use_pileup)
        self.chk_pileup.setToolTip(
            'Two photons inside the shaping time are recorded as one at the '
            'sum energy. Small, but it lands in empty regions - exactly where '
            'a fit will otherwise invent an element.')
        self.chk_nonneg = QtWidgets.QCheckBox('Forbid negative areas')
        self.chk_nonneg.setChecked(o.nonneg == 'strict')
        self.chk_freeze = QtWidgets.QCheckBox('Hold line shape from last fit')
        self.chk_freeze.setChecked(o.freeze_nonlinear)
        self.chk_freeze.setToolTip(
            'Refine nothing non-linear - no calibration, width, tail, shelf '
            'or sum deficit - and solve only the areas, starting from the '
            'the last shape. Twenty times faster, and it is the only way '
            'to compare two regions of one map: with free refinement, two '
            'masks over the same pixels came out 4-14 % apart. Fit the whole '
            'field free first, then tick this for the regions. Unticked, '
            'every fit starts from the detector width and the calibration '
            'boxes, never from the last fit.')
        self.chk_freeze.toggled.connect(self.on_hold_toggled)
        self.btn_reset_shape = QtWidgets.QPushButton('Reset line shape')
        self.btn_reset_shape.setToolTip(
            'Forget the last fit\'s line shape. The next fit, held or not, '
            'refines from the detector width and the calibration boxes.')
        self.btn_reset_shape.clicked.connect(self.on_reset_shape)
        gl.addRow('E low keV', self.ed_elo)
        gl.addRow('E high keV', self.ed_ehi)
        gl.addRow('MAC dataset', self.cmb_mac)
        gl.addRow('Line data', self.cmb_lines)
        for c in (self.chk_escape, self.chk_contact, self.chk_window,
                  self.chk_pileup, self.chk_nonneg, self.chk_freeze,
                  self.btn_reset_shape):
            gl.addRow(c)
        v.addWidget(g)

        # -- geometry, for absolute quantification -----------------------
        gg = QtWidgets.QGroupBox('Charge and solid angle')
        gg.setToolTip(
            'Fill these in and QUANTIFY reports absolute weight percent '
            'instead of normalising to 100. The sum then becomes a test: it '
            'is only near 100 if the whole chain is right. Leave them blank '
            'to normalise as before.')
        ggl = QtWidgets.QFormLayout(gg)
        self.ed_charge = QtWidgets.QLineEdit('')
        self.ed_charge.setPlaceholderText('blank = normalise to 100')
        self.ed_charge.setToolTip(
            'Integrated beam charge in microcoulomb. The weakest link in the '
            'chain: current read on the sample rather than in a Faraday cup '
            'runs high unless secondary electrons are suppressed.')
        self.ed_dist = QtWidgets.QLineEdit('')
        self.ed_dist.setToolTip(
            'Sample to detector face, mm. Enters as 1/r^2, so on a short '
            'working distance a couple of mm of error is a large one.')
        self.ed_area = QtWidgets.QLineEdit('')
        self.ed_area.setToolTip('Active area in mm2 - how Amptek quotes it '
                                '(25, 70). Use the collimated area if there '
                                'is a collimator.')
        self.ed_tilt = QtWidgets.QLineEdit('0')
        self.ed_dead = QtWidgets.QLineEdit('0')
        self.ed_dead.setToolTip(
            'Dead time percent. Ignoring it makes every concentration low by '
            'the same factor - invisible once normalised, obvious absolutely.')
        self.lbl_omega = QtWidgets.QLabel('solid angle: -')
        self.lbl_omega.setStyleSheet('color:#555')
        for e in (self.ed_charge, self.ed_dist, self.ed_area, self.ed_tilt,
                  self.ed_dead):
            e.textChanged.connect(self._geom_changed)
        ggl.addRow('charge uC', self.ed_charge)
        ggl.addRow('distance mm', self.ed_dist)
        ggl.addRow('active area mm2', self.ed_area)
        ggl.addRow('detector tilt deg', self.ed_tilt)
        ggl.addRow('dead time %', self.ed_dead)
        ggl.addRow(self.lbl_omega)
        v.addWidget(gg)

        row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton('FIT')
        self.btn_fit.setMinimumHeight(34)
        self.btn_fit.clicked.connect(self.on_fit)
        self.btn_q = QtWidgets.QPushButton('QUANTIFY')
        self.btn_q.setMinimumHeight(34)
        self.btn_q.clicked.connect(self.on_quantify)
        row.addWidget(self.btn_fit, 2)
        row.addWidget(self.btn_q, 1)
        v.addLayout(row)
        v.addStretch(1)

        return _VScrollArea(w)

    def _centre(self):
        tabs = QtWidgets.QTabWidget()

        # spectrum tab
        sw = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sw)
        self.spec = SpectrumCanvas()
        self.spec.rangeSelected.connect(self.on_range)
        bar = QtWidgets.QHBoxLayout()
        cb_log = QtWidgets.QCheckBox('log y')
        cb_log.setChecked(True)
        cb_log.toggled.connect(self.spec.set_log)
        cb_cmp = QtWidgets.QCheckBox('components')
        cb_cmp.setChecked(True)
        cb_cmp.toggled.connect(self.spec.set_components)
        cb_lab = QtWidgets.QCheckBox('line labels')
        cb_lab.setChecked(True)
        cb_lab.toggled.connect(self.on_labels)
        self._want_labels = True
        self.cmb_legend = QtWidgets.QComboBox()
        self.cmb_legend.addItems(LEGEND_MODES)
        self.cmb_legend.setToolTip(
            'compact: data, fit, background and artefacts. all: every drawn '
            'element component as well. off: no legend.')
        self.cmb_legend.currentTextChanged.connect(self.on_legend)
        self.btn_lines = QtWidgets.QPushButton('Lines...')
        self.btn_lines.setToolTip(
            'Choose which element groups are drawn and listed in the legend. '
            'The choice is kept across fits; the fit itself is unchanged.')
        self.btn_lines.clicked.connect(self.on_lines)
        self.chk_field = QtWidgets.QCheckBox('whole field')
        self.chk_field.setToolTip(
            'With a region (mask) active, draw the whole-field spectrum behind '
            "the region's, scaled by the region's share of the scanned pixels "
            '(equal counts per pixel, not equal charge). A mask thresholded on '
            "an element's map is biased for that element's line: compare "
            'other lines, or cut on half the events and read the other half.')
        self.chk_field.setEnabled(False)       # until a mask is set
        self.chk_field.toggled.connect(self.on_field_overlay)
        b_full = QtWidgets.QPushButton('Full range')
        b_full.clicked.connect(self.sessionChanged.emit)
        self.chk_preview = QtWidgets.QCheckBox('live estimate')
        self.chk_preview.setChecked(True)
        self.chk_preview.setToolTip(
            'Redraw an initial estimate - peak-height areas, no fitting - '
            'whenever the elements, calibration or range change, as '
            'GeoPIXE does while a fit is set up. FIT replaces it.')
        self.chk_preview.toggled.connect(self.on_preview_toggled)
        self.chk_id = QtWidgets.QCheckBox('identify on click')
        self.chk_id.setChecked(True)
        self.chk_id.setToolTip('Click anywhere on the spectrum to list the '
                               'X-ray lines near that energy.')
        b_pk = QtWidgets.QPushButton('Find peaks')
        b_pk.clicked.connect(self.on_identify)
        b_lab = QtWidgets.QPushButton('Label all peaks')
        b_lab.setToolTip('Annotate every detected peak with its best '
                         'identification, ready for export.')
        b_lab.clicked.connect(self.on_label_peaks)
        b_art = QtWidgets.QPushButton('Show escape + sum peaks')
        b_art.setToolTip('Mark where detector escape peaks and pile-up sum '
                         'peaks fall. Both put real features where no element '
                         'emits, and a fit offered one will reach for the '
                         'nearest element.')
        b_art.clicked.connect(self.on_artefacts)
        b_nomark = QtWidgets.QPushButton('Clear markers')
        b_nomark.clicked.connect(self.on_clear_markers)

        # Explicit axis limits. Matplotlib's pan and zoom are there, but
        # reading a fit needs an exact window - "show me 6.0 to 6.8 keV, full
        # scale 700k" - and dragging cannot do that reproducibly.
        self.ed_x0 = QtWidgets.QLineEdit()
        self.ed_x1 = QtWidgets.QLineEdit()
        self.ed_y1 = QtWidgets.QLineEdit()
        for w, tip, ph in ((self.ed_x0, 'left edge, keV. Blank = fit range.',
                            'x lo'),
                           (self.ed_x1, 'right edge, keV. Blank = fit range.',
                            'x hi'),
                           (self.ed_y1, 'full scale in counts. Blank = auto.',
                            'y max')):
            w.setToolTip(tip)
            w.setPlaceholderText(ph)
            w.setMaximumWidth(70)
            w.returnPressed.connect(self.on_axes)
        b_ax = QtWidgets.QPushButton('Set axes')
        b_ax.clicked.connect(self.on_axes)
        b_axr = QtWidgets.QPushButton('Auto')
        b_axr.setToolTip('Back to the fit range and automatic full scale.')
        b_axr.clicked.connect(self.on_axes_reset)
        # Two rows: on one, this bar's minimum width (1244 px at 100 %) left
        # the left panel no room in a 1500 px window.
        bar2 = QtWidgets.QHBoxLayout()
        for x in (cb_log, cb_cmp, cb_lab, self.chk_preview, self.chk_id,
                  self.ed_x0, self.ed_x1, self.ed_y1, b_ax, b_axr):
            bar.addWidget(x)
        for x in (b_full, b_pk, b_lab, b_art, b_nomark,
                  QtWidgets.QLabel('legend'), self.cmb_legend, self.btn_lines,
                  self.chk_field):
            bar2.addWidget(x)
        bar.addStretch(1)
        bar2.addStretch(1)
        sv.addWidget(toolbar_for(self.spec, sw))
        sv.addLayout(bar)
        sv.addLayout(bar2)
        sv.addWidget(self.spec, 1)
        self.spec.mpl_connect('button_press_event', self.on_spec_click)
        self._suggested = []
        tabs.addTab(sw, 'Spectrum')

        # map tab
        mw = QtWidgets.QWidget()
        self._map_tab = mw
        mv = QtWidgets.QVBoxLayout(mw)
        top = QtWidgets.QHBoxLayout()
        self.cmb_mapel = QtWidgets.QComboBox()
        self.cmb_mapel.setEditable(True)
        self.cmb_mapel.setToolTip('A line in the fit range (Si, Ag L, Pb M), '
                                   'or a bare energy in keV')
        b_map = QtWidgets.QPushButton('Show map')
        b_map.clicked.connect(self.on_map)
        # GeoPIXE steps through its element maps with a droplist only
        # (gimage.pro:1636-1647); one click per element is quicker.
        self.btn_map_prev = QtWidgets.QToolButton()
        self.btn_map_prev.setArrowType(QtCore.Qt.LeftArrow)
        self.btn_map_prev.setToolTip('Map the previous element in the list')
        self.btn_map_prev.clicked.connect(lambda: self.on_map_step(-1))
        self.btn_map_next = QtWidgets.QToolButton()
        self.btn_map_next.setArrowType(QtCore.Qt.RightArrow)
        self.btn_map_next.setToolTip('Map the next element in the list')
        self.btn_map_next.clicked.connect(lambda: self.on_map_step(1))
        self.spin_bin = QtWidgets.QSpinBox()
        self.spin_bin.setRange(1, 32)
        self.spin_bin.setValue(4)
        self.chk_sub = QtWidgets.QCheckBox('subtract background')
        self.chk_sub.setChecked(True)
        self.spin_thr = QtWidgets.QSpinBox()
        self.spin_thr.setRange(50, 99)
        self.spin_thr.setValue(90)
        b_thr = QtWidgets.QPushButton('Mask above percentile')
        b_thr.clicked.connect(self.on_threshold)
        b_clr = QtWidgets.QPushButton('Clear mask')
        b_clr.clicked.connect(self.on_clear_mask)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(['rectangle', 'polygon', 'flood fill'])
        self.cmb_mode.setToolTip('rectangle: drag.  polygon: click corners, '
                                 'click the first again to close.  flood '
                                 'fill: click a pixel to seed')
        # GeoPIXE's include / exclude (gimage.pro:1676-1679)
        self.cmb_op = QtWidgets.QComboBox()
        self.cmb_op.addItems(['replace', 'add', 'subtract'])
        self.cmb_op.setToolTip('What a new shape does to the current mask')
        self.cmb_mode.currentTextChanged.connect(self.on_mode_changed)
        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(0.02, 3.0)
        self.spin_tol.setSingleStep(0.05)
        self.spin_tol.setValue(0.35)
        b_grow = QtWidgets.QPushButton('grow')
        b_grow.clicked.connect(lambda: self.on_morph(1))
        b_shrink = QtWidgets.QPushButton('shrink')
        b_shrink.clicked.connect(lambda: self.on_morph(-1))
        for x, lab in ((self.cmb_mapel, 'element'), (self.spin_bin, 'binning'),
                       (self.spin_thr, 'percentile')):
            top.addWidget(QtWidgets.QLabel(lab))
            if x is self.cmb_mapel:
                top.addWidget(self.btn_map_prev)
                top.addWidget(x)
                top.addWidget(self.btn_map_next)
                continue
            top.addWidget(x)
        top.addWidget(self.chk_sub)
        top.addWidget(b_map)
        top.addWidget(b_thr)
        top.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel('mask tool'))
        row2.addWidget(self.cmb_mode)
        row2.addWidget(self.cmb_op)
        row2.addWidget(QtWidgets.QLabel('flood tolerance'))
        row2.addWidget(self.spin_tol)
        row2.addWidget(b_grow)
        row2.addWidget(b_shrink)
        row2.addWidget(b_clr)
        self.btn_map_reset = QtWidgets.QPushButton('Reset view')
        self.btn_map_reset.setToolTip('Whole field and default contrast '
                                      '(the mask is kept)')
        self.btn_map_reset.clicked.connect(self.on_map_reset)
        row2.addWidget(self.btn_map_reset)
        row2.addStretch(1)
        # display: colour table, intensity scale, percentile limits
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(MAP_CMAPS)
        self.cmb_scale = QtWidgets.QComboBox()
        self.cmb_scale.addItems(MAP_SCALES)
        self.spin_plo = QtWidgets.QDoubleSpinBox()
        self.spin_phi = QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.spin_plo, MAP_PCT[0]), (self.spin_phi, MAP_PCT[1])):
            sp.setRange(0.0, 100.0)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setValue(v)
            sp.setSuffix(' %ile')
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self.on_map_display)
        self.spin_plo.setToolTip('Colour scale bottom: this percentile of the map')
        self.spin_phi.setToolTip('Colour scale top: this percentile of the map. '
                                 'Percentiles, so one hot pixel does not set it')
        self.cmb_cmap.currentTextChanged.connect(self.on_map_display)
        self.cmb_scale.currentTextChanged.connect(self.on_map_display)
        row3 = QtWidgets.QHBoxLayout()
        for x, lab in ((self.cmb_cmap, 'colours'), (self.cmb_scale, 'scale'),
                       (self.spin_plo, 'low'), (self.spin_phi, 'high')):
            row3.addWidget(QtWidgets.QLabel(lab))
            row3.addWidget(x)
        b_exp = QtWidgets.QPushButton('Export map...')
        b_exp.setToolTip('The shown map as numbers (CSV), and the mask '
                         'beside it if one is set')
        b_exp.clicked.connect(self.on_export_map)
        row3.addWidget(b_exp)
        row3.addStretch(1)
        self.map = MapCanvas()
        self.map_toolbar = toolbar_for(self.map, mw)
        self.map.regionSelected.connect(self.on_region)
        self.map.polygonSelected.connect(self.on_polygon)
        mv.addLayout(top)
        mv.addLayout(row2)
        mv.addLayout(row3)
        mv.addWidget(self.map_toolbar)
        mv.addWidget(self.map, 1)
        self.map.mpl_connect('button_press_event', self.on_map_click)
        tabs.addTab(mw, 'Maps')

        # results tab
        rw = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rw)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['component', 'area', 'error', 'rel %', 'wt %'])
        self.table.horizontalHeader().setStretchLastSection(True)
        rv.addWidget(self.table)
        tabs.addTab(rw, 'Results')

        # log tab
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFont('Consolas', 9))
        tabs.addTab(self.log, 'Log')
        self.tabs = tabs
        return tabs

    # ------------------------------------------------------------ helpers

    def say(self, msg):
        log_event('message', text=str(msg)[:2000])
        self.log.appendPlainText(msg)
        self.status.showMessage(msg, 8000)

    def _refresh_db_label(self):
        p = config.database_path(required=False)
        self.lbl_db.setText(_wrappable_path(p) if p
                            else 'NOT FOUND - set one via File menu')
        self.lbl_db.setToolTip(str(p) if p else '')

    def preset(self, k, l, m):
        pairs = []
        for shell, names in ((1, k), (2, l), (3, m)):
            for nm in names:
                Z = self.session.db.z.get(nm.lower())
                if Z:
                    pairs.append((Z, shell))
        self.elements.set_selection(pairs)

    def selected_elements(self):
        return self.elements.selection()

    def line_labels(self):
        if not self._want_labels:
            return []
        out = []
        for Z, sh in self.selected_elements():
            e = self.session.db.line_energy(Z, sh)
            if e > 0:
                out.append((self.session.db.sym[Z] +
                            {1: '', 2: 'L', 3: 'M'}[sh], e))
        return out

    def apply_options(self):
        o = self.session.options
        try:
            o.e_low = float(self.ed_elo.text())
            o.e_high = float(self.ed_ehi.text())
        except ValueError:
            pass
        o.mac = self.cmb_mac.currentText()
        if (o.e_low, o.e_high) != getattr(self, '_range_shown', None):
            self.rebuild_element_lists()
        o.use_escape_step = self.chk_escape.isChecked()
        o.use_contact_step = self.chk_contact.isChecked()
        o.use_window_step = self.chk_window.isChecked()
        o.use_pileup = self.chk_pileup.isChecked()
        o.nonneg = 'strict' if self.chk_nonneg.isChecked() else True
        self._sync_hold()
        self.session.invalidate()

    def _sync_hold(self):
        """What the next fit (and the live estimate) starts from.

        Held, with a fit to hold: the last fit's converged shape, fixed.
        Otherwise the detector / options shape at the calibration in the
        boxes, refined freely, every fit. GeoPIXE seeds each fit from its
        detector file too (pixe_setup.pro:71-72, 106-107) but writes a free
        fit's width back into it (pixe_cleanup.pro:75-80); this does not.
        Carrying each fit's shape into the next let one misfit spoil every
        later fit: 287424, 2026-10-20, fit 2.382, without Al Si C 359.4,
        all back 8.411 - started from the misfit's shape (noise 20.05 ch)
        it converged to noise 11.71 ch.
        """
        s = self.session
        held = self.chk_freeze.isChecked() and self.last_shape is not None
        s.options.freeze_nonlinear = held
        s.start_pars = self.last_shape if held else None

    @user_action
    def on_hold_toggled(self, on):
        self._sync_hold()
        self.mark_changed()

    @user_action
    def on_reset_shape(self):
        self.last_shape = None
        self._sync_hold()
        self.say('Line shape reset: the next fit refines from the detector '
                 'width and the calibration boxes.')
        self.mark_changed()

    # ------------------------------------------------------------ actions

    @user_action
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open spectrum or list-mode file', '',
            'All supported (*.lmf *.dam *.spec *.txt);;'
            'OMDAQ list mode (*.lmf);;GeoPIXE DA matrix (*.dam);;'
            'Text (*.txt *.spec);;All files (*)')
        if not path:
            return
        self.open_path(path)

    @user_action
    def open_path(self, path):
        """Load a file and bring the whole window up to date.

        on_open() is only the file dialog; everything after the load lives
        here so that opening from the command line cannot drift out of step
        with opening from the menu.
        """
        try:
            self.session.load(path, adc=self.spin_adc.value())
        except Exception as ex:
            self.say('LOAD FAILED: %s' % ex)
            self.yieldChanged.emit()           # the load cleared it
            QtWidgets.QMessageBox.critical(self, 'Load failed', str(ex))
            return False
        # another spectrum: nothing of the last one's shape is held over
        # (GeoPIXE restores the detector's width on a new spectrum too,
        # fit_setup.pro:95-100)
        self.last_shape = None
        self._sync_hold()
        s = self.session
        self.lbl_file.setText('%s\n%s' % (os.path.basename(path), s.label))
        self.ed_gain.setText('%.7f' % s.cal[0])
        self.ed_off.setText('%.5f' % s.cal[1])
        self.say('Loaded %s: %d channels, %.0f counts%s'
                 % (os.path.basename(path), len(s.spectrum), s.spectrum.sum(),
                    ', %d events with positions' % len(s.events[0])
                    if s.events is not None else ''))
        if s.events is None:
            self.say('  no event positions in this format - maps and masking '
                     'are unavailable (open the .lmf for those)')
        # Fit range. The defaults are for 0.2-6.6 keV light-element work,
        # and leaving them on a spectrum that runs to 43 keV is not a subtle
        # error: the fit covers a few percent of the counts, every real peak
        # is outside it, and the model comes back as one smooth curve with no
        # peaks in it at all. Snap the range to the data whenever the current
        # one misses most of the spectrum.
        self._autorange()

        # Width parameters are in channels, so a value tuned for one gain is
        # wrong for another. Seed them from the tallest peak in the data.
        est = s.estimate_resolution()
        if est:
            self.say('  resolution seeded from the data: %.0f eV FWHM at the '
                     'tallest peak (noise %.1f, Fano %.2f channels)'
                     % (est[2], est[0], est[1]))

        # Charge, if it can be had without asking. The run log beside the data
        # is the measurement; the LMF dose counter is a good proxy but carries
        # whatever digitiser range was set. Say which was used either way, so
        # nobody mistakes the proxy for the measurement.
        q, src = s.auto_charge()
        if q:
            self.ed_charge.setText('%.4f' % q)
            self.say('  charge %.4f uC, from the %s' % (q, src))
            if src != 'run log':
                self.say('  (0.1 nC per count, measured on the Dec-2023 '
                         'session - confirm it against the run log for '
                         'this one)')
        else:
            self.say('  no charge available (%s) - QUANTIFY will normalise '
                     'to 100 wt%% unless you type one in' % src)
        if s.yield_set is not None:
            self.say('  yield files: %s, %s (QUANTIFY uses them; Analysis > '
                     'Sample yield files to change)'
                     % (s.yield_set.describe(s.yield_region),
                        getattr(s.yield_set, 'source', '')))
            if s.yield_region is None and len(s.yield_set.regions) > 1:
                self.say('  the set covers regions %s - choose one in '
                         'Sample model before QUANTIFY'
                         % ', '.join(s.yield_set.regions))
        else:
            self.say('  no .yield files: %s; QUANTIFY will bootstrap a '
                     'one-layer matrix' % s.yield_search)
        self.yieldChanged.emit()
        self._geom_changed()
        self.mark_changed()
        return True

    @user_action
    def on_adc_changed(self):
        if self.session.path and self.session.path.lower().endswith('.lmf'):
            # Same sample, another detector: keep its yield file.
            y = (self.session.yield_path, self.session.yield_source)
            ys = (self.session.yield_set, self.session.yield_region,
                  getattr(self.session.yield_set, 'source', ''))
            self.session.load(self.session.path, adc=self.spin_adc.value())
            if ys[0] is not None:
                self.session.set_yield_set(*ys)
            self.last_shape = None          # another detector's lines
            self._sync_hold()
            if y[0]:
                self.session.set_yield(*y)
            self.yieldChanged.emit()
            self.mark_changed()

    @user_action
    def on_set_db(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select the GeoPIXE database folder (contains dat/)')
        if not d:
            return
        try:
            config.save_database_path(d)
            self.session._db = None
            self._refresh_db_label()
            self.say('Database set to %s' % d)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, 'Not a database folder', str(ex))

    @user_action
    def on_cal_changed(self):
        if self.session.spectrum is None:
            return
        try:
            self.session.set_calibration(float(self.ed_gain.text()),
                                         float(self.ed_off.text()))
        except ValueError:
            return
        self.mark_changed()

    @user_action
    def on_range_changed(self):
        """E low / E high edited: rebuild what the range allows and redraw."""
        o = self.session.options
        was = (o.e_low, o.e_high)
        self.apply_options()
        if (o.e_low, o.e_high) != was:
            self.mark_changed()

    @user_action
    def on_lines_file(self, name):
        self.session.set_lines_file(name)
        self.mark_changed()

    @user_action
    def on_preview_toggled(self, on):
        if not on:
            self._preview_timer.stop()
            self.preview = None
            self.previewUpdated.emit()
        else:
            self._schedule_preview()

    @user_action
    def on_clear_markers(self):
        self.spec.clear_markers()

    @user_action
    def on_labels(self, on):
        self._want_labels = bool(on)
        self.sessionChanged.emit()

    @user_action
    def on_legend(self, mode):
        self.spec.legend_mode = mode
        self.sessionChanged.emit()

    def lines_dialog(self):
        """(dialog, list) choosing the element groups drawn; None before
        anything is drawn. Split from on_lines so tests can drive it."""
        groups = self.spec.element_groups()
        if not groups:
            return None
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Lines drawn and listed')
        v = QtWidgets.QVBoxLayout(dlg)
        lst = QtWidgets.QListWidget()
        # wrapped into columns, so 22 groups fit without scrolling
        lst.setFlow(QtWidgets.QListView.LeftToRight)
        lst.setWrapping(True)
        lst.setResizeMode(QtWidgets.QListView.Adjust)
        lst.setGridSize(QtCore.QSize(80, 22))
        lst.setMinimumSize(420, 180)
        for g in groups:
            it = QtWidgets.QListWidgetItem(g)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked if g in self.spec.hidden
                             else QtCore.Qt.Checked)
            lst.addItem(it)
        v.addWidget(lst)
        row = QtWidgets.QHBoxLayout()
        for text, state in (('All', QtCore.Qt.Checked),
                            ('None', QtCore.Qt.Unchecked)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, st=state: [
                lst.item(i).setCheckState(st) for i in range(lst.count())])
            row.addWidget(b)
        row.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                        | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        row.addWidget(bb)
        v.addLayout(row)
        return dlg, lst

    def apply_lines(self, lst):
        """Hide the unticked groups of a lines_dialog list. Groups not in
        this fit keep their earlier choice."""
        shown = {lst.item(i).text() for i in range(lst.count())}
        off = {lst.item(i).text() for i in range(lst.count())
               if lst.item(i).checkState() != QtCore.Qt.Checked}
        self.spec.set_hidden((self.spec.hidden - shown) | off)

    @user_action
    def on_lines(self):
        made = self.lines_dialog()
        if made is None:
            self.say('Nothing drawn yet: Lines... chooses among the element '
                     'groups of the fit or the live estimate.')
            return
        dlg, lst = made
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.apply_lines(lst)

    @user_action
    def on_field_overlay(self, on):
        self.sessionChanged.emit()

    def _region_legend(self):
        """(data label, reference) for the spectrum plot. With a mask the
        data is the region's spectrum: name, raster pixels and counts. The
        whole field, when asked for, is scaled by the region's share of the
        raster, so equal heights mean equal counts per pixel."""
        s = self.session
        if s.mask is None:
            return 'data', None
        npx = int(s.mask.sum())
        label = 'region %s: %d raster px, %d counts' % (
            s.mask_name or 'mask', npx, int(round(s.spectrum.sum())))
        ref = None
        if self.chk_field.isChecked() and s.full_spectrum is not None:
            # share of the pixels that hold events, so a scan that did not
            # cover the whole 256 x 256 raster does not read as enrichment
            hit = self._scanned_pixels(s.mask.shape)
            f = (npx if hit is None else int((hit & s.mask).sum())) / max(
                s.mask.size if hit is None else int(hit.sum()), 1)
            ref = (s.full_spectrum * f,
                   'whole field x%.3g (region area), %d counts'
                   % (f, int(round(s.full_spectrum.sum()))))
        return label, ref

    def _scanned_pixels(self, shape):
        """Raster pixels holding at least one event (cached per event set),
        or None without list-mode events."""
        ev = self.session.events
        if ev is None:
            return None
        key = (id(ev[0]), len(ev[0]), shape)
        if getattr(self, '_hit_key', None) != key:
            hit = np.zeros(shape, bool)
            x, y = ev[0], ev[1]
            ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
            hit[y[ok], x[ok]] = True
            self._hit_key, self._hit = key, hit
        return self._hit

    @user_action
    def on_range(self, lo, hi):
        self.spec.set_xlim(lo, hi)
        self.say('zoom %.3f - %.3f keV' % (lo, hi))

    def refresh_spectrum(self):
        """Draw the session. Connected to sessionChanged and previewUpdated;
        call mark_changed() rather than this.

        The curve is the fit while it matches the selected elements, else the
        live estimate. A fit made with a different element list is no longer
        a description of what is selected, so it is not drawn as one.
        """
        s = self.session
        if s.spectrum is None:
            return
        title = s.label + (' [%s]' % s.mask_name if s.mask_name else '')
        o = s.options
        cur, bk = None, s._bk
        if self._fit_is_current() or (s.fit is not None
                                      and self._fit_sel is None):
            cur = s.fit
        elif self.preview is not None and self.chk_preview.isChecked():
            cur, bk = self.preview, self.preview.background
            title += '   [initial estimate - not fitted]'
        self.chk_field.setEnabled(s.mask is not None)
        data_label, ref = self._region_legend()
        self.spec.show(s.energy, s.spectrum, fit=cur,
                       background=bk, labels=self.line_labels(),
                       title=title, xlim=(o.e_low, o.e_high),
                       data_label=data_label, reference=ref)

    @user_action
    def on_fit(self):
        s = self.session
        if s.spectrum is None:
            self.say('Nothing loaded.')
            return
        els = self.selected_elements()
        if not els:
            self.say('Select at least one element.')
            return
        self.apply_options()
        if self.chk_freeze.isChecked() and s.start_pars is None:
            self.say('No previous fit to hold the shape from - refining freely '
                     'this once.')
        # Each element spec becomes one component, except an M shell, which
        # becomes five. 22 specs on the perovskite is 47 components and 40 s
        # of free refinement. It runs on the worker thread with a progress
        # bar and Cancel; the message says what is being attempted.
        self.say('Fitting %d element groups%s...'
                 % (len(els), ' (line shape held)'
                    if s.options.freeze_nonlinear and s.start_pars is not None
                    else ''))
        els = list(els)
        self._preview_timer.stop()

        def work(ctx):
            return s.run_fit(els, should_stop=ctx.should_stop,
                             progress=ctx.progress)

        def failed(tb):
            self._set_busy(None)
            self.say('FIT FAILED: %s' % tb.strip().splitlines()[-1])
            self.log.appendPlainText(tb)
            self.mark_changed()

        def cancelled():
            self._set_busy(None)
            self.say('Fit cancelled; no result kept.')
            self.mark_changed()

        if self.jobs.submit(Job('fit', work, on_done=lambda r: self._fit_done(
                r, els), on_fail=failed, on_cancel=cancelled)):
            self._set_busy('fitting')

    def _fit_done(self, r, els):
        s = self.session
        self._set_busy(None)
        self._fit_sel = sorted(els)
        # Keep the converged shape for 'hold line shape', which is what makes
        # region fits comparable. Only the hold starts a fit from it.
        self.last_shape = r.pars
        self._sync_hold()
        self.say('Fit: chi2red = %.3f over %d components%s'
                 % (r.reduced_chi2, len(r.names),
                    ' [%s]' % s.mask_name if s.mask_name else ''))
        self.fill_table(r)
        self.sessionChanged.emit()
        # What the fit could not explain. A peak here that lines up with an
        # element not in the list is the usual cause of a bad chi2 - and,
        # until the width was bounded, of a fit that smeared over it.
        try:
            pk = s.unexplained_peaks(3)
        except Exception:
            pk = []
        for E, ex, sg, cands in pk:
            if sg < 8.0:
                continue
            self.log.appendPlainText(
                '  unexplained peak at %.2f keV: %.0f counts (%.0f sigma)'
                '  candidates: %s' % (E, ex, sg, ', '.join(cands) or '?'))

    def fill_table(self, r):
        conc = {}
        if getattr(self.session, '_conc', None):
            conc = {self.session.db.sym[Z]: v
                    for Z, v in self.session._conc.items()}
        rows = sorted(zip(r.names, r.areas, r.errors), key=lambda t: -t[1])
        self.table.setRowCount(len(rows))
        for i, (nm, a, e) in enumerate(rows):
            rel = '%.1f' % (100 * e / a) if a > 0 else '-'
            wt = '%.3f' % conc[nm] if nm in conc else ''
            for j, txt in enumerate((nm, '%.0f' % a, '%.0f' % e, rel, wt)):
                it = QtWidgets.QTableWidgetItem(txt)
                if a <= 0:
                    it.setForeground(QtGui.QBrush(QtGui.QColor('#999')))
                self.table.setItem(i, j, it)

    @user_action
    def on_map(self):
        s = self.session
        if s.events is None:
            self.say('Maps need a list-mode (.lmf) file.')
            return False
        el = self.cmb_mapel.currentText().strip()
        try:
            m = s.element_map(el, binning=self.spin_bin.value(),
                              subtract=self.chk_sub.isChecked())
        except Exception as ex:
            self.say('MAP FAILED: %s' % ex)
            return False
        self.map.show_map(m, '%s   (%dx%d bin%s)'
                          % (el, self.spin_bin.value(), self.spin_bin.value(),
                             ', net' if self.chk_sub.isChecked() else ', raw'))
        # what made this map, taken now: a later open or calibration edit
        # must not relabel numbers it did not produce
        self._map_meta = dict(
            element=el, binning=self.spin_bin.value(),
            kind='net' if self.chk_sub.isChecked() else 'raw',
            label=s.label, cal=tuple(s.cal),
            stem=re.sub(r'[^\w.-]+', '_', pathlib.Path(s.path or s.label
                                                        or 'map').stem))
        self.map.set_mask(s.mask)
        self.tabs.setCurrentIndex(1)
        if self.chk_sub.isChecked() and np.nanmedian(m) < 0:
            self.say('WARNING: this net map is mostly negative. Flanking '
                     'subtraction fails where a weak line sits between strong '
                     'neighbours (Al between Mg and Si is the usual case). '
                     'Untick "subtract background" and read it as raw counts.')
        return True

    @user_action
    def on_map_step(self, step):
        n = self.cmb_mapel.count()
        if n == 0:
            self.say('No elements to step through - set the range first.')
            return
        prev = self.cmb_mapel.currentText()
        i = self.cmb_mapel.findText(prev.strip(), QtCore.Qt.MatchFixedString)
        i = (i + step) % n if i >= 0 else (0 if step > 0 else n - 1)
        self.cmb_mapel.setCurrentIndex(i)
        if not self.on_map():        # failed: keep the list on the element
            self.cmb_mapel.setCurrentText(prev)       # the canvas shows

    @user_action
    def on_region(self, mask):
        self._apply_mask(mask, 'region')

    @user_action
    def on_polygon(self, mask):
        self._apply_mask(mask, 'polygon')

    @user_action
    def on_threshold(self):
        m = self.map.threshold_mask(self.spin_thr.value())
        if m is None:
            self.say('Show a map first.')
            return
        if m.all() or not m.any():
            # a flat map: every pixel is at the percentile, so the 'top 10 %'
            # would be the whole field (or nothing) under a region's name
            self.say('No mask: this map is flat, so a percentile cannot pick '
                     'out a region. Map a line with counts first.')
            return
        self._apply_mask(m, 'top%d%%' % (100 - self.spin_thr.value()))

    @user_action
    def on_clear_mask(self):
        self.session.clear_mask()
        self.map.set_mask(None)
        self.say('Mask cleared - back to the whole field.')
        self.mark_changed()

    @user_action
    def on_check_cal(self):
        """Fit the centroid of each selected element's main line and report
        how far it is from the true energy. Calibration errors of a few tens
        of eV move light-element lines onto each other."""
        s = self.session
        if s.spectrum is None:
            return
        from scipy.optimize import least_squares
        a, b = s.cal
        out = []
        for Z, sh in self.selected_elements():
            if sh != 1:
                continue
            E0 = s.db.line_energy(Z)
            if E0 <= 0:
                continue
            c0 = (E0 - b) / a
            sg = max(np.sqrt(s.options.noise ** 2 +
                             s.options.fano ** 2 * E0) / 2.355, 3.0)
            lo, hi = int(max(c0 - 4 * sg, 2)), int(min(c0 + 4 * sg,
                                                       len(s.spectrum) - 2))
            if hi - lo < 8:
                continue
            x = np.arange(lo, hi + 1, dtype=float)
            y = s.spectrum[lo:hi + 1].astype(float)

            def mdl(p):
                return p[0] * np.exp(-0.5 * ((x - p[1]) / max(p[2], .5)) ** 2) + p[3]

            def res(p):
                mm = np.maximum(mdl(p), 1e-9)
                return (y - mm) / np.sqrt(np.maximum(mm, 1.0))
            try:
                p = least_squares(
                    res, [max(y.max(), 1), c0, sg, max(np.median(y), .1)],
                    bounds=([0, c0 - 12, .5, 0],
                            [np.inf, c0 + 12, 5 * sg, np.inf]),
                    max_nfev=2000).x
                out.append('   %-3s %7.3f keV   error %+5.0f eV'
                           % (s.db.sym[Z], E0, 1000 * (a * p[1] + b - E0)))
            except Exception:
                pass
        self.say('Calibration check:\n' + ('\n'.join(out) if out else
                                           '   select some K-line elements first'))
        self.tabs.setCurrentIndex(3)

    @user_action
    def on_export(self):
        s = self.session
        if s.spectrum is None:
            self.say('Nothing to export.')
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Choose a folder to write the analysis into')
        if not d:
            return
        try:
            w = s.export(d)
        except Exception as ex:
            self.say('EXPORT FAILED: %s' % ex)
            return
        self.say('Exported %d files to %s:\n   %s'
                 % (len(w), d, '\n   '.join(w)))
        self.tabs.setCurrentIndex(3)

    # -- markers and image export -----------------------------------------

    @user_action
    def on_label_peaks(self):
        """Annotate every detected peak with its best identification.

        Intended for producing a figure. Each marker sits where the assigned
        line falls under the current calibration, so a mislabelled or
        miscalibrated peak is visible as a guide that misses.
        """
        s = self.session
        if s.spectrum is None:
            self.say('Nothing loaded.')
            return
        o = s.options
        pk = lineid.find_peaks(s.spectrum, s.cal, background=s._bk,
                               e_low=o.e_low, e_high=o.e_high, min_sigma=6.0)
        if not pk:
            self.say('No peaks above 6 sigma.')
            return
        # prefer an identification among the elements actually being fitted -
        # if the fit says it is iron, do not label it something else
        fitted = set()
        for Z, sh in self.selected_elements():
            fitted.add((s.db.sym[Z], sh))
        marks, unknown = [], 0
        for e, net, sg in pk:
            cands = lineid.identify(s.db, e, window_eV=70.0, limit=6)
            if not cands:
                unknown += 1
                continue
            pick = next((c for c in cands if (c.symbol, c.shell) in fitted),
                        cands[0])
            col = '#0F766E' if (pick.symbol, pick.shell) in fitted else '#A8323F'
            marks.append((pick.label(), pick.energy, col))
        self.spec.set_markers(marks)
        self.say('Labelled %d peaks (%d unidentified). Teal = an element in '
                 'the current fit, red = suggested by the line table only.'
                 % (len(marks), unknown))

    @user_action
    def on_artefacts(self):
        """Mark predicted escape and sum peaks.

        These are the two places a spectrum has a real, sharp feature that no
        element produced. Knowing where they are before choosing elements is
        the cheapest way to avoid fitting a fake line - the 2.24 keV Si+O sum
        in a quartz spectrum was otherwise being handed to mercury.
        """
        s = self.session
        if s.spectrum is None:
            self.say('Nothing loaded.')
            return
        els = self.selected_elements()
        if not els:
            self.say('Select some elements first - artefacts are predicted '
                     'from the lines you expect.')
            return
        o = s.options

        def res(E):
            return float(np.sqrt(o.noise ** 2 + o.fano ** 2 * max(E, 0)))

        # Prefer the model the fit is actually using over a fresh estimate.
        # Escape and sum peaks are fitted components now, so the honest thing
        # to show is what the fit did with them, not a separate prediction
        # that may disagree with the curve already on screen.
        em = s.escape_model if o.use_escape_peaks else None
        fitted_pileup = s.areas().get('pileup')
        esc, sums, p = artefacts.predict(
            s.db, s.spectrum, s.cal, els, e_low=o.e_low, e_high=o.e_high,
            areas=s.areas() or None, resolution=res,
            escape_fraction=(em.fraction if em else None),
            escape_fraction_beta=((lambda E: em.fraction(E, beta=True))
                                  if em else None),
            crystal_Z=o.crystal_Z)
        crystal = s.db.sym.get(o.crystal_Z, 'Z=%d' % o.crystal_Z)
        e_esc, k_edge = artefacts.crystal_escape(s.db, o.crystal_Z)
        marks = [('%s' % a.label, a.energy, '#7B3FA8') for a in esc]
        marks += [('%s' % a.label, a.energy, '#E07A00') for a in sums]
        if not marks:
            if e_esc > 0:
                why = ('Escape peaks need a parent above the %s K edge at '
                       '%.3f keV.' % (crystal, k_edge))
            else:
                why = 'A %s crystal has no escape peaks.' % crystal
            self.say('No escape or sum peaks fall inside %.2f-%.2f keV. %s'
                     % (o.e_low, o.e_high, why))
            return
        self.spec.set_markers(marks)
        rows = []
        if fitted_pileup is not None:
            tot = float(np.sum(s.spectrum))
            rows.append('pile-up FITTED: %.0f counts (%.3g%% of the spectrum)'
                        % (fitted_pileup, 100 * fitted_pileup / max(tot, 1)))
        if em is not None:
            rows.append('escape peaks FITTED, tied to their parents. %s '
                        'prefactor omega_K(1-1/r)/2 = %.4f, gamma = %.3g'
                        % (crystal, em.prefactor, em.gamma))
        if p is not None:
            rows.append('pile-up fraction inferred from the data: %.4g' % p)
        elif fitted_pileup is None:
            rows.append('pile-up fraction could not be measured - no clean '
                        'sum peak. Positions below are still valid; the '
                        'predicted counts are not.')
        if esc:
            rows.append('')
            rows.append('%s escape peaks (purple):' % crystal)
            for a in esc:
                rows.append('   %-14s %7.4f keV   ~%.0f counts'
                            % (a.label, a.energy, a.intensity))
        if sums:
            rows.append('')
            rows.append('sum peaks (orange):')
            for a in sorted(sums, key=lambda x: -(x.intensity or 0))[:12]:
                extra = ('   FWHM ~%.0f eV (a real line here would be %.0f)'
                         % (a.fwhm_eV, res(a.energy))) if a.fwhm_eV else ''
                rows.append('   %-14s %7.4f keV   ~%.0f counts%s'
                            % (a.label, a.energy, a.intensity or 0, extra))
        rows.append('')
        rows.append('A feature at one of these energies needs no element. '
                    'Tick "Model sum-peak pile-up" to have the fit account '
                    'for the sums directly.')
        self.say(chr(10).join(rows))
        self.tabs.setCurrentIndex(3)

    @user_action
    def on_save_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save the spectrum view', '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:
            return
        s = self.session
        title = s.label + (' [%s]' % s.mask_name if s.mask_name else '')
        if s.fit is not None:
            title += '   chi2r = %.2f' % s.fit.reduced_chi2
        try:
            self.spec.save_image(path, dpi=200, title=title)
        except Exception as ex:
            self.say('SAVE IMAGE FAILED: %s' % ex)
            return
        self.say('Wrote %s at 200 dpi - exactly the view on screen, markers '
                 'and zoom included.' % os.path.basename(path))

    # -- element lists ----------------------------------------------------

    def rebuild_element_lists(self):
        """Tell the element dialog what the current energy window allows.

        An element with no line in the window is greyed out, and a selection
        that becomes impossible is dropped. Widening the range to 12 keV makes
        the transition-metal K lines available; narrowing to 3 keV removes
        them again. The table always shows what is actually detectable.
        """
        o = self.session.options
        try:
            self.elements.set_range(o.e_low, o.e_high)
        except Exception as ex:
            self.say('Cannot rebuild the element table: %s' % ex)
            return
        self._range_shown = (o.e_low, o.e_high)
        # map lines, not elements: 'Ag' alone mapped Ag K, 22 keV, off a
        # 0.2-6.5 keV spectrum, as an all-zero map
        s = self.session
        lines = []
        for b in sorted(self.elements.table.buttons.values(),
                        key=lambda x: x.Z):
            for sh in sorted(b.available):
                if o.e_low <= s.map_line_energy(b.Z, sh) <= o.e_high:
                    lines.append(s.map_line_name(b.Z, sh))
        self.cmb_mapel.clear()
        self.cmb_mapel.addItems(lines)
        n = sum(len(b.available)
                for b in self.elements.table.buttons.values())
        self.say('Element table set for %.2f-%.2f keV: %d element-shell '
                 'combinations available.' % (o.e_low, o.e_high, n))

    @user_action
    def on_choose_elements(self):
        self.elements.show()
        self.elements.raise_()
        self.elements.activateWindow()

    @user_action
    def _on_elements_changed(self):
        self.lbl_els.setText('selected: ' + self.elements.table.summary())
        self.mark_changed()

    @user_action
    def on_load_dam(self):
        """Attach a DA matrix to the data already open.

        Separate from File > Open, which would treat the .dam as the thing
        being opened and pull in ITS list-mode file, replacing whatever is
        loaded. A matrix is built once from a good fit and then applied to
        other runs from the same session, so attaching it to the current data
        is the normal case and deserves its own entry.
        """
        s = self.session
        if s.events is None:
            self.say('Open the list-mode file first - a DA matrix is applied '
                     'to events.')
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open a GeoPIXE DA matrix', '', 'DA matrix (*.dam);;All (*)')
        if not path:
            return
        try:
            s.load_dam(path, keep_data=True)
        except Exception as ex:
            self.say('DA LOAD FAILED: %s' % ex)
            return
        els = s.dam_elements()
        self.say('DA matrix loaded: %d elements (%s).%s  Calibration left as '
                 'it was. Analysis > Concentration profiles now gives real '
                 'deconvolved concentrations.'
                 % (len(els), ', '.join(str(e) for e in els[:14]),
                    ' ...' if len(els) > 14 else ''))

    @user_action
    def on_profiles(self):
        """Traverses and regions on the element maps.

        Needs maps, and there are two ways to have them. A loaded DA matrix
        gives proper deconvolved concentrations and is what should be used
        when one exists; without it, simple window maps are offered instead,
        with a warning, because a window map counts everything under the
        window and an overlapped element will read high.
        """
        s = self.session
        if s.events is None:
            self.say('Profiles need a list-mode file - the maps come from '
                     'event positions. Open the .lmf.')
            return
        if not hasattr(self, 'profileview'):
            self.profileview = ProfileDialog(s, self)
        try:
            if s.dam is not None:
                maps = s.da_maps(binning=self.spin_bin.value()
                                 if hasattr(self, 'spin_bin') else 2)
                note = 'DA matrix: %d elements, overlaps deconvolved.' % len(maps)
            else:
                els = self.selected_elements()
                if not els:
                    self.say('Select some elements, or load a DA matrix.')
                    return
                maps = {}
                for Z, sh in els:
                    nm = s.db.sym[Z] + {1: '', 2: 'L', 3: 'M'}[sh]
                    e = s.db.line_energy(Z, sh)
                    if not (s.options.e_low < e < s.options.e_high):
                        continue
                    try:
                        maps[nm] = s.element_map(e, binning=2)
                    except Exception:
                        continue
                note = ('WINDOW maps, no DA matrix loaded. A window counts '
                        'everything under it, so an overlapped element reads '
                        'high. Load a .dam for real concentrations.')
        except Exception as ex:
            self.say('could not build maps: %s' % ex)
            return
        if not maps:
            self.say('no maps could be built')
            return
        self.profileview.set_maps(maps)
        self.profileview.out.setPlainText(
            note + chr(10) + 'Drag on the map to place a traverse.')
        self.profileview.show()
        self.profileview.raise_()
        self.profileview.activateWindow()

    @user_action
    def on_layers(self):
        if not hasattr(self, 'layerview'):
            self.layerview = LayerStackDialog(self.session, self)
        self.layerview.show()
        self.layerview.raise_()
        self.layerview.activateWindow()

    @user_action
    def on_sample_model(self):
        self.sample.show()
        self.sample.raise_()
        self.sample.activateWindow()

    def _sample_summary(self):
        s = self.session
        if self.sample.use_yields() and s.yield_set is not None:
            return 'yield files: %s' % s.yield_set.describe(s.yield_region)
        m = self.sample.matrix()
        t = self.sample.thickness()
        return ('matrix %s, %s, %.3g MeV, %.0f deg'
                % ('typed (%d elements)' % len(m) if m else 'bootstrapped',
                   'thick target' if t is None else '%.4g mg/cm2' % t,
                   self.sample.beam(), self.sample.theta()))

    # -- line identification ---------------------------------------------

    @user_action
    def on_spec_click(self, event):
        """Click anywhere on the spectrum to see what lines sit there.

        This is the tool you reach for when the residual shows something you
        did not put in the model. It lists candidates, ranked - it does not
        decide, because at any energy in a light-element spectrum there are
        several honest answers and the fit is what distinguishes them.
        """
        if not self.chk_id.isChecked():
            return
        if event.inaxes is None or event.xdata is None:
            return
        if self.session.spectrum is None:
            return
        e = float(event.xdata)
        cands = lineid.identify(self.session.db, e, window_eV=80.0, limit=8)
        if not cands:
            self.say('%.4f keV - nothing tabulated within 80 eV' % e)
            return
        lines = ['%.4f keV - candidate lines:' % e]
        for c in cands:
            mark = '  <' if abs(c.delta_eV) < 15 else ''
            lines.append('   %-9s %8.4f keV  %+5.0f eV   rel.int %.3f%s'
                         % (c.label(), c.energy, c.delta_eV, c.intensity, mark))
        self.say(chr(10).join(lines))
        # draw where each candidate WOULD fall under the current calibration.
        # A guide that misses the peak means either the identification or the
        # calibration is wrong, and seeing that is the whole point.
        cols = ['#A8323F', '#0F766E', '#B8860B', '#2B6CB0', '#7C3AED']
        self.spec.set_markers([(c.label(), c.energy, cols[i % len(cols)])
                               for i, c in enumerate(cands[:5])])
        self.status.showMessage('%.4f keV: %s' % (e, ', '.join(
            c.label() for c in cands[:4])), 15000)

    @user_action
    def on_identify(self):
        """Find every significant peak and name the best candidate for each."""
        s = self.session
        if s.spectrum is None:
            self.say('Nothing loaded.')
            return
        o = s.options
        pk = lineid.find_peaks(s.spectrum, s.cal, background=s._bk,
                               e_low=o.e_low, e_high=o.e_high, min_sigma=6.0)
        if not pk:
            self.say('No peaks above 6 sigma found.')
            return
        rows = ['%d peaks above 6 sigma:' % len(pk)]
        for e, net, sg in pk:
            c = lineid.identify(s.db, e, limit=3)
            names = ', '.join(x.label() for x in c) if c else '(unidentified)'
            rows.append('   %7.3f keV  net %9.0f  %6.1f sig   %s'
                        % (e, net, sg, names))
        self.say(chr(10).join(rows))
        self._peaks = pk
        self.tabs.setCurrentIndex(3)

    @user_action
    def on_suggest(self):
        """Highlight the elements whose lines explain the observed peaks.

        Scoring rejects coincidences by insisting an element's STRONGEST
        visible line is present and penalising lines that should be there and
        are not - otherwise a dense heavy-element L series matches anything.
        Matches are weighted by peak size, so explaining the dominant peak
        counts for more than clipping a small one.

        Treat the result as a shortlist. It is a starting point for a fit,
        not a result.
        """
        s = self.session
        if s.spectrum is None:
            self.say('Nothing loaded.')
            return
        o = s.options
        pk = lineid.find_peaks(s.spectrum, s.cal, background=s._bk,
                               e_low=o.e_low, e_high=o.e_high, min_sigma=6.0)
        sug = lineid.suggest_elements(s.db, pk, limit=18, min_score=0.4)
        if not sug:
            self.say('No elements scored above threshold.')
            return
        self._suggested = sug
        self._highlight_suggested(sug)
        rows = ['From %d peaks, %d candidate elements:' % (len(pk), len(sug))]
        for sym, sh, sc, n in sug:
            rows.append('   %-3s %s-shell   score %5.2f   %d line%s matched'
                        % (sym, {1: 'K', 2: 'L', 3: 'M'}[sh], sc, n,
                           '' if n == 1 else 's'))
        rows.append('Highlighted in the element lists. These are candidates, '
                    'not conclusions - add them and let the fit decide.')
        self.say(chr(10).join(rows))
        top = ', '.join('%s%s' % (sym, {1: '', 2: 'L', 3: 'M'}[sh])
                        for sym, sh, _, _ in sug[:8])
        self.lbl_sugg.setText('suggested: ' + top)

    def _highlight_suggested(self, sug):
        self.elements.set_suggested(sug)

    @user_action
    def on_accept_suggested(self):
        """Add every suggested element-shell to the fit selection."""
        if not self._suggested:
            self.say('Run "Suggest from spectrum" first.')
            return
        have = {}
        for Z, sh in self.elements.selection():
            have.setdefault(Z, set()).add(sh)
        added, skipped = 0, []
        for sym, sh, sc, n in self._suggested:
            Z = self.session.db.z.get(sym.lower())
            btn = self.elements.table.buttons.get(sym)
            if Z is None or btn is None or sh not in btn.available:
                skipped.append('%s%s' % (sym, {1: 'K', 2: 'L', 3: 'M'}[sh]))
                continue
            if sh not in have.get(Z, ()):
                have.setdefault(Z, set()).add(sh)
                added += 1
        self.elements.set_selection(
            [(Z, sh) for Z, shs in have.items() for sh in shs])
        msg = 'Added %d suggested element-shells.' % added
        if skipped:
            msg += (' Outside the %.2f-%.2f keV window: %s.'
                    % (self.session.options.e_low,
                       self.session.options.e_high, ', '.join(skipped)))
        msg += (' Prune anything you cannot justify - every spurious '
                'component is another way for the fit to explain a feature '
                'with the wrong element.')
        self.say(msg)

    # -- efficiency and quantification ----------------------------------

    @user_action
    def on_eff_changed(self, idx):
        path = self.cmb_eff.itemData(idx)
        if not path:
            self.session.efficiency = None
            self.say('Efficiency cleared - areas only, no concentrations.')
            self.mark_changed()
            return
        try:
            e = self.session.load_efficiency(path)
        except Exception as ex:
            self.say('EFFICIENCY FAILED: %s' % ex)
            return
        self.say('Efficiency: %s  (C Ka %.4f, Si Ka %.3f, Fe Ka %.3f)'
                 % (e.name, e(0.277), e(1.740), e(6.404)))
        self.mark_changed()

    @user_action
    def on_load_eff(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Detector efficiency curve', '', 'Text (*.txt);;All (*)')
        if not path:
            return
        self.cmb_eff.addItem(os.path.basename(path), path)
        self.cmb_eff.setCurrentIndex(self.cmb_eff.count() - 1)

    @user_action
    def on_axes(self):
        """Apply the typed axis limits to the spectrum plot."""
        ax = self.spec.ax
        o = self.session.options
        x0 = self._f(self.ed_x0, o.e_low)
        x1 = self._f(self.ed_x1, o.e_high)
        if x1 > x0:
            ax.set_xlim(x0, x1)
        y1 = self._f(self.ed_y1)
        if y1 and y1 > 0:
            lo = 0.5 if ax.get_yscale() == 'log' else 0.0
            ax.set_ylim(lo, y1)
        self.spec.draw_idle()

    @user_action
    def on_axes_reset(self):
        for w in (self.ed_x0, self.ed_x1, self.ed_y1):
            w.clear()
        self.sessionChanged.emit()

    def _autorange(self, force=False):
        """Set the fit range from the data when the current one does not fit.

        Only moves when it has to: if the present range already covers most
        of the counts it is left alone, because an operator who narrowed it
        deliberately should not have it silently widened again.
        """
        s = self.session
        if s.spectrum is None:
            return
        y = np.asarray(s.spectrum, float)
        a, b = s.cal
        E = a * np.arange(len(y)) + b
        tot = y.sum()
        if tot <= 0:
            return
        o = s.options
        inside = y[(E >= o.e_low) & (E <= o.e_high)].sum() / tot
        if inside > 0.60 and not force:
            return

        # the span holding the counts, trimmed of empty ends
        nz = np.nonzero(y > 0)[0]
        if len(nz) < 2:
            return
        lo = max(float(E[nz[0]]), a)          # never below one channel
        hi = float(E[nz[-1]])
        # ignore a sparse high tail: stop where 99.9% of the counts are in
        c = np.cumsum(y) / tot
        j = int(np.searchsorted(c, 0.999))
        hi = min(hi, float(E[min(j + 20, len(E) - 1)]))
        if hi <= lo:
            return
        o.e_low, o.e_high = round(lo, 4), round(hi, 4)
        self.ed_elo.setText('%.4g' % o.e_low)
        self.ed_ehi.setText('%.4g' % o.e_high)
        self.say('  fit range set to %.3f - %.3f keV (the previous range '
                 'held only %.1f%% of the counts)'
                 % (o.e_low, o.e_high, 100 * inside))
        s.invalidate()

    def _f(self, edit, default=None):
        t = edit.text().strip()
        if not t:
            return default
        try:
            return float(t)
        except ValueError:
            return default

    @user_action
    def on_load_detector(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open GeoPIXE detector file', '',
            'GeoPIXE detector (*.detector);;All files (*)')
        if not path:
            return
        try:
            det = self.session.load_detector(path)
        except Exception as ex:
            self.say('detector file failed: %s' % ex)
            return
        self.show_geometry()
        self.say('detector: %s' % os.path.basename(path))
        self.mark_changed()

    def show_geometry(self):
        """Push session.geometry back into the boxes, for values that were
        set from a file rather than typed (load_detector fills the active
        area and the distance)."""
        g = self.session.geometry
        for ed, v in ((self.ed_dist, getattr(g, 'distance_mm', None)),
                      (self.ed_area, getattr(g, 'area_mm2', None)),
                      (self.ed_charge, getattr(g, 'charge_uC', None))):
            if v and not ed.text().strip():
                # 6 figures, not 4: _geom_changed reads these boxes straight
                # back into the geometry, so what is shown IS the value used.
                ed.setText('%.6g' % v)
        self._geom_changed()
        det = self.session.detector
        if det and getattr(self, 'lbl_det', None):
            self.lbl_det.setText(
                'detector file: %s\ncrystal Z %d, %.0f um; w0 %.5f w1 %.5f'
                % (os.path.basename(det.get('path') or '?'),
                   self.session.options.crystal_Z,
                   self.session.options.crystal_thick_um,
                   det.get('w0') or 0.0, det.get('w1') or 0.0))

    def _geom_changed(self, *_):
        """Keep session.geometry in step with the boxes, and show the solid
        angle as soon as it can be worked out - seeing it appear is the
        cheapest check that the distance and area were entered sanely."""
        from ..physics.geometry import live_fraction
        g = self.session.geometry
        g.charge_uC = self._f(self.ed_charge)
        g.distance_mm = self._f(self.ed_dist)
        g.area_mm2 = self._f(self.ed_area)
        g.tilt_deg = self._f(self.ed_tilt, 0.0) or 0.0
        g.live_fraction = live_fraction(dead_percent=self._f(self.ed_dead, 0.0))
        o = g.solid_angle_msr
        if o:
            self.lbl_omega.setText(
                'solid angle: %.3f msr   (%.4f%% of 4pi)'
                % (o, 100 * o * 1e-3 / 12.566))
        else:
            self.lbl_omega.setText('solid angle: -')

    @user_action
    def on_quantify(self):
        s = self.session
        if s.fit is None:
            self.say('Fit first.')
            return
        if s.efficiency is None:
            self.say('Select a detector efficiency curve first - without one '
                     'a peak area cannot become a concentration.')
            return
        self._geom_changed()
        if self.sample.use_yields() and s.yield_set is not None:
            self._quantify_yields()
            return
        absolute = s.geometry.complete()
        try:
            s.quantify(matrix=self.sample.matrix(),
                       thickness=self.sample.thickness(),
                       beam_MeV=self.sample.beam(),
                       theta_deg=self.sample.theta(),
                       normalise=not absolute, absolute=absolute)
        except Exception as ex:
            self.say('QUANTIFY FAILED: %s' % ex)
            return
        self.lbl_sm.setText(self._sample_summary())
        rows = s.concentration_table()
        body = chr(10).join('   %-3s %8.3f wt%%  +-%.1f%%' % r for r in rows)
        if not absolute:
            miss = ', '.join(s.geometry.missing())
            self.say('Concentrations, NORMALISED to 100 wt%% over K-shell '
                     'elements:' + chr(10) + body + chr(10) + chr(10) +
                     'Normalised, so the total is 100 by construction and '
                     'cannot test the fit. Fill in ' + miss + ' for an '
                     'absolute total, which is the one number that catches a '
                     'fit that has quietly lost intensity.')
        else:
            tot = s._conc_sum or 0.0
            if tot < 90:
                verdict = ('LOW. Intensity is unaccounted for: an element '
                           'missing from the fit, dead time not corrected, '
                           'or the charge reading high.')
            elif tot > 110:
                verdict = ('HIGH. Something is counted twice, or the charge '
                           'reads low, or the solid angle is overstated.')
            else:
                verdict = 'consistent - the chain holds together.'
            self.say('ABSOLUTE concentrations (no normalisation):'
                     + chr(10) + body + chr(10) + chr(10) +
                     'total %.1f wt%%  -  %s' % (tot, verdict) + chr(10) +
                     s.geometry.describe())
        self.say('Yields: ' + s.quantify_basis)
        self.fill_table(s.fit)
        if s._conc:
            was_boot = self.sample.rb_boot.isChecked()
            self.sample.fill_from(s._conc, s.db)
            if was_boot:
                self.sample.rb_boot.setChecked(True)
        self.tabs.setCurrentIndex(2)

    def _quantify_yields(self):
        """QUANTIFY from the sample's .yield files, GeoPIXE's way."""
        s = self.session
        try:
            rows = s.quantify_yields()
        except Exception as ex:
            self.say('QUANTIFY FAILED: %s' % ex)
            return
        q, frac = getattr(s, '_yield_charge', (None, 1.0))
        g = s.geometry
        unit = 'ppm' if q else 'ppm.uC.msr'
        out = ['Concentrations from the .yield files, %s by weight IN THE '
               'LAYER named (each file gives the elements of its UNKNOWN '
               'layer):' % unit]
        for r in sorted(rows, key=lambda r: (r['layer'].lower(),
                                             -(r['ppm'] if r['ppm'] == r['ppm']
                                               else 0))):
            shell = {1: 'K', 2: 'L', 3: 'M'}.get(r['shell'], '?')
            if r['note']:
                out.append('   %-3s %s  %-11s  %s   %s' % (
                    r['element'], shell, r['layer'], r['note'],
                    os.path.basename(r['file']) if r['file'] else ''))
            else:
                out.append('   %-3s %s  %-11s %12.5g +- %-9.3g %s' % (
                    r['element'], shell, r['layer'], r['ppm'], r['err'],
                    os.path.basename(r['file'])))
        out.append('')
        out.append('Yields: ' + s.quantify_basis)
        if q:
            out.append('charge %.4g uC (entered %.4g x live %.4g%s), solid '
                       'angle %.4g msr' % (
                           q, g.charge_uC, g.live_fraction,
                           ' x region pixel fraction %.4f' % frac
                           if s.mask is not None else '',
                           g.solid_angle_msr))
        else:
            out.append('No charge or solid angle (fill in %s): the values '
                       'are ppm x uC x msr. Ratios within one layer hold.'
                       % ', '.join(g.missing()))
        self.say(chr(10).join(out))
        self.lbl_sm.setText(self._sample_summary())
        self.fill_table(s.fit)
        self.tabs.setCurrentIndex(2)

    # -- mask tools ------------------------------------------------------

    @user_action
    def on_mode_changed(self, mode):
        self.map.set_tool(mode)
        extra = {'flood fill': '  (click a pixel to seed)',
                 'polygon': '  (click corners; click the first to close)'
                 }.get(mode, '')
        self.say('Mask tool: %s%s' % (mode, extra))

    @user_action
    def on_export_map(self):
        data = self.map._data
        meta = getattr(self, '_map_meta', None)
        if data is None or meta is None:
            self.say('Show a map first.')
            return
        s = self.session
        stem = '%s_%s_bin%d' % (meta['stem'] or 'map', meta['element'],
                                meta['binning'])
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export the shown map', stem + '.csv', 'CSV (*.csv)')
        if not path:
            return
        written = export_map_csv(path, data, s, meta)
        self.say('Wrote %s.' % ', '.join(os.path.basename(p) for p in written))

    @user_action
    def on_map_reset(self):
        for sp, v in ((self.spin_plo, MAP_PCT[0]), (self.spin_phi, MAP_PCT[1])):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self.map.reset_view()

    @user_action
    def on_map_display(self, *_):
        lo, hi = self.spin_plo.value(), self.spin_phi.value()
        if lo >= hi:
            self.say('Colour scale: low must be below high (%g >= %g).'
                     % (lo, hi))
            return
        self.map.set_display(cmap=self.cmb_cmap.currentText(),
                             scale=self.cmb_scale.currentText(),
                             pct=(lo, hi))

    @user_action
    def on_map_click(self, event):
        if self.cmb_mode.currentText() != 'flood fill':
            return
        if self.map_toolbar.mode:        # zooming or panning, not seeding
            return
        if event.inaxes is not self.map.ax or event.xdata is None:
            return
        data = self.map._data
        if data is None:
            return
        m = MSK.flood(data, (self.map._pix(event.xdata),
                             self.map._pix(event.ydata)),
                      tolerance=self.spin_tol.value())
        if not m.any():
            self.say('Flood found nothing there - widen the tolerance.')
            return
        self._apply_mask(m, 'flood')

    @user_action
    def on_morph(self, direction):
        # One step is one DISPLAYED pixel, taken from the session's mask
        # whichever tool made it. A full-resolution pixel is a quarter of a
        # displayed one at the default 4x binning: no visible change.
        base = self.session.mask
        if base is None:
            self.say('Make a mask first.')
            return
        step = self.map.binning()
        m = (MSK.grow(base, step, connectivity=8) if direction > 0
             else MSK.shrink(base, step, connectivity=8))
        if not m.any():
            self.say('Shrink would leave nothing - mask kept.')
            return
        self._apply_mask(m, self.session.mask_name or 'mask', combine=False)

    def _apply_mask(self, m, name, combine=True):
        """Make m (map- or raster-shaped) the session mask, or with the
        mask op set, add it to or take it from the current one."""
        op = self.cmb_op.currentText() if combine else 'replace'
        cur = self.session.mask
        if op == 'subtract' and cur is None:
            # nothing set means the whole field is analysed, so cutting a
            # shape out of it leaves everything else - never the shape
            # itself (our choice; not checked against GeoPIXE's exclude)
            cur = np.ones((256, 256), dtype=bool)
        if op != 'replace' and cur is not None:
            f = cur.shape[0] // m.shape[0]
            r = np.repeat(np.repeat(np.asarray(m, bool), f, 0), f, 1)
            if r.shape != cur.shape:
                self.say('MASK FAILED: a %dx%d map does not tile the %dx%d '
                         'raster' % (m.shape + cur.shape))
                return
            m = MSK.combine(cur, r, 'or' if op == 'add' else 'sub')
            if self.session.mask is not None and (m == cur).all():
                return       # e.g. a stray click re-sending the same shape
            name = '%s%s%s' % (self.session.mask_name or 'field',
                               '+' if op == 'add' else '-', name)
            if not m.any():
                self.say('That would leave an empty mask - mask kept.')
                return
        try:
            n = self.session.set_mask(m, name)
        except Exception as ex:
            self.say('MASK FAILED: %s' % ex)
            return
        self.map.set_mask(self.session.mask)
        self.say('%s mask: %s, %d events. Refit to analyse it.'
                 % (name, MSK.describe(self.session.mask), n))
        self.mark_changed()

    # -- batch -----------------------------------------------------------

    @user_action
    def on_batch(self):
        els = self.selected_elements()
        if not els:
            self.say('Select elements first - a batch applies one set to '
                     'every file.')
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Choose files to process', '',
            'All supported (*.lmf *.dam *.spec *.txt);;All files (*)')
        if not paths:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Where should the results go?')
        if not out:
            return
        self.apply_options()
        cal = None
        ask = QtWidgets.QMessageBox.question(
            self, 'Calibration',
            'Force the current calibration on every file?' + chr(10) + chr(10)
            + 'Usually yes - they were acquired on one setup, and it removes '
              'a per-file variable from the comparison.',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ask == QtWidgets.QMessageBox.Yes:
            cal = self.session.cal
        eff = self.cmb_eff.itemData(self.cmb_eff.currentIndex())
        log_event('batch request', paths=list(paths), out=out,
                  calibration=cal, efficiency=eff)
        opts, adc, n = self.session.options, self.spin_adc.value(), len(paths)
        self.batch_done, self.last_batch = 0, None
        self._preview_timer.stop()

        def work(ctx):
            def tick(i, total, path):
                ctx.progress(os.path.basename(path), i, total)
            return run_batch(paths, els, out, options=opts, calibration=cal,
                             efficiency=eff, adc=adc, quantify=bool(eff),
                             progress=tick, should_stop=ctx.should_stop)

        def done(res):
            self._set_busy(None)
            self.last_batch = res
            self.batch_done = len(res)
            head = ('Batch CANCELLED after %d of %d files.' % (len(res), n)
                    if len(res) < n else 'Batch complete.')
            self.say(head + chr(10) + summarise(res) + chr(10)
                     + 'Summary table: %s'
                     % os.path.join(out, 'batch_summary.csv'))
            self.tabs.setCurrentIndex(3)

        def failed(tb):
            self._set_busy(None)
            self.say('BATCH FAILED: %s' % tb.strip().splitlines()[-1])
            self.log.appendPlainText(tb)

        def cancelled():
            self._set_busy(None)
            self.say('Batch cancelled.')

        if self.jobs.submit(Job('batch', work, on_done=done, on_fail=failed,
                                on_cancel=cancelled)):
            self._set_busy('batch')

    # -- interop and live -------------------------------------------------

    @user_action
    def on_export_dam(self):
        s = self.session
        if s.fit is None:
            self.say('Fit first - a DA matrix is built from the fitted '
                     'component shapes.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Write DA matrix', '', 'GeoPIXE DA matrix (*.dam)')
        if not path:
            return
        try:
            write_dam_from_session(s, path)
        except Exception as ex:
            self.say('DAM EXPORT FAILED: %s' % ex)
            return
        units = 'ppm' if s.efficiency is not None else 'area units'
        self.say('Wrote %s in %s. GeoPIXE can load this and project maps '
                 'with it.' % (os.path.basename(path), units))

    @user_action
    def on_export_yield(self):
        """Write a GeoPIXE .yield for the current sample model.

        Version -3: the newest whose content is fully determined by things we
        actually know. From -8 the format embeds an IDL beam struct whose
        exact byte layout would have to be guessed, and a calibration file
        that loads but is subtly wrong is the worst possible outcome. The MAC
        provenance field only exists from -12, so the real settings go into a
        sidecar .provenance.json rather than being silently lost.
        """
        s = self.session
        if s.fit is None:
            self.say('Fit first - the element list comes from the fit.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Write yield file', '', 'GeoPIXE yield (*.yield)')
        if not path:
            return
        try:
            write_yield_from_session(
                s, path, matrix=self.sample.matrix(),
                thickness=self.sample.thickness(),
                beam_MeV=self.sample.beam(), theta_deg=self.sample.theta())
        except Exception as ex:
            self.say('YIELD EXPORT FAILED: %s' % ex)
            return
        self.set_session_yield(path, 'exported')
        self.say('Wrote %s (format version -3) plus a .provenance.json '
                 'recording the MAC dataset, fluorescence yields and '
                 'efficiency curve, which version -3 cannot store.'
                 % os.path.basename(path))

    @user_action
    def on_attach_live(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Attach to a list-mode file being written', '',
            'OMDAQ list mode (*.lmf)')
        if not path:
            return
        try:
            self._live = LiveLMF(path)
            self._live.poll()
            refresh_session(self.session, self._live, self.spin_adc.value())
        except Exception as ex:
            self.say('ATTACH FAILED: %s' % ex)
            return
        self.lbl_file.setText('%s [LIVE]' % os.path.basename(path))
        self.say('Attached: %s' % self._live.status())
        if not hasattr(self, '_live_timer'):
            self._live_timer = QtCore.QTimer(self)
            self._live_timer.timeout.connect(self._live_tick)
        self._live_timer.start(1000)
        self.mark_changed()

    def _live_tick(self):
        r = getattr(self, '_live', None)
        if r is None or self.jobs.running_heavy():
            return                  # never swap the data under a running fit
        if r.poll():
            refresh_session(self.session, r, self.spin_adc.value())
            self.mark_changed()
            self.status.showMessage('LIVE: ' + r.status())

    @user_action
    def on_about(self):
        QtWidgets.QMessageBox.about(
            self, 'scrappyFIT',
            '<b>scrappyFIT %s</b><br><br>'
            'Light-element PIXE fitting and mapping.<br><br>'
            'Reads OMDAQ list-mode and GeoPIXE formats. Adds selectable mass '
            'attenuation datasets, rebuilt L and M line data, and detector '
            'response terms that GeoPIXE does not model - without which a fit '
            'can invent elements the sample does not contain.<br><br>'
            'Database: %s' % (__version__,
                              config.database_path(required=False)))


def main(argv=None):
    """python -m scrappyfit [file] [--elements C,N,O,...] [--fit] [--cal g,o]

    The optional arguments exist so a spectrum can be put on screen already
    loaded, already populated with elements, and already fitted - which is
    what you want when someone else is going to watch the result rather than
    drive the dialogs themselves.
    """
    import argparse

    argv = list(sys.argv if argv is None else argv)
    ap = argparse.ArgumentParser(prog='scrappyfit', add_help=True)
    ap.add_argument('path', nargs='?', help='.lmf, .spec, .dam or .txt')
    ap.add_argument('--elements', help='comma-separated symbols to select')
    ap.add_argument('--cal', help='gain,offset in keV per channel and keV')
    ap.add_argument('--eff', help='substring of a built-in efficiency curve')
    ap.add_argument('--range', help='fit range as lo,hi in keV')
    ap.add_argument('--detector', help='a GeoPIXE .detector file')
    ap.add_argument('--fit', action='store_true', help='fit once on startup')
    ns, rest = ap.parse_known_args(argv[1:])

    app = QtWidgets.QApplication([argv[0]] + rest)
    app.setApplicationName('scrappyFIT')
    from . import observe
    logpath = observe.install()
    win = MainWindow()
    win.show()
    win.say('Action log: %s' % logpath)

    if ns.path:
        if not os.path.exists(ns.path):
            win.say('no such file: %s' % ns.path)
        elif win.open_path(ns.path):
            if ns.cal:
                try:
                    g, o = (float(x) for x in ns.cal.split(','))
                    win.ed_gain.setText('%.7f' % g)
                    win.ed_off.setText('%.5f' % o)
                    win.on_cal_changed()
                except Exception as ex:
                    win.say('bad --cal (%s)' % ex)
            if ns.eff:
                try:
                    hit = [i for i in range(win.cmb_eff.count())
                           if ns.eff.lower() in win.cmb_eff.itemText(i).lower()]
                    if hit:
                        win.cmb_eff.setCurrentIndex(hit[0])
                except Exception:
                    pass
            if ns.detector:
                try:
                    win.session.load_detector(ns.detector)
                    win.show_geometry()
                    win.say('detector: %s' % ns.detector)
                except Exception as ex:
                    win.say('bad --detector (%s)' % ex)
            if ns.range:
                try:
                    lo, hi = (float(x) for x in ns.range.split(','))
                    win.ed_elo.setText('%.6g' % lo)
                    win.ed_ehi.setText('%.6g' % hi)
                    win.on_range_changed()
                    win.session.options.e_low = lo
                    win.session.options.e_high = hi
                    win.session.invalidate()
                    win.mark_changed()
                except Exception as ex:
                    win.say('bad --range (%s)' % ex)
            if ns.elements:
                # "Fe" means the K lines; "FeL" or "Fe:L" means the L lines
                db = win.session.db
                pairs = []
                for tok in ns.elements.split(','):
                    tok = tok.strip().replace(':', '')
                    if not tok:
                        continue
                    sh = 1
                    if len(tok) > 1 and tok[-1] in 'LM':
                        sh = 2 if tok[-1] == 'L' else 3
                        tok = tok[:-1]
                    Z = db.z.get(tok.lower())
                    if Z:
                        pairs.append((Z, sh))
                    else:
                        win.say('unknown element %r, skipped' % tok)
                if pairs:
                    win.elements.set_selection(pairs)
            if ns.fit:
                # after the event loop starts, so the window is painted and
                # visible before the fit begins rather than after it ends
                QtCore.QTimer.singleShot(400, win.on_fit)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
