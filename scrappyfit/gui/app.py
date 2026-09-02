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

import os
import sys
import traceback

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
from .canvases import MapCanvas, SpectrumCanvas, toolbar_for
from .dialogs import ElementDialog, SampleModelDialog
from .layerview import LayerStackDialog
from .profileview import ProfileDialog

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


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.session = Session()
        self.setWindowTitle('scrappyFIT %s' % __version__)
        self.resize(1500, 950)
        self.elements = ElementDialog(self.session.db, self)
        self.elements.selectionChanged.connect(self._on_elements_changed)
        self.sample = SampleModelDialog(self.session.db, self)
        self._build()
        self._refresh_db_label()
        self.rebuild_element_lists()

    # ------------------------------------------------------------ layout

    def _build(self):
        self._build_menu()
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._centre())
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1140])
        self.setCentralWidget(split)
        self.status = self.statusBar()
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
        f.addAction('&Quit', self.close, 'Ctrl+Q')
        a = m.addMenu('&Analysis')
        a.addAction('&Fit', self.on_fit, 'Ctrl+F')
        a.addAction('&Quantify', self.on_quantify, 'Ctrl+Shift+Q')
        a.addAction('Sample &model...', self.on_sample_model)
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
        self.lbl_db.setStyleSheet('color: #666; font-size: 10px;')
        gl.addRow('Database', self.lbl_db)
        v.addWidget(g)

        # -- calibration
        g = QtWidgets.QGroupBox('Calibration   E = offset + gain x channel')
        gl = QtWidgets.QFormLayout(g)
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
        self.lbl_els.setStyleSheet('font-size:11px; color:#0F766E;')
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
        self.lbl_sugg.setStyleSheet('color:#0F766E; font-size:10px;')
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
        self.lbl_sm.setStyleSheet('color:#666; font-size:10px;')
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
        note = QtWidgets.QLabel('Needed for concentrations. At C Ka a C1 '
                                'window transmits about 3%.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666; font-size:10px;')
        gl.addWidget(note)
        v.addWidget(g)

        # -- options
        g = QtWidgets.QGroupBox('Fit options')
        gl = QtWidgets.QFormLayout(g)
        o = self.session.options
        self.ed_elo = QtWidgets.QLineEdit(str(o.e_low))
        self.ed_ehi = QtWidgets.QLineEdit(str(o.e_high))
        self.cmb_mac = QtWidgets.QComboBox()
        self.cmb_mac.addItems(['mixed', 'ffast', 'henke1993', 'xcom',
                               'sabbatuccisalvat2016'])
        self.cmb_lines = QtWidgets.QComboBox()
        self.cmb_lines.addItems(['xray_lines_rebuilt.txt', 'xray_lines.txt'])
        self.cmb_lines.currentTextChanged.connect(
            lambda s: self.session.set_lines_file(s))
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
        gl.addRow('E low keV', self.ed_elo)
        gl.addRow('E high keV', self.ed_ehi)
        gl.addRow('MAC dataset', self.cmb_mac)
        gl.addRow('Line data', self.cmb_lines)
        for c in (self.chk_escape, self.chk_contact, self.chk_window,
                  self.chk_pileup, self.chk_nonneg):
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

        sc = QtWidgets.QScrollArea()
        sc.setWidget(w)
        sc.setWidgetResizable(True)
        sc.setMinimumWidth(340)
        return sc

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
        b_full = QtWidgets.QPushButton('Full range')
        b_full.clicked.connect(self.refresh_spectrum)
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
        b_art.setToolTip('Mark where silicon escape peaks and pile-up sum '
                         'peaks fall. Both put real features where no element '
                         'emits, and a fit offered one will reach for the '
                         'nearest element.')
        b_art.clicked.connect(self.on_artefacts)
        b_nomark = QtWidgets.QPushButton('Clear markers')
        b_nomark.clicked.connect(lambda: self.spec.clear_markers())

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
        for x in (cb_log, cb_cmp, cb_lab, b_full, self.chk_id, b_pk,
                  b_lab, b_art, b_nomark,
                  self.ed_x0, self.ed_x1, self.ed_y1, b_ax, b_axr):
            bar.addWidget(x)
        bar.addStretch(1)
        sv.addWidget(toolbar_for(self.spec, sw))
        sv.addLayout(bar)
        sv.addWidget(self.spec, 1)
        self.spec.mpl_connect('button_press_event', self.on_spec_click)
        self._suggested = []
        tabs.addTab(sw, 'Spectrum')

        # map tab
        mw = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mw)
        top = QtWidgets.QHBoxLayout()
        self.cmb_mapel = QtWidgets.QComboBox()
        self.cmb_mapel.setEditable(True)
        self.cmb_mapel.setToolTip('An element symbol, or a bare energy in keV')
        b_map = QtWidgets.QPushButton('Show map')
        b_map.clicked.connect(self.on_map)
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
        self.cmb_mode.addItems(['rectangle', 'flood fill'])
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
            top.addWidget(x)
        top.addWidget(self.chk_sub)
        top.addWidget(b_map)
        top.addWidget(b_thr)
        top.addStretch(1)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel('mask tool'))
        row2.addWidget(self.cmb_mode)
        row2.addWidget(QtWidgets.QLabel('flood tolerance'))
        row2.addWidget(self.spin_tol)
        row2.addWidget(b_grow)
        row2.addWidget(b_shrink)
        row2.addWidget(b_clr)
        row2.addStretch(1)
        self.map = MapCanvas()
        self.map.regionSelected.connect(self.on_region)
        mv.addLayout(top)
        mv.addLayout(row2)
        mv.addWidget(self.map, 1)
        self.map.mpl_connect('button_press_event', self.on_map_click)
        self._last_mask = None
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
        self.log.appendPlainText(msg)
        self.status.showMessage(msg, 8000)

    def _refresh_db_label(self):
        p = config.database_path(required=False)
        self.lbl_db.setText(str(p) if p else 'NOT FOUND - set one via File menu')

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
        self.session.invalidate()

    # ------------------------------------------------------------ actions

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open spectrum or list-mode file', '',
            'All supported (*.lmf *.dam *.spec *.txt);;'
            'OMDAQ list mode (*.lmf);;GeoPIXE DA matrix (*.dam);;'
            'Text (*.txt *.spec);;All files (*)')
        if not path:
            return
        self.open_path(path)

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
            QtWidgets.QMessageBox.critical(self, 'Load failed', str(ex))
            return False
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
        self._geom_changed()
        self.refresh_spectrum()
        return True

    def on_adc_changed(self):
        if self.session.path and self.session.path.lower().endswith('.lmf'):
            self.session.load(self.session.path, adc=self.spin_adc.value())
            self.refresh_spectrum()

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

    def on_cal_changed(self):
        if self.session.spectrum is None:
            return
        try:
            self.session.set_calibration(float(self.ed_gain.text()),
                                         float(self.ed_off.text()))
        except ValueError:
            return
        self.refresh_spectrum()

    def on_labels(self, on):
        self._want_labels = bool(on)
        self.refresh_spectrum()

    def on_range(self, lo, hi):
        self.spec.set_xlim(lo, hi)
        self.say('zoom %.3f - %.3f keV' % (lo, hi))

    def refresh_spectrum(self):
        s = self.session
        if s.spectrum is None:
            return
        title = s.label + (' [%s]' % s.mask_name if s.mask_name else '')
        o = s.options
        self.spec.show(s.energy, s.spectrum, fit=s.fit,
                       background=s._bk, labels=self.line_labels(),
                       title=title, xlim=(o.e_low, o.e_high))

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
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            r = s.run_fit(els)
        except Exception as ex:
            self.say('FIT FAILED: %s' % ex)
            self.log.appendPlainText(traceback.format_exc())
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.say('Fit: chi2red = %.3f over %d components%s'
                 % (r.reduced_chi2, len(r.names),
                    ' [%s]' % s.mask_name if s.mask_name else ''))
        self.fill_table(r)
        self.refresh_spectrum()

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

    def on_map(self):
        s = self.session
        if s.events is None:
            self.say('Maps need a list-mode (.lmf) file.')
            return
        el = self.cmb_mapel.currentText().strip()
        try:
            m = s.element_map(el, binning=self.spin_bin.value(),
                              subtract=self.chk_sub.isChecked())
        except Exception as ex:
            self.say('MAP FAILED: %s' % ex)
            return
        self.map.show_map(m, '%s   (%dx%d bin%s)'
                          % (el, self.spin_bin.value(), self.spin_bin.value(),
                             ', net' if self.chk_sub.isChecked() else ', raw'))
        self.tabs.setCurrentIndex(1)
        if self.chk_sub.isChecked() and np.nanmedian(m) < 0:
            self.say('WARNING: this net map is mostly negative. Flanking '
                     'subtraction fails where a weak line sits between strong '
                     'neighbours (Al between Mg and Si is the usual case). '
                     'Untick "subtract background" and read it as raw counts.')

    def on_region(self, mask):
        try:
            n = self.session.set_mask(mask, 'region')
        except Exception as ex:
            self.say('MASK FAILED: %s' % ex)
            return
        self.say('Region mask: %d pixels, %d events. Refit to analyse it.'
                 % (int(mask.sum()), n))
        self.refresh_spectrum()

    def on_threshold(self):
        m = self.map.threshold_mask(self.spin_thr.value())
        if m is None:
            self.say('Show a map first.')
            return
        try:
            n = self.session.set_mask(m, 'top%d' % self.spin_thr.value())
        except Exception as ex:
            self.say('MASK FAILED: %s' % ex)
            return
        self.say('Threshold mask (top %d%%): %d pixels, %d events.'
                 % (100 - self.spin_thr.value(), int(m.sum()), n))
        self.refresh_spectrum()

    def on_clear_mask(self):
        self.session.clear_mask()
        self.say('Mask cleared - back to the whole field.')
        self.refresh_spectrum()

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
            escape_fraction=(em.fraction if em else None))
        marks = [('%s' % a.label, a.energy, '#7B3FA8') for a in esc]
        marks += [('%s' % a.label, a.energy, '#E07A00') for a in sums]
        if not marks:
            self.say('No escape or sum peaks fall inside %.2f-%.2f keV. '
                     'Escape peaks need a parent above the Si K edge at '
                     '1.839 keV, so a purely light-element spectrum has none.'
                     % (o.e_low, o.e_high))
            return
        self.spec.set_markers(marks)
        rows = []
        if fitted_pileup is not None:
            tot = float(np.sum(s.spectrum))
            rows.append('pile-up FITTED: %.0f counts (%.3g%% of the spectrum)'
                        % (fitted_pileup, 100 * fitted_pileup / max(tot, 1)))
        if em is not None:
            rows.append('escape peaks FITTED, tied to their parents. Silicon '
                        'prefactor omega_K(1-1/r)/2 = %.4f, gamma = %.3g'
                        % (em.prefactor, em.gamma))
        if p is not None:
            rows.append('pile-up fraction inferred from the data: %.4g' % p)
        elif fitted_pileup is None:
            rows.append('pile-up fraction could not be measured - no clean '
                        'sum peak. Positions below are still valid; the '
                        'predicted counts are not.')
        if esc:
            rows.append('')
            rows.append('silicon escape peaks (purple):')
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
        syms = [b.symbol for b in
                sorted(self.elements.table.buttons.values(), key=lambda x: x.Z)
                if b.available]
        self.cmb_mapel.clear()
        self.cmb_mapel.addItems(syms)
        n = sum(len(b.available)
                for b in self.elements.table.buttons.values())
        self.say('Element table set for %.2f-%.2f keV: %d element-shell '
                 'combinations available.' % (o.e_low, o.e_high, n))

    def on_choose_elements(self):
        self.elements.show()
        self.elements.raise_()
        self.elements.activateWindow()

    def _on_elements_changed(self):
        self.lbl_els.setText('selected: ' + self.elements.table.summary())

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

    def on_layers(self):
        if not hasattr(self, 'layerview'):
            self.layerview = LayerStackDialog(self.session, self)
        self.layerview.show()
        self.layerview.raise_()
        self.layerview.activateWindow()

    def on_sample_model(self):
        self.sample.show()
        self.sample.raise_()
        self.sample.activateWindow()

    def _sample_summary(self):
        m = self.sample.matrix()
        t = self.sample.thickness()
        return ('matrix %s, %s, %.3g MeV, %.0f deg'
                % ('typed (%d elements)' % len(m) if m else 'bootstrapped',
                   'thick target' if t is None else '%.4g mg/cm2' % t,
                   self.sample.beam(), self.sample.theta()))

    # -- line identification ---------------------------------------------

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

    def on_eff_changed(self, idx):
        path = self.cmb_eff.itemData(idx)
        if not path:
            self.session.efficiency = None
            self.say('Efficiency cleared - areas only, no concentrations.')
            return
        try:
            e = self.session.load_efficiency(path)
        except Exception as ex:
            self.say('EFFICIENCY FAILED: %s' % ex)
            return
        self.say('Efficiency: %s  (C Ka %.4f, Si Ka %.3f, Fe Ka %.3f)'
                 % (e.name, e(0.277), e(1.740), e(6.404)))

    def on_load_eff(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Detector efficiency curve', '', 'Text (*.txt);;All (*)')
        if not path:
            return
        self.cmb_eff.addItem(os.path.basename(path), path)
        self.cmb_eff.setCurrentIndex(self.cmb_eff.count() - 1)

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

    def on_axes_reset(self):
        for w in (self.ed_x0, self.ed_x1, self.ed_y1):
            w.clear()
        self.refresh_spectrum()

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
        self.fill_table(s.fit)
        if s._conc:
            was_boot = self.sample.rb_boot.isChecked()
            self.sample.fill_from(s._conc, s.db)
            if was_boot:
                self.sample.rb_boot.setChecked(True)
        self.tabs.setCurrentIndex(2)

    # -- mask tools ------------------------------------------------------

    def on_mode_changed(self, mode):
        self.map._sel.set_active(mode == 'rectangle')
        extra = '  (click a pixel to seed)' if mode == 'flood fill' else ''
        self.say('Mask tool: %s%s' % (mode, extra))

    def on_map_click(self, event):
        if self.cmb_mode.currentText() != 'flood fill':
            return
        if event.inaxes is not self.map.ax or event.xdata is None:
            return
        data = self.map._data
        if data is None:
            return
        m = MSK.flood(data, (int(event.xdata), int(event.ydata)),
                      tolerance=self.spin_tol.value())
        if not m.any():
            self.say('Flood found nothing there - widen the tolerance.')
            return
        self._apply_mask(m, 'flood')

    def on_morph(self, direction):
        if self._last_mask is None:
            self.say('Make a mask first.')
            return
        m = (MSK.grow(self._last_mask, 1) if direction > 0
             else MSK.shrink(self._last_mask, 1))
        self._apply_mask(m, self.session.mask_name or 'mask')

    def _apply_mask(self, m, name):
        try:
            n = self.session.set_mask(m, name)
        except Exception as ex:
            self.say('MASK FAILED: %s' % ex)
            return
        self._last_mask = m
        self.say('%s mask: %s, %d events. Refit to analyse it.'
                 % (name, MSK.describe(m), n))
        self.refresh_spectrum()

    # -- batch -----------------------------------------------------------

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
        prog = QtWidgets.QProgressDialog('Processing...', 'Cancel', 0,
                                         len(paths), self)
        prog.setWindowModality(QtCore.Qt.WindowModal)

        def tick(i, n, path):
            prog.setValue(i)
            prog.setLabelText(os.path.basename(path))
            QtWidgets.QApplication.processEvents()

        try:
            res = run_batch(paths, els, out, options=self.session.options,
                            calibration=cal, efficiency=eff,
                            adc=self.spin_adc.value(),
                            quantify=bool(eff), progress=tick)
        except Exception as ex:
            self.say('BATCH FAILED: %s' % ex)
            return
        finally:
            prog.setValue(len(paths))
        self.say('Batch complete.' + chr(10) + summarise(res) + chr(10)
                 + 'Summary table: %s'
                 % os.path.join(out, 'batch_summary.csv'))
        self.tabs.setCurrentIndex(3)

    # -- interop and live -------------------------------------------------

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
        self.say('Wrote %s (format version -3) plus a .provenance.json '
                 'recording the MAC dataset, fluorescence yields and '
                 'efficiency curve, which version -3 cannot store.'
                 % os.path.basename(path))

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
        self.refresh_spectrum()

    def _live_tick(self):
        r = getattr(self, '_live', None)
        if r is None:
            return
        if r.poll():
            refresh_session(self.session, r, self.spin_adc.value())
            self.refresh_spectrum()
            self.status.showMessage('LIVE: ' + r.status())

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
    win = MainWindow()
    win.show()

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
                    win.say('detector: %s' % ns.detector)
                except Exception as ex:
                    win.say('bad --detector (%s)' % ex)
            if ns.range:
                try:
                    lo, hi = (float(x) for x in ns.range.split(','))
                    win.ed_elo.setText('%.6g' % lo)
                    win.ed_ehi.setText('%.6g' % hi)
                    win.on_range_changed() if hasattr(win, 'on_range_changed')                         else win.on_cal_changed()
                    win.session.options.e_low = lo
                    win.session.options.e_high = hi
                    win.session.invalidate()
                    win.refresh_spectrum()
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
                    win._on_elements_changed()
                    win.refresh_spectrum()
            if ns.fit:
                # after the event loop starts, so the window is painted and
                # visible before the fit begins rather than after it ends
                QtCore.QTimer.singleShot(400, win.on_fit)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
