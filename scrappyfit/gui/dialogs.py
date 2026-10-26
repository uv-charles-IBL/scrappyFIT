"""Dialogs: the element selector, and the sample model.

Both exist because the left panel is 360 px wide and these need more room.
A periodic table is 18 columns; squeezing it into a sidebar would defeat the
point of using one.
"""

import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .periodic import LineTable, PeriodicTable, state_label


class ElementDialog(QtWidgets.QDialog):
    """Two tabs over the same periodic table layout.

    **Fit selection** cycles each element through none / K / L / M / KL / LM /
    KLM, so one repeated click walks the possibilities without leaving the
    cell.

    **Line viewer** is read-only: click an element and read its lines. Kept
    separate deliberately - browsing what an element emits and deciding to fit
    it are different acts, and a viewer that silently changes the fit is a
    trap.

    Non-modal, so it can stay open beside the spectrum while you work.
    """

    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, db, parent=None, lines_fn=None, groups_fn=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle('Elements')
        self.setModal(False)
        self.resize(820, 560)

        v = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()

        # -- fit selection
        w1 = QtWidgets.QWidget()
        v1 = QtWidgets.QVBoxLayout(w1)
        self.table = PeriodicTable(db, lines_fn=lines_fn)
        self.table.selectionChanged.connect(self._on_change)
        v1.addWidget(self.table)
        self.lbl = QtWidgets.QLabel('(nothing selected)')
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet('font-size:11px; color:#0F766E;')
        v1.addWidget(self.lbl)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (('Clear all', self.table.clear),
                         ('Light preset', lambda: self._preset('light')),
                         ('Silicate preset', lambda: self._preset('silicate'))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        v1.addLayout(row)
        self.tabs.addTab(w1, 'Fit selection')

        # -- line viewer
        w2 = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(w2)
        self.viewer = PeriodicTable(db, pick_only=True, lines_fn=lines_fn)
        self.lines = LineTable(db, groups_fn=groups_fn)
        self.viewer.elementPicked.connect(self.lines.show_element)
        v2.addWidget(self.viewer)
        v2.addWidget(self.lines, 1)
        self.tabs.addTab(w2, 'Line viewer')

        v.addWidget(self.tabs)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.hide)
        v.addWidget(bb)

    def _on_change(self):
        self.lbl.setText('selected: ' + self.table.summary())
        self.selectionChanged.emit()

    def _preset(self, which):
        light = ['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                 'K', 'Ca']
        sil = light + ['Ti', 'Cr', 'Mn', 'Fe']
        names = light if which == 'light' else sil
        pairs = [(self.db.z[n.lower()], 1) for n in names
                 if n.lower() in self.db.z]
        if which == 'silicate':
            pairs += [(self.db.z[n.lower()], 2) for n in ('Fe', 'Ni')]
        self.table.set_selection(pairs)

    def set_range(self, lo, hi):
        self.table.set_range(lo, hi)
        self.viewer.set_range(lo, hi)
        self.lines.range = (lo, hi)

    def selection(self):
        return self.table.selection()

    def set_selection(self, pairs):
        self.table.set_selection(pairs)

    def set_suggested(self, sug):
        self.table.set_suggested(sug)
        self.viewer.set_suggested(sug)


class SampleModelDialog(QtWidgets.QDialog):
    """The sample the yield calculation assumes.

    Every concentration depends on this and it is the least visible part of
    the chain, so it gets its own window rather than being buried in options.

    The matrix matters because self-absorption is computed through it, which
    makes quantification mildly circular: you need the composition to get the
    composition. Two ways out, and the dialog offers both. Bootstrapping from
    the fitted areas is right for an unknown; typing a known matrix is right
    for a standard, and is what makes a standard useful.

    Thickness matters differently. A thick target - thicker than the proton
    range - is the simple case and is the default. A thin film is not, because
    then the yield scales with thickness and the answer is an areal density
    rather than a concentration.

    The sample's GeoPIXE yield files (session.yield_set) are shown at the
    top: one file per UNKNOWN layer, per region for a sample whose stack
    changes across the map. "Quantify with these .yield files" makes
    QUANTIFY read them as GeoPIXE does, file by file, instead of computing
    a one-layer model; the other two choices are the one-layer model,
    bootstrapped or typed. "Copy its composition" fills the model from the
    file shown.
    """

    yieldPicked = QtCore.pyqtSignal(str)
    yieldSetPicked = QtCore.pyqtSignal(list)    # .yield paths
    regionPicked = QtCore.pyqtSignal(str)
    searchPathChanged = QtCore.pyqtSignal()

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle('Sample model')
        self.setModal(False)
        self.resize(460, 520)

        v = QtWidgets.QVBoxLayout(self)

        gy = QtWidgets.QGroupBox('Yield files (GeoPIXE .yield)')
        gyl = QtWidgets.QVBoxLayout(gy)
        self.lbl_yset = QtWidgets.QLabel('none for this sample')
        self.lbl_yset.setWordWrap(True)
        self.lbl_yset.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        gyl.addWidget(self.lbl_yset)
        rrow = QtWidgets.QHBoxLayout()
        rrow.addWidget(QtWidgets.QLabel('region'))
        self.cmb_region = QtWidgets.QComboBox()
        self.cmb_region.setToolTip(
            'Files named <stem>_<region>_UNKNOWN-<layer>.yield form one set '
            "per region; QUANTIFY reads the chosen region's files")
        self.cmb_region.activated.connect(self._on_region)
        rrow.addWidget(self.cmb_region, 1)
        gyl.addLayout(rrow)
        brow = QtWidgets.QHBoxLayout()
        b_dir = QtWidgets.QPushButton('Folder...')
        b_dir.setToolTip('Use every .yield in a folder (remembered for this '
                         'data file)')
        b_dir.clicked.connect(self._pick_folder)
        b_files = QtWidgets.QPushButton('Files...')
        b_files.setToolTip('Use the .yield files picked (remembered for this '
                           'data file)')
        b_files.clicked.connect(self._pick_files)
        b_sp = QtWidgets.QPushButton('Search folders...')
        b_sp.setToolTip("Folders searched, after the data's own, for .yield "
                        "files whose name contains the data file's name")
        b_sp.clicked.connect(self._edit_search_path)
        for b in (b_dir, b_files, b_sp):
            brow.addWidget(b)
        brow.addStretch(1)
        gyl.addLayout(brow)
        v.addWidget(gy)

        self.yield_path = None
        self.lbl_yield = QtWidgets.QLabel('yield file: none for this sample')
        self.lbl_yield.setWordWrap(True)
        v.addWidget(self.lbl_yield)
        yrow = QtWidgets.QHBoxLayout()
        b_ych = QtWidgets.QPushButton('Change...')
        b_ych.setToolTip('Pick the .yield that describes this sample; every '
                         'window that uses one switches to it')
        b_ych.clicked.connect(self._pick_yield)
        self.btn_ycopy = QtWidgets.QPushButton('Copy its composition')
        self.btn_ycopy.setToolTip('Fill the composition, thickness, beam and '
                                  'angle below from the yield file')
        self.btn_ycopy.clicked.connect(self._on_copy)
        self.btn_ycopy.setEnabled(False)
        yrow.addWidget(b_ych)
        yrow.addWidget(self.btn_ycopy)
        yrow.addStretch(1)
        v.addLayout(yrow)

        self.rb_yields = QtWidgets.QRadioButton(
            'Quantify with these .yield files')
        self.rb_yields.setToolTip(
            "GeoPIXE's way: each file gives the elements of its UNKNOWN "
            'layer, in ppm of that layer')
        self.rb_yields.setEnabled(False)
        self.rb_boot = QtWidgets.QRadioButton(
            'Bootstrap the matrix from the fitted areas')
        self.rb_boot.setChecked(True)
        self.rb_known = QtWidgets.QRadioButton('Use the composition below')
        v.addWidget(self.rb_yields)
        v.addWidget(self.rb_boot)
        v.addWidget(self.rb_known)
        note = QtWidgets.QLabel(
            'Bootstrapping is right for an unknown. For a standard, type the '
            'known composition - that is what makes a standard useful, and it '
            'removes the circularity in the absorption correction.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666; font-size:10px;')
        v.addWidget(note)

        self.tbl = QtWidgets.QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(['element', 'weight %'])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl, 1)

        row = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton('Add row')
        b_add.clicked.connect(lambda: self._add('', ''))
        b_del = QtWidgets.QPushButton('Remove row')
        b_del.clicked.connect(self._remove)
        b_norm = QtWidgets.QPushButton('Normalise to 100')
        b_norm.clicked.connect(self._normalise)
        for b in (b_add, b_del, b_norm):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        f = QtWidgets.QFormLayout()
        self.ed_thick = QtWidgets.QLineEdit('')
        self.ed_thick.setPlaceholderText('blank = thick target')
        self.ed_beam = QtWidgets.QLineEdit('1.0')
        self.ed_theta = QtWidgets.QLineEdit('135.0')
        f.addRow('thickness mg/cm2', self.ed_thick)
        f.addRow('beam energy MeV', self.ed_beam)
        f.addRow('take-off angle deg', self.ed_theta)
        v.addLayout(f)

        note2 = QtWidgets.QLabel(
            'Leave thickness blank unless the sample is genuinely thinner '
            'than the proton range. For a thin film the result is an areal '
            'density, not a concentration.')
        note2.setWordWrap(True)
        note2.setStyleSheet('color:#666; font-size:10px;')
        v.addWidget(note2)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.hide)
        v.addWidget(bb)

    # -- the sample's yield files ---------------------------------------

    def show_yield_set(self, ys, region, source='', why=''):
        """Show the session's yield set; why says why there is none."""
        had = self.rb_yields.isEnabled()
        self.cmb_region.clear()
        if ys is None or not len(ys):
            self.lbl_yset.setText(
                'none for this sample - QUANTIFY bootstraps a one-layer '
                'matrix.%s' % (('\n' + why) if why else ''))
            self.lbl_yset.setToolTip('')
            if self.rb_yields.isChecked():
                self.rb_boot.setChecked(True)
            self.rb_yields.setEnabled(False)
            self.cmb_region.setEnabled(False)
            return
        regs = [r for r in ys.regions if r]
        for r in regs:
            self.cmb_region.addItem(r)
        if not regs:
            self.cmb_region.addItem('(one set)')
        self.cmb_region.setEnabled(len(regs) > 1)
        if region in regs:
            self.cmb_region.setCurrentIndex(regs.index(region))
        elif len(regs) > 1:
            self.cmb_region.setCurrentIndex(-1)
        lines = ['%s (%s)' % (ys.describe(region), source or 'chosen')]
        if region is None and len(regs) > 1:
            lines.append('choose a region: QUANTIFY needs one')
        for e in ys.entries(region if regs else None):
            lines.append('  %-11s %s' % (e.layer, e.name))
        self.lbl_yset.setText('\n'.join(lines))
        self.lbl_yset.setToolTip('\n'.join(ys.paths))
        self.rb_yields.setEnabled(True)
        # A set found or chosen for the sample is what QUANTIFY should use.
        # A file just exported from this model is the model itself, so the
        # model's own (normalised) QUANTIFY stays the default for it.
        if not had and source != 'exported':
            self.rb_yields.setChecked(True)

    def use_yields(self):
        return self.rb_yields.isEnabled() and self.rb_yields.isChecked()

    def _on_region(self, i):
        r = self.cmb_region.itemText(i)
        if r and r != '(one set)':
            self.regionPicked.emit(r)

    def _pick_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Folder of .yield files', os.path.dirname(
                self.yield_path or ''))
        if not d:
            return
        files = sorted(os.path.join(d, n) for n in os.listdir(d)
                       if n.lower().endswith('.yield'))
        if files:
            self.yieldSetPicked.emit(files)
        else:
            self.lbl_yset.setText('no .yield files in %s' % d)

    def _pick_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'GeoPIXE yield files', os.path.dirname(
                self.yield_path or ''), 'Yield (*.yield);;All (*)')
        if files:
            self.yieldSetPicked.emit(list(files))

    def _edit_search_path(self):
        from .. import yieldset
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, 'Yield search folders',
            "Folders searched for a data file's .yield files, one per line "
            '(a file belongs to the data when its name contains the data '
            "file's name):", '\n'.join(yieldset.search_path()))
        if ok:
            yieldset.set_search_path(
                [t.strip() for t in text.splitlines() if t.strip()])
            self.searchPathChanged.emit()

    def show_yield(self, path, source=''):
        self.yield_path = path or None
        if path:
            self.lbl_yield.setText('yield file: %s (%s)'
                                   % (os.path.basename(path), source))
            self.lbl_yield.setToolTip(str(path))
        else:
            self.lbl_yield.setText('yield file: none for this sample')
            self.lbl_yield.setToolTip('')
        self.btn_ycopy.setEnabled(bool(path))

    def _pick_yield(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open a GeoPIXE yield file', '', 'Yield (*.yield);;All (*)')
        if path:
            self.yieldPicked.emit(path)

    def _on_copy(self):
        try:
            what = self.copy_from_yield()
        except Exception as ex:
            self.lbl_yield.setText('could not read %s: %s' % (
                os.path.basename(self.yield_path or ''), ex))
            return
        self.lbl_yield.setText('%s - copied %s' % (self.lbl_yield.text(), what))

    def copy_from_yield(self):
        """Fill the model from the yield file's UNKNOWN layer.

        Weight fractions, beam and take-off angle always. Thickness when the
        layer is thinner than the 1e4 mg/cm2 that File > Export yield writes
        for a thick target; otherwise it is left blank (thick).
        Returns a one-line description, or raises on an unreadable file.
        """
        from ..io.gpyield import read_yield
        Y = read_yield(self.yield_path)[0]
        layers = Y['layers']
        k = min(max(int(Y.get('unknown', 1) or 1), 1), len(layers)) - 1
        L = layers[k]
        self.tbl.setRowCount(0)
        for Z, w in sorted(zip(L['Z'], L['W']), key=lambda t: -t[1]):
            self._add(self.db.sym[Z], '%.4f' % (100 * w))
        self.rb_known.setChecked(True)
        t = float(L['thick'])
        thin = 0 < t < 1e4
        self.ed_thick.setText('%.6g' % t if thin else '')
        self.ed_beam.setText('%.6g' % Y.get('e_beam', 1.0))
        self.ed_theta.setText('%.6g' % Y.get('theta', 135.0))
        # theta is GeoPIXE's detector angle to the beam (layer_setup.pro:1740),
        # the convention this model and from_session use; the layer record's
        # thick is always mg/cm2 (make_layer.pro:26). The model has no target
        # tilt and one layer, so say what was dropped.
        what = 'layer %d of %d (%s), %s' % (
            k + 1, len(layers), L.get('name') or 'unnamed',
            '%.4g mg/cm2' % t if thin else 'thick target')
        if len(layers) > 1:
            what += ('; the other layers were NOT copied (one-layer model, '
                     'so their absorption is not in QUANTIFY)')
        if Y.get('alpha') or Y.get('beta'):
            what += ('; target tilt alpha %.3g beta %.3g deg NOT copied'
                     % (Y.get('alpha', 0.0), Y.get('beta', 0.0)))
        return what

    def _add(self, sym, pct):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(str(sym)))
        self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(str(pct)))

    def _remove(self):
        r = self.tbl.currentRow()
        if r >= 0:
            self.tbl.removeRow(r)

    def _normalise(self):
        vals = []
        for r in range(self.tbl.rowCount()):
            try:
                vals.append(float(self.tbl.item(r, 1).text()))
            except Exception:
                vals.append(0.0)
        t = sum(vals)
        if t <= 0:
            return
        for r, v in enumerate(vals):
            self.tbl.setItem(r, 1,
                             QtWidgets.QTableWidgetItem('%.4f' % (100 * v / t)))

    def fill_from(self, conc, db):
        """Populate from a previous result, so refining is one click."""
        self.tbl.setRowCount(0)
        for Z, v in sorted(conc.items(), key=lambda t: -t[1]):
            self._add(db.sym[Z], '%.4f' % v)
        self.rb_known.setChecked(True)

    def matrix(self):
        """{Z: weight fraction}, or None to bootstrap."""
        if self.rb_boot.isChecked():
            return None
        out = {}
        for r in range(self.tbl.rowCount()):
            it0, it1 = self.tbl.item(r, 0), self.tbl.item(r, 1)
            if not it0 or not it1:
                continue
            Z = self.db.z.get(it0.text().strip().lower())
            try:
                v = float(it1.text())
            except Exception:
                continue
            if Z and v > 0:
                out[Z] = out.get(Z, 0.0) + v
        if not out:
            return None
        t = sum(out.values())
        return {Z: v / t for Z, v in out.items()}

    def thickness(self):
        t = self.ed_thick.text().strip()
        try:
            return float(t) if t else None
        except ValueError:
            return None

    def beam(self):
        try:
            return float(self.ed_beam.text())
        except ValueError:
            return 1.0

    def theta(self):
        try:
            return float(self.ed_theta.text())
        except ValueError:
            return 135.0
