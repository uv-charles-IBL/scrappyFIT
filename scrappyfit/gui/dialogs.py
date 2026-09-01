"""Dialogs: the element selector, and the sample model.

Both exist because the left panel is 360 px wide and these need more room.
A periodic table is 18 columns; squeezing it into a sidebar would defeat the
point of using one.
"""

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

    def __init__(self, db, parent=None):
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
        self.table = PeriodicTable(db)
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
        self.viewer = PeriodicTable(db, pick_only=True)
        self.lines = LineTable(db)
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
    """

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle('Sample model')
        self.setModal(False)
        self.resize(460, 520)

        v = QtWidgets.QVBoxLayout(self)

        self.rb_boot = QtWidgets.QRadioButton(
            'Bootstrap the matrix from the fitted areas')
        self.rb_boot.setChecked(True)
        self.rb_known = QtWidgets.QRadioButton('Use the composition below')
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
