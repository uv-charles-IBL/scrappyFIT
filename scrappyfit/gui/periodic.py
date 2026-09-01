"""A periodic table for choosing elements and inspecting their lines.

Why a table rather than a list
------------------------------
A list of 51 symbols tells you nothing about which of them are plausible
together. A periodic table does: neighbours in a group behave alike, the
transition metals sit in a block, the lanthanides are a row you can dismiss at
a glance. On an unknown sample that spatial memory is most of how you decide
what to try.

It also solves the shell problem. An element can contribute K, L or M lines,
and which of them you want depends on the energy window - iron in a 6.6 keV
window shows both K and L, and they behave completely differently. Rather than
three separate lists, each cell CYCLES through the combinations:

    none  ->  K  ->  L  ->  M  ->  K+L  ->  L+M  ->  K+L+M  ->  none

so one repeated click walks the possibilities without leaving the cell. Only
shells that actually put a line in the current window are offered - clicking
past an impossible combination would be noise.

Two tabs, because looking and choosing are different activities:

    Lines   click an element, read its line table. Nothing is selected.
    Fit     click an element, cycle what gets fitted.
"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# (symbol, column, row). Lanthanides and actinides on rows 9 and 10, offset,
# as they are drawn conventionally.
LAYOUT = [
    ('H', 1, 1), ('He', 18, 1),
    ('Li', 1, 2), ('Be', 2, 2), ('B', 13, 2), ('C', 14, 2), ('N', 15, 2),
    ('O', 16, 2), ('F', 17, 2), ('Ne', 18, 2),
    ('Na', 1, 3), ('Mg', 2, 3), ('Al', 13, 3), ('Si', 14, 3), ('P', 15, 3),
    ('S', 16, 3), ('Cl', 17, 3), ('Ar', 18, 3),
    ('K', 1, 4), ('Ca', 2, 4), ('Sc', 3, 4), ('Ti', 4, 4), ('V', 5, 4),
    ('Cr', 6, 4), ('Mn', 7, 4), ('Fe', 8, 4), ('Co', 9, 4), ('Ni', 10, 4),
    ('Cu', 11, 4), ('Zn', 12, 4), ('Ga', 13, 4), ('Ge', 14, 4),
    ('As', 15, 4), ('Se', 16, 4), ('Br', 17, 4), ('Kr', 18, 4),
    ('Rb', 1, 5), ('Sr', 2, 5), ('Y', 3, 5), ('Zr', 4, 5), ('Nb', 5, 5),
    ('Mo', 6, 5), ('Tc', 7, 5), ('Ru', 8, 5), ('Rh', 9, 5), ('Pd', 10, 5),
    ('Ag', 11, 5), ('Cd', 12, 5), ('In', 13, 5), ('Sn', 14, 5),
    ('Sb', 15, 5), ('Te', 16, 5), ('I', 17, 5), ('Xe', 18, 5),
    ('Cs', 1, 6), ('Ba', 2, 6), ('La', 3, 6), ('Hf', 4, 6), ('Ta', 5, 6),
    ('W', 6, 6), ('Re', 7, 6), ('Os', 8, 6), ('Ir', 9, 6), ('Pt', 10, 6),
    ('Au', 11, 6), ('Hg', 12, 6), ('Tl', 13, 6), ('Pb', 14, 6),
    ('Bi', 15, 6), ('Po', 16, 6), ('At', 17, 6), ('Rn', 18, 6),
    ('Fr', 1, 7), ('Ra', 2, 7), ('Ac', 3, 7),
    ('Ce', 4, 9), ('Pr', 5, 9), ('Nd', 6, 9), ('Pm', 7, 9), ('Sm', 8, 9),
    ('Eu', 9, 9), ('Gd', 10, 9), ('Tb', 11, 9), ('Dy', 12, 9),
    ('Ho', 13, 9), ('Er', 14, 9), ('Tm', 15, 9), ('Yb', 16, 9),
    ('Lu', 17, 9),
    ('Th', 4, 10), ('Pa', 5, 10), ('U', 6, 10),
]

# The cycle. Each state is the set of shells fitted for that element.
CYCLE = [(), (1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3)]
SHELL_CH = {1: 'K', 2: 'L', 3: 'M'}

AVAILABLE = '#FFFFFF'
UNAVAILABLE = '#EFEFEF'
SELECTED = '#0F766E'
SUGGESTED = '#CFF3EA'


def state_label(shells):
    return ''.join(SHELL_CH[s] for s in sorted(shells))


class ElementButton(QtWidgets.QToolButton):
    def __init__(self, symbol, Z, parent=None):
        super().__init__(parent)
        self.symbol, self.Z = symbol, Z
        self.shells = ()
        self.available = ()
        self.suggested = False
        self.setFixedSize(38, 34)
        self.setAutoRaise(False)
        self.refresh()

    def refresh(self):
        lab = state_label(self.shells)
        self.setText('%s\n%s' % (self.symbol, lab) if lab else self.symbol)
        if not self.available:
            bg, fg, bold = UNAVAILABLE, '#B0B0B0', False
        elif self.shells:
            bg, fg, bold = SELECTED, 'white', True
        elif self.suggested:
            bg, fg, bold = SUGGESTED, '#0F766E', True
        else:
            bg, fg, bold = AVAILABLE, '#151A22', False
        self.setStyleSheet(
            'QToolButton { background:%s; color:%s; border:1px solid #CCC; '
            'border-radius:3px; font-size:10px; font-weight:%s; }'
            'QToolButton:hover { border:1px solid #0F766E; }'
            % (bg, fg, 'bold' if bold else 'normal'))
        self.setEnabled(bool(self.available))

    def advance(self):
        """Next state in the cycle that uses only available shells."""
        if not self.available:
            return
        try:
            i = CYCLE.index(tuple(sorted(self.shells)))
        except ValueError:
            i = 0
        av = set(self.available)
        for step in range(1, len(CYCLE) + 1):
            cand = CYCLE[(i + step) % len(CYCLE)]
            if set(cand) <= av:
                self.shells = cand
                break
        self.refresh()


class PeriodicTable(QtWidgets.QWidget):
    """Grid of element buttons.

    selectionChanged fires whenever the fitted set changes.
    elementPicked fires on any click, for the Lines tab.
    """

    selectionChanged = QtCore.pyqtSignal()
    elementPicked = QtCore.pyqtSignal(int)

    def __init__(self, db, parent=None, pick_only=False):
        super().__init__(parent)
        self.db = db
        self.pick_only = pick_only
        self.buttons = {}
        g = QtWidgets.QGridLayout(self)
        g.setSpacing(2)
        g.setContentsMargins(4, 4, 4, 4)
        for sym, col, row in LAYOUT:
            Z = db.z.get(sym.lower())
            if Z is None:
                continue
            b = ElementButton(sym, Z, self)
            b.clicked.connect(lambda _=False, bb=b: self._clicked(bb))
            g.addWidget(b, row, col)
            self.buttons[sym] = b
        g.setRowMinimumHeight(8, 8)
        note = QtWidgets.QLabel(
            'click to cycle  none - K - L - M - KL - LM - KLM'
            if not pick_only else 'click an element to see its lines')
        note.setStyleSheet('color:#666; font-size:10px;')
        g.addWidget(note, 11, 1, 1, 18)

    def _clicked(self, b):
        self.elementPicked.emit(b.Z)
        if not self.pick_only:
            b.advance()
            self.selectionChanged.emit()

    def set_range(self, e_low, e_high, min_intensity=0.05):
        """Grey out anything with no usable line in the window, and drop any
        selection that is no longer possible."""
        changed = False
        for sym, b in self.buttons.items():
            av = []
            for sh in (1, 2, 3):
                try:
                    lines = self.db.line_list(b.Z, sh)
                except Exception:
                    lines = []
                if any(e_low <= e <= e_high and i >= min_intensity
                       for e, i in lines):
                    av.append(sh)
            b.available = tuple(av)
            keep = tuple(s for s in b.shells if s in av)
            if keep != b.shells:
                b.shells = keep
                changed = True
            b.refresh()
        if changed:
            self.selectionChanged.emit()

    def selection(self):
        """[(Z, shell)] currently chosen."""
        out = []
        for b in self.buttons.values():
            for sh in b.shells:
                out.append((b.Z, sh))
        return sorted(out)

    def set_selection(self, pairs):
        want = {}
        for Z, sh in pairs:
            want.setdefault(Z, set()).add(sh)
        for b in self.buttons.values():
            b.shells = tuple(sorted(want.get(b.Z, ())))
            b.refresh()
        self.selectionChanged.emit()

    def clear(self):
        for b in self.buttons.values():
            b.shells = ()
            b.refresh()
        self.selectionChanged.emit()

    def set_suggested(self, pairs):
        want = {sym for sym, sh, sc, n in pairs} if pairs and len(
            pairs[0]) == 4 else set()
        for sym, b in self.buttons.items():
            b.suggested = sym in want
            b.refresh()

    def summary(self):
        bits = []
        for b in sorted(self.buttons.values(), key=lambda x: x.Z):
            if b.shells:
                bits.append('%s%s' % (b.symbol, state_label(b.shells)))
        return ', '.join(bits) if bits else '(nothing selected)'


class LineTable(QtWidgets.QWidget):
    """Every tabulated line of one element, per shell. The reference you want
    open while deciding whether a feature is really that element."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        v = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel('click an element above')
        self.title.setStyleSheet('font-weight:bold; font-size:13px;')
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['shell', 'line', 'energy keV', 'rel. intensity', 'in window'])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.title)
        v.addWidget(self.table)
        self.range = (0.2, 6.5)

    def show_element(self, Z):
        sym = self.db.sym[Z]
        lo, hi = self.range
        rows = []
        for sh in (1, 2, 3):
            try:
                lines = self.db.line_list(Z, sh)
            except Exception:
                lines = []
            for e, i in sorted(lines, key=lambda t: -t[1]):
                rows.append((SHELL_CH[sh], e, i, lo <= e <= hi))
        edge = self.db.edge.get((Z, 'K'), 0.0)
        self.title.setText('%s  (Z = %d)%s' % (sym, Z,
                           '     K edge %.4f keV' % edge if edge else ''))
        self.table.setRowCount(len(rows))
        for r, (shn, e, i, inw) in enumerate(rows):
            vals = (shn, '%s line' % shn, '%.4f' % e, '%.4f' % i,
                    'yes' if inw else '-')
            for c, t in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(t)
                if not inw:
                    it.setForeground(QtGui.QBrush(QtGui.QColor('#AAA')))
                self.table.setItem(r, c, it)
        if not rows:
            self.title.setText('%s (Z = %d) - no tabulated lines' % (sym, Z))
