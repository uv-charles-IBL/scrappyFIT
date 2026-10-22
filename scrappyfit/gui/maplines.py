"""Pick a map line from a periodic table.

The Maps tab's line list is long (over a hundred element-shell lines in a
0.2-6.5 keV window) and alphabetical by Z, so finding 'Pb M' means scrolling.
This lays the same lines out as the element dialog's periodic table
(periodic.LAYOUT): each cell shows the shells it can map, one click maps the
only one, and a cell with several asks which ('Ag K' is off a 6.5 keV
spectrum; 'Ag L' is not). Anything in the list that is not an element line
on the table (a DA matrix row named its own way) gets a plain button below
it, so the table offers exactly what the list does. GeoPIXE has no table
here: its image window picks from a droplist (gimage.pro:1636-1647;
geopixe-ref 2026-10-21).
"""

from PyQt5 import QtCore, QtWidgets

from .periodic import LAYOUT, SHELL_CH, ElementButton

_SHELL = {'K': 1, 'L': 2, 'M': 3}
_ON_TABLE = {sym for sym, _, _ in LAYOUT}


def split_line(name):
    """'Ag L' -> ('Ag', 2); 'Si' -> ('Si', 1); 'ag l' -> ('Ag', 2).
    None for anything that is not a symbol on the table with an optional
    K, L or M ('1.74', 'Ag La', 'Back1')."""
    parts = str(name).split()
    if not parts or len(parts) > 2:
        return None
    sym = parts[0][:1].upper() + parts[0][1:].lower()
    if sym not in _ON_TABLE:
        return None
    if len(parts) == 1:
        return sym, 1
    sh = _SHELL.get(parts[1].upper())
    return (sym, sh) if sh else None


class _LineButton(ElementButton):
    """A table cell labelled with the shells it can map, not a fit state;
    shells marks the line now shown."""

    def refresh(self):
        super().refresh()
        if self.available:
            self.setText('%s\n%s' % (self.symbol, ''.join(
                SHELL_CH[s] for s in self.available)))


class MapLinePicker(QtWidgets.QDialog):
    """A periodic table of the lines in `lines` (Maps tab names: 'Si',
    'Ag L', 'Pb M'). linePicked carries the chosen name, exactly as it
    stands in `lines`."""

    linePicked = QtCore.pyqtSignal(str)

    def __init__(self, db, lines, current='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pick a map line')
        self.db = db
        self.lines = list(lines)
        self._names = {}                    # (sym, shell) -> name in lines
        others = []
        for nm in self.lines:
            key = split_line(nm)
            if key is None:
                others.append(nm)
            else:
                self._names.setdefault(key, nm)
        cur = split_line(current) if str(current).strip() else None
        g = QtWidgets.QGridLayout()
        g.setSpacing(2)
        self.buttons = {}
        for sym, col, row in LAYOUT:
            Z = db.z.get(sym.lower())
            if Z is None:
                continue
            b = _LineButton(sym, Z, self)
            b.available = tuple(sorted(sh for s, sh in self._names
                                       if s == sym))
            b.shells = (cur[1],) if cur and cur[0] == sym else ()
            b.refresh()
            b.setToolTip(', '.join(self._names[(sym, s)] for s in b.available)
                         or 'no line on the spectrum')
            b.clicked.connect(lambda _=False, bb=b: self._clicked(bb))
            g.addWidget(b, row, col)
            self.buttons[sym] = b
        g.setRowMinimumHeight(8, 8)
        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(g)
        self.other_buttons = {}
        if others:
            h = QtWidgets.QHBoxLayout()
            h.addWidget(QtWidgets.QLabel('also in the list:'))
            for nm in others:
                ob = QtWidgets.QPushButton(nm)
                ob.clicked.connect(lambda _=False, n=nm: self.pick(n))
                h.addWidget(ob)
                self.other_buttons[nm] = ob
            h.addStretch(1)
            v.addLayout(h)
        note = QtWidgets.QLabel(
            'Click an element to map it; one with several shells (K, L, M) '
            'asks which. Greyed: no line on the spectrum.')
        note.setStyleSheet('color:#666; font-size:10px;')
        v.addWidget(note)

    def _clicked(self, b):
        av = b.available
        if not av:
            return
        if len(av) == 1:
            self.pick(self._names[(b.symbol, av[0])])
            return
        menu = QtWidgets.QMenu(self)
        for sh in av:
            nm = self._names[(b.symbol, sh)]
            menu.addAction('%s  (%s)' % (nm, SHELL_CH[sh]),
                           lambda n=nm: self.pick(n))
        menu.popup(b.mapToGlobal(QtCore.QPoint(0, b.height())))
        self._menu = menu            # kept for tests and to stay alive

    def pick(self, name):
        """Choose a line by its Maps tab name and close."""
        if name not in self.lines:
            raise ValueError('%s is not a map line here' % name)
        self.linePicked.emit(name)
        self.accept()
