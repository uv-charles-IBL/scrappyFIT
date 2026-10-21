"""Subshell-resolved reader for ElamDB12.txt.

Why this exists
---------------
GeoPIXE's xray_lines.txt collapses L emission into four flat groups per
element (La_, Lb_, Ll, Leta) with a single relative intensity each. That
throws away the structure that makes L lines tractable, and it is where the
100:10:1:1 placeholders for Ca/Sc/Ti come from.

ElamDB12.txt, shipped in the same directory, keeps the real structure:

    Edge  L3  <energy_eV>  <omega_L3>  <jump>
      Lines
        L3-M1  Ll   <energy_eV>  <branching ratio within L3>
        ...
      CK  L3 <f23>

Every line is filed under the subshell whose vacancy produces it, and the
branching ratios within one subshell sum to 1. Those intra-subshell ratios are
atomic constants (Scofield / Campbell-Wang class data) - they do NOT depend on
how the vacancy was made. The subshell POPULATIONS do, via the ionisation
cross sections and Coster-Kronig redistribution.

That split is what allows overlapping L lines to be deconvoluted: lock every
intra-subshell ratio, and let only the three subshell populations float. For a
heavy element with a dozen L lines in a crowded region, that takes the free
parameters from twelve to three.

Reference for the compilation:
    W.T. Elam, B.D. Ravel, J.R. Sieber (2002), "A new atomic database for
    X-ray spectroscopic calculations", Radiat. Phys. Chem. 63, 121-128.
"""

import os
import re

SHELLS = ('K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5')


class ElamDB:
    def __init__(self, dat):
        self.el = {}                      # sym -> dict
        self._parse(os.path.join(dat, 'ElamDB12.txt'))

    def _parse(self, path):
        cur = None
        edge = None
        in_lines = False
        for raw in open(path, errors='ignore'):
            line = raw.rstrip('\n')
            if line.startswith('//'):
                continue
            t = line.split()
            if not t:
                continue
            if t[0] == 'Element':
                cur = dict(sym=t[1], Z=int(t[2]), A=float(t[3]),
                           density=float(t[4]), edges={}, lines={}, ck={})
                self.el[t[1]] = cur
                edge, in_lines = None, False
            elif cur is None:
                continue
            elif t[0] == 'Edge':
                edge = t[1]
                cur['edges'][edge] = dict(energy=float(t[2]) / 1000.0,   # keV
                                          omega=float(t[3]),
                                          jump=float(t[4]))
                cur['lines'][edge] = []
                in_lines = False
            elif t[0] == 'Lines':
                in_lines = True
            elif t[0] == 'CK':
                # CK  L2 0.300  L3 0.570   -> from THIS edge into those
                d = {}
                for i in range(1, len(t) - 1, 2):
                    d[t[i]] = float(t[i + 1])
                cur['ck'][edge] = d
                in_lines = False
            elif t[0] == 'CKtotal':
                in_lines = False
            elif t[0] == 'Photo' or t[0] == 'Scatter':
                in_lines = False
                edge = None
            elif in_lines and edge is not None and len(t) >= 4:
                # e.g.  L3-M5  La1  7.04800e+02  8.11679e-01
                try:
                    cur['lines'][edge].append(
                        dict(trans=t[0], name=t[1],
                             energy=float(t[2]) / 1000.0, ratio=float(t[3])))
                except ValueError:
                    pass

    def lines(self, sym, shell):
        """[(name, energy_keV, branching_ratio_within_shell), ...]"""
        e = self.el.get(sym)
        if not e:
            return []
        return [(d['name'], d['energy'], d['ratio'])
                for d in e['lines'].get(shell, [])]

    def omega(self, sym, shell):
        e = self.el.get(sym)
        return e['edges'].get(shell, {}).get('omega', 0.0) if e else 0.0

    def edge(self, sym, shell):
        e = self.el.get(sym)
        return e['edges'].get(shell, {}).get('energy', 0.0) if e else 0.0

    def ck(self, sym, frm, to):
        e = self.el.get(sym)
        return e['ck'].get(frm, {}).get(to, 0.0) if e else 0.0

    def l_populations(self, sym, sigma_L1, sigma_L2, sigma_L3):
        """Vacancy populations after Coster-Kronig redistribution.

        f12, f13 move L1 vacancies down; f23 moves L2 vacancies (including
        those that arrived from L1) into L3. Order matters.
        """
        f12 = self.ck(sym, 'L1', 'L2')
        f13 = self.ck(sym, 'L1', 'L3')
        f23 = self.ck(sym, 'L2', 'L3')
        n1 = sigma_L1
        n2 = sigma_L2 + f12 * n1
        n3 = sigma_L3 + f13 * n1 + f23 * n2
        return n1, n2, n3
