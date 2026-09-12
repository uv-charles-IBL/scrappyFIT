"""
Readers for the GeoPIXE fundamental-parameter database.

Every file read here is one GeoPIXE already ships, so the physics stays
identical - this is a second implementation of the same model against the same
data, not a different model. That is the point: it lets the numbers be checked
without an IDL licence.

Paths default to the database directory of a GeoPIXE checkout.
"""

import math
import os
import struct

# ---------------------------------------------------------------------------

N_A = 0.60221408          # Avogadro / 1e24, the form used throughout
DEDX_CONV = 0.60225       # (SE+SN)*0.60225/A2  ->  MeV/(mg/cm^2), per dedx.pro


def _f(s):
    return float(s.replace('D', 'E').replace('d', 'e'))


class Database:
    """Loads the GeoPIXE database files once and answers FP queries."""

    def __init__(self, root=None, lines_file='xray_lines.txt'):
        """root: .../Workspace/main/database

        lines_file  which line table to read. 'xray_lines_elam.txt' is the
                    rebuild whose L and M sections come from ElamDB12 instead
                    of the placeholder-contaminated original.
        """
        if root is None:
            from .. import config
            root = str(config.database_path())
        self.root = root
        self.lines_file = lines_file
        self.dat = os.path.join(root, 'dat')
        self._elam()
        self._edges()
        self._lines()
        self._xsect_k()
        self._proton()
        self._mac = {}

    # -- Elam: symbols, Z, A, fluorescence yields, edges, jump ratios --------

    def _elam(self):
        self.sym = {}          # Z -> symbol
        self.z = {}            # symbol(lower) -> Z
        self.A = {}            # Z -> atomic weight
        self.rho = {}          # Z -> density
        self.omega = {}        # (Z, shell) -> fluorescence yield
        self.jump = {}         # (Z, shell) -> edge jump
        self.eedge = {}        # (Z, shell) -> edge energy, keV
        cur = None
        with open(os.path.join(self.dat, 'ElamDB12.txt'), errors='ignore') as fh:
            for line in fh:
                t = line.split()
                if not t:
                    continue
                if t[0] == 'Element':
                    cur = int(t[2])
                    self.sym[cur] = t[1]
                    self.z[t[1].lower()] = cur
                    self.A[cur] = float(t[3])
                    self.rho[cur] = float(t[4])
                elif t[0] == 'Edge' and cur is not None:
                    sh = t[1]
                    self.eedge[(cur, sh)] = float(t[2]) / 1000.0
                    self.omega[(cur, sh)] = float(t[3])
                    self.jump[(cur, sh)] = float(t[4])

    # -- edge.txt: the table 'edge()' actually uses -------------------------

    EDGE_NAMES = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
                  'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7',
                  'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3']

    def _edges(self):
        """Fixed-width, three lines per element, 11 columns of width 11."""
        self.edge = {}
        path = os.path.join(self.dat, 'edge.txt')
        lines = open(path, errors='ignore').read().splitlines()
        i = 0
        while i + 2 < len(lines):
            s1, s2, s3 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
            try:
                Z = int(s1[0:4])
            except ValueError:
                continue
            if not (1 <= Z <= 94):
                continue
            for k in range(11):
                for row, off in ((s1, 0), (s2, 11), (s3, 22)):
                    seg = row[4 + 11 * k: 4 + 11 * (k + 1)].strip()
                    idx = k + off
                    if seg and idx < 24:
                        try:
                            self.edge[(Z, self.EDGE_NAMES[idx])] = float(seg) / 1000.0
                        except ValueError:
                            pass
        # apply the supplement's edge corrections if present
        sup = os.path.join(self.dat, 'xray_lines_light.txt')
        if os.path.exists(sup):
            for line in open(sup, errors='ignore'):
                t = line.split(';')[0].split('#')[0].split()
                if len(t) >= 4 and t[0].lower() == 'edge':
                    Z = self.z.get(t[1].lower())
                    if Z:
                        self.edge[(Z, t[2].upper())] = float(t[3])

    # -- x-ray lines, main table + light-element supplement ------------------

    def _lines(self):
        """energy[Z][mnemonic] and relint[Z][mnemonic], keV."""
        self.lineE = {}
        self.lineI = {}
        path = os.path.join(self.dat, getattr(self, 'lines_file', 'xray_lines.txt'))
        Z = None
        for raw in open(path, errors='ignore'):
            s = ' '.join(raw.split())
            if not s:
                continue
            t = s.split(' ')
            if len(t) == 1:
                Z = self.z.get(t[0].lower())
                continue
            if Z is None:
                continue
            for k in range(0, len(t) - 2, 3):
                nm = t[k]
                try:
                    e, r = float(t[k + 1]), float(t[k + 2])
                except ValueError:
                    continue
                self.lineE.setdefault(Z, {})[nm] = e
                self.lineI.setdefault(Z, {})[nm] = r
        sup = os.path.join(self.dat, 'xray_lines_light.txt')
        if os.path.exists(sup):
            for raw in open(sup, errors='ignore'):
                t = raw.split(';')[0].split('#')[0].split()
                if len(t) >= 5 and t[0].lower() == 'line':
                    Z = self.z.get(t[1].lower())
                    if Z:
                        self.lineE.setdefault(Z, {})[t[2]] = float(t[3])
                        self.lineI.setdefault(Z, {})[t[2]] = float(t[4])

    # -- K-shell ionisation cross sections, Cohen & Harrigan 1985 -----------

    def _xsect_k(self):
        self._xsect_shell('K')
        for sh in ('L', 'M'):
            try:
                self._xsect_shell(sh)
            except Exception:
                pass                # optional; absent on older databases

    def _xsect_shell(self, shell):
        """sigma[Z] on the shared energy grid, cm^2, tabulated per 1 amu.

        xsect_L.txt and xsect_M.txt use the same layout as xsect_K.txt and
        give the TOTAL shell cross section - they are not subshell-resolved
        (init_xsect.pro declares sig_L as fltarr(nen,nz), one curve per
        element), so an L-subshell split has to come from elsewhere.
        """
        path = os.path.join(self.dat, 'xsect_%s.txt' % shell)
        tok = open(path, errors='ignore').read().split()
        p = 0
        nz = int(tok[p]); p += 1
        p += 1                      # 'Energy'
        _, _, nen = int(tok[p]), int(tok[p + 1]), int(tok[p + 2]); p += 3
        grid = [_f(x) for x in tok[p:p + nen]]; p += nen
        tab = {}
        for _ in range(nz):
            p += 1                  # element label
            Z = int(tok[p]); p += 3
            tab[Z] = [_f(x) for x in tok[p:p + nen]]; p += nen
        # grid is descending in MeV; flip to ascending for interpolation
        if grid[0] > grid[-1]:
            grid = grid[::-1]
            for Z in tab:
                tab[Z] = tab[Z][::-1]
        if shell == 'K':
            self.xk_e, self.xk = grid, tab
        setattr(self, 'x%s_e' % shell.lower(), grid)
        setattr(self, 'x%s' % shell.lower(), tab)

    def sigma_K(self, Z, E_MeV, A1=1.0, Z1=1):
        """K-shell ionisation cross section, cm^2."""
        return self.sigma_shell(Z, E_MeV, 'K', A1, Z1)

    def sigma_shell(self, Z, E_MeV, shell='K', A1=1.0, Z1=1):
        """Ionisation cross section for a shell, cm^2.

        The tables are for a unit-charge, 1 amu projectile. Two scalings
        turn them into any light ion, both from ion_xsect_pixe.pro:
          * the energy is divided by A1, since ionisation depends on the
            projectile VELOCITY and the table is indexed by energy per amu
          * the result is multiplied by Z1^2, the first-Born charge scaling
        For 2 MeV 4He that is a lookup at 0.5 MeV/amu times four. Before
        this the Z1^2 was missing, so any helium yield was low by 4x.
        L and M are TOTAL shell cross sections, not subshell-resolved."""
        tab = getattr(self, 'x%s' % shell.lower(), None)
        grid = getattr(self, 'x%s_e' % shell.lower(), None)
        if not tab or Z not in tab:
            return 0.0
        e = E_MeV / A1
        z1sq = float(Z1) ** 2
        g, s = grid, tab[Z]
        if e <= g[0] or e >= g[-1]:
            return 0.0
        for i in range(1, len(g)):
            if g[i] >= e:
                if s[i - 1] <= 0 or s[i] <= 0:
                    return 0.0
                f = ((math.log(e) - math.log(g[i - 1])) /
                     (math.log(g[i]) - math.log(g[i - 1])))
                return z1sq * math.exp(math.log(s[i - 1]) + f *
                                       (math.log(s[i]) - math.log(s[i - 1])))
        return 0.0

    # -- proton stopping, Andersen-Ziegler as in dedx.pro -------------------

    def _proton(self):
        self.pcoef = {}
        path = os.path.join(self.dat, 'proton.txt')
        for Z, line in enumerate(open(path, errors='ignore'), start=1):
            v = [_f(x) for x in line.split(',')]
            if len(v) >= 14:
                self.pcoef[Z] = v
        # Ziegler helium coefficients, one row per element, used by dedx()
        # for Z1 = 2 exactly as dedx.pro's DEDXA branch does.
        self.acoef = {}
        apath = os.path.join(self.dat, 'alpha.txt')
        if os.path.exists(apath):
            for Z, line in enumerate(open(apath, errors='ignore'), start=1):
                v = [_f(x) for x in line.split(',')]
                if len(v) >= 5:
                    self.acoef[Z] = v

    def _dedxp(self, eprot, A):
        """Hydrogen electronic stopping, eV/(1e15 atoms/cm^2)."""
        betasq = eprot / 465740.5
        if eprot < 10.0:
            return A[0] * math.sqrt(eprot)
        if eprot > 1000.0:
            tot = sum(A[7 + i] * math.log(eprot) ** i for i in range(5))
            return (A[5] / betasq) * (math.log(A[6] * betasq / (1.0 - betasq))
                                      - betasq - tot)
        slow = A[1] * eprot ** 0.45
        shigh = (A[2] / eprot) * math.log(1.0 + (A[3] / eprot) + A[4] * eprot)
        return 1.0 / (1.0 / slow + 1.0 / shigh)

    def _dedxn(self, Z1, Z2, A1, A2, E_eV):
        """Nuclear stopping, eV/(1e15 atoms/cm^2)."""
        d1 = Z1 ** 0.667 + Z2 ** 0.667
        d2 = A1 + A2
        d3 = Z1 * Z2
        d4 = math.sqrt(d1)
        eps = 32.53 * A2 * E_eV / (d3 * d2 * d4)
        conv = 8.462 * d3 * A1 / (d2 * d4)
        if eps < 0.01:
            return 1.593 * conv * math.sqrt(eps)
        if eps <= 10.0:
            return (1.7 * conv * math.sqrt(eps) * math.log(eps + math.e) /
                    (1.0 + 6.8 * eps + 3.4 * eps ** 1.5))
        return conv * math.log(0.47 * eps) / (2.0 * eps)

    def _dedxa(self, EK, A):
        """Helium electronic stopping, dedx.pro's DEDXA.

        EK is the energy in keV already scaled to 4.0026 amu, i.e.
        A1 * E_keV / 4.0026, so the same table serves 3He and 4He.
        """
        small = 1.0e-10
        E = EK / 1000.0
        if E <= 0:
            return 0.0
        SL = A[0] * EK ** A[1]
        SH = A[2] * math.log(1.0 + A[3] / E + A[4] * E) / E
        if SL < small:
            return SL
        if SH < small:
            return SH
        return 1.0 / (1.0 / SL + 1.0 / SH)

    def dedx(self, Z1, A1, Z2, E_MeV):
        """Stopping power in MeV/(mg/cm^2), for protons (Z1 = 1) and
        helium (Z1 = 2), following dedx.pro. Heavier ions are not handled
        and return 0 rather than a guess."""
        E_keV = 1000.0 * E_MeV
        A2 = self.A[Z2]
        if Z1 == 1:
            if Z2 not in self.pcoef:
                return 0.0
            se = self._dedxp(E_keV / A1, self.pcoef[Z2])
        elif Z1 == 2:
            if Z2 not in self.acoef:
                return 0.0
            se = self._dedxa(A1 * E_keV / 4.0026, self.acoef[Z2])
        else:
            return 0.0
        sn = self._dedxn(Z1, Z2, A1, A2, E_keV)
        return (se + sn) * DEDX_CONV / A2

    def dedx_compound(self, Z1, A1, zlist, wfrac, E_MeV):
        """Bragg additivity on mass fractions."""
        return sum(w * self.dedx(Z1, A1, Z, E_MeV) for Z, w in zip(zlist, wfrac))

    # -- mass attenuation coefficients --------------------------------------

    #: Every accepted spelling of a tabulated MAC database.
    #: Aliases matter here: the file is called MAC_SabbatucciSalvat2016.txt,
    #: nobody types that, and the bare KeyError it used to raise looked like
    #: a missing file rather than a misspelt name.
    MAC_FILES = {
        'henke1993': 'MAC_Henke1993.txt',
        'henke': 'MAC_Henke1993.txt',
        'sabbatuccisalvat2016': 'MAC_SabbatucciSalvat2016.txt',
        'sabbatuccisalvat': 'MAC_SabbatucciSalvat2016.txt',
        'sabbatucci': 'MAC_SabbatucciSalvat2016.txt',
        'salvat': 'MAC_SabbatucciSalvat2016.txt',
        'ffast': 'MAC_FFAST.txt',
        'chantler': 'MAC_FFAST.txt',
    }

    #: Names handled inside mu() rather than by loading a table.
    MAC_COMPUTED = ('xcom', 'hubbell', 'berger-hubbell', 'mixed')

    @classmethod
    def mac_datasets(cls):
        """Every dataset name mu() will accept, sorted."""
        return sorted(set(cls.MAC_FILES) | set(cls.MAC_COMPUTED))

    def load_mac(self, name):
        """Load one of the tabulated MAC databases written for GeoPIXE."""
        key = name.lower()
        if key not in self.MAC_FILES:
            raise KeyError(
                'unknown MAC dataset %r. Valid names are: %s'
                % (name, ', '.join(self.mac_datasets())))
        fn = self.MAC_FILES[key]
        tab = {}
        cur = None
        mode = 0
        for raw in open(os.path.join(self.dat, fn), errors='ignore'):
            t = raw.strip()
            if not t or t.startswith(';'):
                continue
            w = t.split()
            if w[0] == 'Element':
                cur = int(w[2]); tab[cur] = ([], []); mode = 0
            elif w[0] == 'Data':
                mode = 1
            elif w[0] == 'EndElement':
                mode = 0
            elif mode == 1 and len(w) == 2:
                tab[cur][0].append(float(w[0]))
                tab[cur][1].append(float(w[1]))
        self._mac[key] = tab
        return tab

    # Below this energy no XCOM-class tabulation supplies anything, so the
    # mixed rule has no choice of source.
    MIXED_LOW_KEV = 1.0

    def mu(self, Z, E_keV, dataset='henke1993'):
        """mu/rho in cm^2/g. Edge-aware: never interpolates across a repeated
        abscissa. Returns None out of range rather than extrapolating.

        dataset='mixed' applies Heirwegh's (2014) selection rule, which beat
        either single database on silicate standards:

            FFAST  below the absorber's own K edge, and everywhere below 1 keV
            other  above the K edge   (self.mixed_above, default Sabbatucci)

        The rule is not arbitrary. Below 1 keV, XCOM and Scofield tabulate
        nothing at all, so FFAST is the only option. Below the K edges of Mg,
        Al and Si, Chantler's DHS photo-absorption cross sections are strongly
        favoured over Scofield's by experiment. Above the edges the evidence
        runs the other way. Using the mixed selection took the pure-element to
        silicate-oxide discrepancy from 6-8% to under 2%, and under 1% for Mg
        and Si.

        The above-edge partner is XCOM, matching Heirwegh exactly. XCOM is
        the Berger-Hubbell evaluation, which GeoPIXE already ships as
        hubbell.dat, so nothing had to be imported for this.
        """
        key = dataset.lower()
        if key in ('xcom', 'hubbell', 'berger-hubbell'):
            # XCOM is the Berger-Hubbell evaluation; GeoPIXE ships it as
            # hubbell.dat, so no separate import is needed.
            if not hasattr(self, '_hub'):
                from ..io.hubbell import read_hubbell
                self._hub = read_hubbell(self.dat)
            t = self._hub.get(Z)
            if t is None:
                return None
            E, M = t
            if E_keV < E[0] or E_keV > E[-1]:
                return None
            return float(math.exp(
                __import__('numpy').interp(math.log(E_keV),
                                           __import__('numpy').log(E),
                                           __import__('numpy').log(M))))
        if key == 'mixed':
            k_edge = self.edge.get((Z, 'K'), 0.0)
            if E_keV < self.MIXED_LOW_KEV or (k_edge > 0 and E_keV < k_edge):
                v = self.mu(Z, E_keV, 'ffast')
                if v is not None:
                    return v
                return self.mu(Z, E_keV, 'henke1993')
            above = getattr(self, 'mixed_above', 'xcom')
            v = self.mu(Z, E_keV, above)
            return v if v is not None else self.mu(Z, E_keV, 'ffast')
        if key not in self._mac:
            self.load_mac(key)
        tab = self._mac[key]
        if Z not in tab:
            return None
        E, M = tab[Z]
        if E_keV < E[0] or E_keV > E[-1]:
            return None
        for i in range(1, len(E)):
            if E[i] >= E_keV:
                j = i - 1
                while j > 0 and E[j + 1] == E[j]:
                    j -= 1
                if E[j + 1] == E[j]:
                    return M[j + 1]
                if E_keV == E[j]:
                    return M[j + 1] if (j + 1 < len(E) and E[j + 1] == E[j]) else M[j]
                f = ((math.log(E_keV) - math.log(E[j])) /
                     (math.log(E[j + 1]) - math.log(E[j])))
                return math.exp(math.log(M[j]) + f *
                                (math.log(M[j + 1]) - math.log(M[j])))
        return None

    def mu_compound(self, zlist, wfrac, E_keV, dataset='henke1993'):
        """Bragg additivity on mass fractions, cm^2/g. None if any term is
        out of range - a partial sum would be silently wrong."""
        tot = 0.0
        for Z, w in zip(zlist, wfrac):
            m = self.mu(Z, E_keV, dataset)
            if m is None:
                return None
            tot += w * m
        return tot

    # -- fluorescence yield --------------------------------------------------

    KRAUSE = [
        (8.51051E-2, 2.63414E-2, 1.63531E-4, -1.85999E-6),   # K
        (1.70027E-1, 2.98746E-3, 8.70636E-5, -2.19916E-7),   # L1
        (2.72447E-1, -9.47505E-4, 1.37650E-4, -4.64780E-7),  # L2
        (1.77650E-1, 2.98937E-3, 8.91297E-5, -2.67184E-7),   # L3
        (-3.25820E-1, 9.01791E-3, 3.80983E-5, -4.62897E-7),  # M4
    ]

    def fluor_yield(self, Z, shell='K', source='krause', elam_zmax=10):
        """PIXE fluorescence yield.

        'krause' reproduces GeoPIXE's default exactly. 'elam' substitutes the
        Elam tabulation for Z <= elam_zmax, where the Krause polynomial (fitted
        for Z = 15-92) is extrapolating.
        """
        if source.lower() == 'elam' and Z <= elam_zmax:
            return self.omega.get((Z, shell), 0.0)
        idx = {'K': 0, 'L1': 1, 'L2': 2, 'L3': 3}.get(shell, 4)
        c = self.KRAUSE[idx]
        v = (c[0] + Z * (c[1] + Z * (c[2] + Z * c[3]))) ** 4
        return v / (1.0 + v)

    def m_subshell_lines(self, Z, min_rate=0.002):
        """M-shell transitions grouped by initial subshell, from xraylib.

        {'M1': [(E, rate)], ..., 'M5': [...]} with rates normalised within
        each subshell. Returns {} when xraylib is not installed or Z has no
        M lines above 1 keV.

        Why this exists. The line table carries five M lines - Ma, Mb, Mg,
        Mz and one more - which is the M4 and M5 story only. The M1, M2 and
        M3 subshells radiate too: for gold, M2-N4 at 2.796, M1-N2 at 2.781,
        M3-N1 at 1.984, M2-N1 at 2.389, M1-N3 at 2.880 keV. On a 2 MeV
        helium spectrum with 983,000 Au M counts those lines are thousands
        of counts each, and they sit on Si Ka and the Ti escape. Fitting the
        five-line table gave chi2 43.6 with silicon driven to zero; the full
        series, one free component per subshell, gave 5.6 and Si 1623.

        The relative POPULATION of the subshells depends on the projectile
        and is not in the radiative rates, which is exactly why they are
        returned per subshell - so each can be a component of its own and
        the data sets the ratio, as GeoPIXE already does for L1, L2, L3.
        """
        try:
            import xraylib as xl
        except ImportError:
            return {}
        import re
        out = {}
        for name in dir(xl):
            m = re.match(r'^(M[1-5])[NOP]\d_LINE$', name)
            if not m:
                continue
            try:
                e = xl.LineEnergy(Z, getattr(xl, name))
                r = xl.RadRate(Z, getattr(xl, name))
            except Exception:
                continue
            if e > 1.0 and r > min_rate:
                out.setdefault(m.group(1), []).append((float(e), float(r)))
        # The Mz transitions - M4-N2, M4-N3, M5-N3 - are pulled out into a
        # group of their own. xraylib's radiative rates put them near 8% of
        # the M4+M5 decay; the gold data reads 0.6%. Tied at the rate they
        # dump ~80,000 counts at 1.65 keV on a spectrum that holds ~6,000
        # there and take silicon with them. As a free group the data sets
        # them, which is the point of splitting at all.
        mz = []
        for k in ('M4', 'M5'):
            keep = []
            for e, r in out.get(k, []):
                if e < 1.9:
                    mz.append((e, r))
                else:
                    keep.append((e, r))
            if keep:
                out[k] = keep
            elif k in out:
                del out[k]
        if mz:
            out['Mz'] = mz
        for k, v in list(out.items()):
            t = sum(r for _, r in v)
            out[k] = sorted([(e, r / t) for e, r in v], key=lambda t2: -t2[1])
        return out

    # -- convenience ---------------------------------------------------------

    # GeoPIXE's line-splitting thresholds, from line_split_definitions.def
    KA_S, KB_S, LA_S = 27, 19, 52

    def line_list(self, Z, shell=1):
        """Lines for one shell, following list_line_index.pro.

        Below the split thresholds GeoPIXE uses the unresolved GROUP entries
        (Ka_, Kb_) rather than the individual Ka1/Ka2/Kb1... members. Mixing
        the two double-counts intensity, so this mirrors the rule exactly.
        Returned intensities are renormalised to sum to 1.
        """
        E = self.lineE.get(Z, {})
        I = self.lineI.get(Z, {})
        names = []
        if shell <= 1:
            names += ['Ka_'] if Z <= self.KA_S else ['Ka1', 'Ka2', 'Ka3']
            names += ['Kb_'] if Z <= self.KB_S else ['Kb1', 'Kb2', 'Kb3',
                                                     'Kb4', 'Kb5']
        if shell == 2:
            names += ['La_'] if Z <= self.LA_S else ['La1', 'La2']
            names += ['Lb_', 'Lg_'] if Z <= self.LA_S else [
                'Lb1', 'Lb2', 'Lb3', 'Lb4', 'Lb5', 'Lb6',
                'Lg1', 'Lg2', 'Lg3']
            names += ['Ll', 'Leta']
        if shell == 3:
            names += ['Ma1', 'Ma2', 'Mb_', 'Mg_', 'Mz_']
        out = []
        for nm in names:
            e, i = E.get(nm, 0.0), I.get(nm, 0.0)
            if e > 0.01 and i > 1.0e-6:
                out.append((e, i))
        if not out:
            return []
        s = sum(i for _, i in out)
        return [(e, i / s) for e, i in out]

    def major_line(self, Z, shell=1):
        """GeoPIXE uses the Ka_ group below Z=27, split Ka1/2/3 above."""
        return 'Ka_' if Z <= 27 else 'Ka1'

    def line_energy(self, Z, shell=1):
        nm = self.major_line(Z, shell)
        return self.lineE.get(Z, {}).get(nm, 0.0)

    def N_A_over_A(self, Z):
        """Atoms per gram / 1e24, the factor turning a cross section in cm^2
        into a yield per unit mass fraction."""
        return N_A / self.A[Z]

    def formula_to_mass_fractions(self, zlist, atom_frac):
        m = [self.A[Z] * f for Z, f in zip(zlist, atom_frac)]
        s = sum(m)
        return [x / s for x in m]
