"""Rebuild the L and M sections of xray_lines.txt from ElamDB12.

Why
---
The shipped xray_lines.txt has demonstrable errors in its L section:
  * Ca, Sc, Ti carry an identical fabricated 100:10:1:1 La:Lb:Ll:Leta pattern
  * Ca and Sc are given 89% La, a transition they CANNOT make - L3->M4,5 needs
    a populated 3d shell and theirs is empty. Their L3 emission is pure Ll.
  * Ll is understated by 11-13x for Ti-Mn and 112x for Ca/Sc
  * Mn's Lb/La = 0.301 breaks the Z sequence (Cr 0.201, Fe 0.205, Co 0.185)
  * At, Rn, Ac have Ll and Leta set to exactly zero

Elam has all of this correctly, filed by originating subshell.

The one thing Elam cannot give
------------------------------
xray_lines.txt stores L intensities normalised across ALL THREE subshells
combined. Collapsing Elam's per-subshell branching into that form requires
knowing how the vacancies are distributed between L1, L2 and L3 - and that
depends on the projectile, which is exactly why a single flat table is the
wrong representation in the first place.

GeoPIXE's xsect_L.txt is not subshell-resolved (sig_L is fltarr(nen,nz), one
curve per element), so it cannot supply the split either. This rebuild
therefore uses an explicit, documented assumption:

    statistical ionisation, 2j+1     ->  L1:L2:L3 = 2:2:4
    propagated through Coster-Kronig ->  n1, n2, n3
    weighted by subshell yield       ->  I_s = n_s * omega_s
    distributed by Elam branching    ->  per-line intensity

That is defensible and vastly better than a placeholder, but it is still an
assumption. For any fit where L lines matter, use the subshell-resolved path
(subshell.py) instead, which carries no such assumption - it lets the three
populations float and locks only the ratios that are genuinely constant.

Output goes to a NEW file. Nothing GeoPIXE currently reads is overwritten.
"""
import sys, os
sys.path.insert(0, '.')
from elamdb import ElamDB
import xrl_lines as X
import xraylib

DAT = r'C:\Users\Charles\Desktop\GeoPIXE-main\Workspace\main\database\dat'
SRC = os.path.join(DAT, 'xray_lines.txt')
OUT = os.path.join(DAT, 'xray_lines_rebuilt.txt')

# mnemonic -> Elam line name, per subshell
L_MAP = {
    'L1': {'Lb3': 'Lb3', 'Lb4': 'Lb4', 'Lg2': 'Lg2', 'Lg3': 'Lg3',
           'Lg4': 'Lg4'},
    'L2': {'Leta': 'Ln', 'Lb1': 'Lb1', 'Lg1': 'Lg1', 'Lg5': 'Lg5',
           'Lg6': 'Lg6'},
    'L3': {'Ll': 'Ll', 'La1': 'La1', 'La2': 'La2', 'Lb2': 'Lb2,15',
           'Lb5': 'Lb5', 'Lb6': 'Lb6'},
}
M_MAP = {'M3': {'Mg_': 'Mg'}, 'M4': {'Mb_': 'Mb', 'Mz_': 'Mz'},
         'M5': {'Ma1': 'Ma', 'Ma2': 'Ma'}}
STAT = {'L1': 2.0, 'L2': 2.0, 'L3': 4.0}


def l_weights(E, sym, Z):
    """Relative TOTAL emission from each L subshell, statistical + CK + omega.

    Coster-Kronig factors come from xraylib, not Elam: Elam gives a non-zero
    f23 for the 3d metals (0.25 Ti, 0.42 Fe, 0.45 Ni) but that transition is
    energetically forbidden below about Z=30, and xraylib correctly returns 0.
    """
    f12 = X.ck(Z, 'L1', 'L2'); f13 = X.ck(Z, 'L1', 'L3')
    f23 = X.ck(Z, 'L2', 'L3')
    n1 = STAT['L1']
    n2 = STAT['L2'] + f12 * n1
    n3 = STAT['L3'] + f13 * n1 + f23 * n2
    w = {'L1': n1 * X.omega(Z, 'L1'),
         'L2': n2 * X.omega(Z, 'L2'),
         'L3': n3 * X.omega(Z, 'L3')}
    t = sum(w.values())
    return {k: (v / t if t > 0 else 0.0) for k, v in w.items()}


def m_weights(E, sym, Z):
    w = {s: X.omega(Z, s) for s in ('M3', 'M4', 'M5')}
    t = sum(w.values())
    return {k: (v / t if t > 0 else 0.0) for k, v in w.items()}


# every L/M mnemonic this rebuild is responsible for. Any of these that Elam
# does NOT list must be written as zero, not left at its old value - otherwise
# Ca and Sc keep the 89% La they physically cannot emit.
OWNED = (['Ll', 'Leta', 'La1', 'La2', 'Lb1', 'Lb2', 'Lb3', 'Lb4', 'Lb5',
          'Lb6', 'Lg1', 'Lg2', 'Lg3', 'Lg4', 'Lg5', 'Lg6', 'La_', 'Lb_',
          'Lg_'] + ['Ma1', 'Ma2', 'Mb_', 'Mg_', 'Mz_'])


def build(E, sym, Z):
    """{mnemonic: (energy_keV, relint)} for the L and M mnemonics.

    Line energies and intra-subshell branching come from xraylib (Campbell &
    Wang Dirac-Fock rates); Elam is kept only as a fallback for a subshell
    xraylib does not populate. Note xraylib gives the real Ma1:Ma2 split
    (0.936:0.064 for Pb) where Elam has an exact 50:50 placeholder.
    """
    out = {m: (0.0, 0.0) for m in OWNED}
    lw = l_weights(E, sym, Z)
    for sh in ('L1', 'L2', 'L3'):
        got = X.lines(Z, sh)
        if not got:                     # fall back to Elam for this subshell
            byname = {nm: (en, r) for nm, en, r in E.lines(sym, sh)}
            for mnem, elname in L_MAP[sh].items():
                if elname in byname:
                    en, r = byname[elname]
                    out[mnem] = (en, r * lw[sh])
            continue
        for mnem, en, r in got:
            out[mnem] = (en, r * lw[sh])
    mw = m_weights(E, sym, Z)
    for sh in ('M3', 'M4', 'M5'):
        for mnem, en, r in X.lines(Z, sh):
            out[mnem] = (en, r * mw[sh])
    # grouped mnemonics GeoPIXE uses below its split thresholds
    def grp(keys):
        tot = sum(out[k][1] for k in keys if k in out)
        if tot <= 0:
            return None
        en = sum(out[k][0] * out[k][1] for k in keys if k in out) / tot
        return (en, tot)
    for g, keys in (('La_', ('La1', 'La2')),
                    ('Lb_', ('Lb1', 'Lb2', 'Lb3', 'Lb4', 'Lb5', 'Lb6')),
                    ('Lg_', ('Lg1', 'Lg2', 'Lg3', 'Lg4', 'Lg5', 'Lg6'))):
        v = grp(keys)
        out[g] = v if v else (0.0, 0.0)
    return out


def main():
    E = ElamDB(DAT)
    lines = open(SRC, errors='ignore').read().splitlines()
    out_lines, cur, changed = [], None, 0
    ORDER = None
    blocks = []          # (symbol, {mnem: (e, r)}, original order)
    i = 0
    while i < len(lines):
        s = ' '.join(lines[i].split())
        if s and len(s.split(' ')) == 1:
            cur = s
            vals, order = {}, []
            j = i + 1
            while j < len(lines):
                t = ' '.join(lines[j].split()).split(' ')
                if len(t) < 3 or len(t) == 1:
                    break
                for k in range(0, len(t) - 2, 3):
                    vals[t[k]] = (float(t[k + 1]), float(t[k + 2]))
                    order.append(t[k])
                j += 1
            blocks.append((cur, vals, order))
            i = j
        else:
            i += 1
    with open(OUT, 'w') as fh:
        fh.write('# xray_lines.txt with the L and M sections rebuilt from '
                 'ElamDB12.txt\n')
        fh.write('# K lines are unchanged. See rebuild_lines.py for the '
                 'subshell-weighting assumption.\n')
        for sym, vals, order in blocks:
            try:
                Z = xraylib.SymbolToAtomicNumber(sym)
            except Exception:
                Z = 0
            new = build(E, sym, Z) if Z > 0 else {}
            n_ch = 0
            for mnem, (en, r) in new.items():
                if mnem in vals and (abs(vals[mnem][1] - r) > 1e-5
                                     or abs(vals[mnem][0] - en) > 5e-4):
                    n_ch += 1
                vals[mnem] = (round(en, 4), round(r, 6))
            changed += 1 if n_ch else 0
            fh.write('%s \n' % sym)
            row = []
            for mnem in order:
                e_, r_ = vals.get(mnem, (0.0, 0.0))
                row.append('%-6s %.4f %.6f' % (mnem, e_, r_))
                if len(row) == 6:
                    fh.write(' '.join(row) + '\n'); row = []
            if row:
                fh.write(' '.join(row) + '\n')
            fh.write('\n')
    print('wrote %s' % OUT)
    print('elements with changed L/M values: %d of %d' % (changed, len(blocks)))


if __name__ == '__main__':
    main()
