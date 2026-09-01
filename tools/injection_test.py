"""Spike-recovery test for the subshell-tied deconvolution.

The demonstration a referee would ask for. Take a REAL spectrum whose answer
is known, inject a KNOWN amount of an interfering element, and see how much
each method recovers.

Host: run 287427, a quartz (SiO2) with no significant heavy-element content.
Real counting statistics, real background, real peak shapes - only the spike
is synthetic, and its amount is known exactly.

Two methods compared:
  flat      one free area per element, line ratios from xray_lines.txt
  subshell  three free areas per element (L1/L2/L3), intra-shell ratios locked
            to Elam, then screened by Coster-Kronig and M/L corroboration

Three things are measured:
  recovery      injected vs recovered, at several spike sizes
  false positive  what each method claims when NOTHING was injected
  cross-talk    what the spike does to OTHER elements' recovered areas

On circularity: the spike is generated with Elam line structure, which is the
same structure the subshell fit assumes, so the shape comparison favours it by
construction. That is stated rather than hidden - and it is why the false
positive and cross-talk numbers matter more than the recovery number. Those
two do not depend on which line list generated the spike.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import fit as F
from compare_le import DB, snip
from subshell import shell_components
from screen import screen
from gpda import read_dam
import lfix
import refit_all as R

EL = R.EL
HOST = '287427'
HOST_DAM = 'F:/GeoPIXE/topaz.dam'
BASE_Z = [6, 7, 8, 9, 11, 12, 13, 14, 17, 19, 20]
NOISE, FANO = 32.74, 28.05


def make_spike(sym, total_counts, pars, n):
    """A synthetic element built from Elam subshell structure.

    The shape parameters come from a real fit to the host spectrum, so the
    spike has exactly the same resolution, tail and calibration as the data it
    is injected into. Subshells are weighted by their fluorescence yields.
    """
    w = {sh: EL.omega(sym, sh) for sh in ('L1', 'L2', 'L3')}
    tot_w = sum(w.values()) or 1.0
    spec = np.zeros(n)
    for sh in ('L1', 'L2', 'L3'):
        lines = [(en, r) for _, en, r in EL.lines(sym, sh) if 0.2 < en < 6.5]
        if not lines:
            continue
        c = F.Component(sym + sh, lines)
        c.tail_amp_fn = lambda E: 0.08
        c.tail_len_fn = lambda E: 1.0
        spec += total_counts * (w[sh] / tot_w) * c.profile(pars, n)
    return spec


def fit_both(S, A, B, heavy):
    bk = snip(S, A, B, 0.20, 6.50, passes=3, use_low_stats=True)
    out = {}
    for mode in ('flat', 'subshell'):
        comps = [c for c in (R.kcomp(Z) for Z in BASE_Z) if c]
        for sym in heavy:
            if mode == 'flat':
                Z = DB.z[sym.lower()]
                L = DB.line_list(Z, 2)
                if L:
                    c = F.Component(sym + 'L', L)
                    c.tail_amp_fn = lambda E: 0.08
                    c.tail_len_fn = lambda E: 1.0
                    comps.append(c)
            else:
                comps += shell_components(EL, sym, F.Component,
                                          shells=('L1', 'L2', 'L3'))
        r = F.fit_spectrum(S, A, B, comps, 0.20, 6.50, noise=NOISE, fano=FANO,
                           tail_amp=0.08, tail_len=1.0, background=bk,
                           refine=('cal', 'width', 'tail'), nonneg='strict')
        a = dict(zip(r.names, r.areas))
        e = dict(zip(r.names, r.errors))
        if mode == 'flat':
            got = {s: a.get(s + 'L', 0.0) for s in heavy}
            err = {s: e.get(s + 'L', 0.0) for s in heavy}
        else:
            got = {s: sum(a.get('%s_L%d' % (s, i), 0.0) for i in (1, 2, 3))
                   for s in heavy}
            err = {s: np.sqrt(sum(e.get('%s_L%d' % (s, i), 0.0) ** 2
                                  for i in (1, 2, 3))) for s in heavy}
        out[mode] = dict(area=a, err=e, got=got, goterr=err, chi2=r.reduced_chi2)
    return out
