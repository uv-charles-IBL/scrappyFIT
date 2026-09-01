"""Do the K-shell and L-shell give the same concentration?

The strongest internal check available. For Ti through Ni at 1 MeV both shells
emit inside the 6.6 keV window, and the two routes share nothing except the
element being there:

    K route    sigma_K  x  omega_K  x  transmission at ~5-7 keV
    L route    sigma_L  x  omega_L  x  transmission at ~0.4-0.9 keV

Different cross sections, different fluorescence yields, and mass attenuation
coefficients that differ by two orders of magnitude between the two energies.
If they agree, the whole chain is consistent. If they do not, the disagreement
localises the error - and a ratio that is wild for one element but sane for
its neighbour means that element's L assignment is not real.

omega_L is the population-weighted mean of the three subshell yields, using
the same statistical + Coster-Kronig weighting as rebuild_lines.py, because
GeoPIXE's xsect_L.txt is not subshell-resolved.
"""
import sys, math
import numpy as np
from .gpdb import Database
from .elamdb import ElamDB

DAT = r'C:\Users\Charles\Desktop\GeoPIXE-main\Workspace\main\database'
STAT = {'L1': 2.0, 'L2': 2.0, 'L3': 4.0}


def omega_L_mean(E, sym):
    f12 = E.ck(sym, 'L1', 'L2'); f13 = E.ck(sym, 'L1', 'L3')
    f23 = E.ck(sym, 'L2', 'L3')
    n1 = STAT['L1']; n2 = STAT['L2'] + f12 * n1
    n3 = STAT['L3'] + f13 * n1 + f23 * n2
    num = n1 * E.omega(sym, 'L1') + n2 * E.omega(sym, 'L2') + n3 * E.omega(sym, 'L3')
    return num / (n1 + n2 + n3)


def line_energy_L(E, sym):
    """Intensity-weighted mean L emission energy, same weighting."""
    f12 = E.ck(sym, 'L1', 'L2'); f13 = E.ck(sym, 'L1', 'L3')
    f23 = E.ck(sym, 'L2', 'L3')
    n1 = STAT['L1']; n2 = STAT['L2'] + f12 * n1
    n3 = STAT['L3'] + f13 * n1 + f23 * n2
    num = den = 0.0
    for sh, n in (('L1', n1), ('L2', n2), ('L3', n3)):
        w = n * E.omega(sym, sh)
        for _, en, r in E.lines(sym, sh):
            num += w * r * en; den += w * r
    return num / den if den > 0 else 0.0


def predicted_LK(db, elam, Z, E0, zlist, wfrac, mac, cos_out, eff, n_steps=600):
    """Predicted L/K X-ray intensity ratio, depth-integrated."""
    sym = db.sym[Z]
    eK, eL = db.line_energy(Z), line_energy_L(elam, sym)
    if eK <= 0 or eL <= 0:
        return None
    wK, wL = db.fluor_yield(Z, 'K', 'krause', 10), omega_L_mean(elam, sym)
    muK = db.mu_compound(zlist, wfrac, eK, mac)
    muL = db.mu_compound(zlist, wfrac, eL, mac)
    if muK is None or muL is None:
        return None
    muK *= 1e-3 / cos_out; muL *= 1e-3 / cos_out
    dE = (E0 - 0.02) / n_steps
    x = 0.0; E = E0; iK = iL = 0.0
    for _ in range(n_steps):
        s = db.dedx_compound(1, 1.0, zlist, wfrac, E)
        if s <= 0:
            break
        dx = dE / s
        sK = db.sigma_shell(Z, E, 'K'); sL = db.sigma_shell(Z, E, 'L')
        iK += sK * math.exp(-muK * x) * dx
        iL += sL * math.exp(-muL * x) * dx
        x += dx; E -= dE
        if E <= 0.02:
            break
    if iK <= 0:
        return None
    return dict(ratio=(iL * wL * eff(eL)) / (iK * wK * eff(eK)),
                eK=eK, eL=eL, wK=wK, wL=wL,
                esc_K=cos_out / muK / cos_out, esc_L=cos_out / muL / cos_out)
