"""Subshell-resolved L and M components, with intra-shell ratios locked.

The deconvolution argument
--------------------------
A heavy element puts a dozen L lines into a few hundred eV. Fitting one free
area per element with flat tabulated ratios gets the shape wrong; fitting one
free area per LINE has no constraint at all and will happily trade intensity
with a neighbouring element.

The physics splits cleanly in two:

  * WITHIN a subshell, the branching ratios are atomic constants. They do not
    depend on the projectile, its energy, or the sample. Elam tabulates them
    normalised to 1 per subshell. Lock them.

  * BETWEEN subshells, the populations depend on the ionisation cross sections
    and on Coster-Kronig redistribution, which do depend on the projectile.
    Let those float - there are only three of them for L, five for M.

So a twelve-line L spectrum becomes three free parameters carrying the correct
internal structure, instead of one free parameter carrying the wrong one.

Validation handle: the fitted L1:L2:L3 populations should vary smoothly with Z
across elements measured in the same run, and L3 should dominate. A fit that
returns a wild population ratio is telling you the element is not really there.
"""

import numpy as np


def shell_components(elam, sym, Component, shells=('L1', 'L2', 'L3'),
                     e_lo=0.2, e_hi=6.6, min_ratio=1e-3,
                     tail_amp=0.08, tail_len=1.0):
    """One Component per subshell, lines locked to Elam branching ratios.

    Returns [] for a subshell with no line inside the window, so an element
    whose L3 is in range but whose L1 is not contributes only what it can.
    """
    out = []
    for sh in shells:
        lines = [(en, r) for _, en, r in elam.lines(sym, sh)
                 if e_lo < en < e_hi and r >= min_ratio]
        if not lines:
            continue
        c = Component('%s_%s' % (sym, sh), lines)
        c.tail_amp_fn = lambda E, a=tail_amp: a
        c.tail_len_fn = lambda E, l=tail_len: l
        out.append(c)
    return out


def ck_consistency(elam, sym, n1, n2, n3, Z=None):
    """Is the fitted population set consistent with Coster-Kronig?

    n3 must be at least f13*n1 + f23*n2 - vacancies that CANNOT avoid arriving
    in L3. A fit returning less than that is unphysical, which is a useful
    check on whether the element is really present.

    Factors come from xraylib, NOT Elam. Elam reports a non-zero f23 for the
    3d metals (0.25 Ti, 0.42 Fe, 0.45 Ni), but the L2-L3 Coster-Kronig channel
    is energetically forbidden below about Z=30 and xraylib correctly gives
    zero. Using Elam's value inflates the L3 floor and rejects real elements -
    it is why iron failed this test in the quartz.
    """
    from . import xrl_lines as _X
    import xraylib as _xl
    if Z is None:
        try:
            Z = _xl.SymbolToAtomicNumber(sym)
        except Exception:
            Z = 0
    if Z > 0:
        f12 = _X.ck(Z, 'L1', 'L2')
        f13 = _X.ck(Z, 'L1', 'L3')
        f23 = _X.ck(Z, 'L2', 'L3')
    else:
        f12 = elam.ck(sym, 'L1', 'L2')
        f13 = elam.ck(sym, 'L1', 'L3')
        f23 = elam.ck(sym, 'L2', 'L3')
    floor3 = f13 * n1 + f23 * n2
    floor2 = f12 * n1
    return dict(f12=f12, f13=f13, f23=f23,
                n3_floor=floor3, n3_ok=n3 >= 0.9 * floor3,
                n2_floor=floor2, n2_ok=n2 >= 0.9 * floor2)


def ml_corroboration(elam, sym, l_pops, m_pops, sigma_slack=10.0):
    """Is the fitted M intensity consistent with the fitted L intensity?

    For an element with both L and M lines inside the window, the two are not
    independent. Emission from a subshell scales as (vacancies x fluorescence
    yield), and for the same projectile the M-shell ionisation cross section
    exceeds the L-shell one by at most about an order of magnitude at these
    beam energies. So

        M_total / L3_total  <~  sigma_slack * (omega_M / omega_L3)

    Ag is the cautionary case: omega(M3)=3.2e-4 against omega(L3)=5.2e-2, so
    M3 should be ~0.6% of L3 even before cross sections. A fit returning M3 at
    230x L3 is not detecting silver - it is using a 0.569 keV line to absorb
    unmodelled structure near the oxygen peak.

    Returns None when the element has no L line in the window (Pb, Hg), where
    no corroboration is possible and the M assignment must be judged on other
    grounds.
    """
    l3 = l_pops.get('L3', 0.0)
    m = sum(m_pops.values())
    if m <= 0:
        return dict(ok=True, reason='no M intensity')
    if l3 <= 0 and not any(l_pops.values()):
        return None
    w_l3 = elam.omega(sym, 'L3')
    w_m = max(elam.omega(sym, s) for s in ('M3', 'M4', 'M5')) or 1e-12
    if l3 <= 0 or w_l3 <= 0:
        return dict(ok=False, reason='M intensity with no L3', ratio=float('inf'),
                    bound=0.0)
    bound = sigma_slack * (w_m / w_l3)
    ratio = m / l3
    return dict(ok=ratio <= bound, ratio=ratio, bound=bound,
                reason='M/L3 = %.3g, physical bound ~%.3g' % (ratio, bound))
