"""Physical screening of L/M components, applied per SUBSHELL GROUP.

The three tests, in the order they are applied:

  1. Coster-Kronig floor. Vacancies created in L1 and L2 cannot all stay
     there; f13 and f23 force a minimum L3 population. A fit that puts
     intensity into L1/L2 while leaving L3 empty is describing something that
     cannot happen, so the element's L assignment is rejected outright.

  2. M/L corroboration. When both L and M lines are in the window, the M
     intensity is bounded by the L intensity through the fluorescence yields.
     Failing this rejects only the M COMPONENTS - not the element. In and I in
     a perovskite are real; it is their 0.4-0.8 keV M lines that are absorbing
     unmodelled oxygen structure. Dropping the whole element instead sent chi2
     from 3.3 to 65.

  3. Significance. What survives must be a >3 sigma detection.

The distinction in (2) matters more than it looks. Sub-keV M lines from heavy
elements and sub-keV L lines from 3d metals are the same trap: both are weak,
both land on the shoulder of an enormous oxygen peak, and both will happily
absorb whatever the peak-shape model gets wrong.
"""

from .subshell import ck_consistency, ml_corroboration


def screen(elam, symbols, areas, errors, min_sigma=3.0):
    """Return (keep_L, keep_M, rejected) after the three tests."""
    keep_L, keep_M, rej = [], [], []
    for sym in symbols:
        lp = {'L%d' % i: areas.get('%s_L%d' % (sym, i), 0.0) for i in (1, 2, 3)}
        mp = {'M%d' % i: areas.get('%s_M%d' % (sym, i), 0.0) for i in (3, 4, 5)}
        has_L, has_M = any(v > 0 for v in lp.values()), any(v > 0 for v in mp.values())
        if not has_L and not has_M:
            rej.append((sym, 'all', 'no intensity'))
            continue

        okL = has_L
        if has_L:
            ck = ck_consistency(elam, sym, lp['L1'], lp['L2'], lp['L3'])
            if not ck['n3_ok']:
                rej.append((sym, 'L', 'Coster-Kronig: n(L3)=%.0f below forced floor %.0f'
                            % (lp['L3'], ck['n3_floor'])))
                okL = False

        okM = has_M
        if has_M:
            ml = ml_corroboration(elam, sym, lp if okL else {}, mp)
            if ml is None:
                pass                       # no L in window; cannot corroborate
            elif not ml['ok']:
                rej.append((sym, 'M', ml['reason']))
                okM = False

        def sig(prefixes):
            best = 0.0
            for k, v in areas.items():
                if any(k == '%s_%s' % (sym, p) for p in prefixes) and v > 0:
                    best = max(best, v / max(errors.get(k, 1e-9), 1e-9))
            return best

        if okL:
            s = sig(('L1', 'L2', 'L3'))
            if s < min_sigma:
                rej.append((sym, 'L', 'only %.1f sigma' % s))
            else:
                keep_L.append(sym)
        if okM:
            s = sig(('M3', 'M4', 'M5'))
            if s < min_sigma:
                rej.append((sym, 'M', 'only %.1f sigma' % s))
            else:
                keep_M.append(sym)
    return keep_L, keep_M, rej
