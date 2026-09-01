"""Corrections to the tabulated L-line intensities, applied explicitly.

Two defects in the shipped table matter for 3d-transition-metal L emission,
which is exactly what crowds the 0.4-0.9 keV region of a silicate:

1. Ca, Sc, Ti and V carry a 100:10:1:1 La:Lb:Ll:Leta pattern - identical for
   every one of them, and summing to 1.0 exactly. That is a placeholder, not a
   measurement. Real tabulated values begin at Cr.

2. Mn has Lb/La = 0.301 while its neighbours run Cr 0.201, Fe 0.205, Co 0.185.
   A single-element spike in an otherwise smooth Z sequence is a tabulation
   error. Interpolating Cr-Fe gives 0.203.

Both corrections are opt-in and logged, because they change fitted areas: the
Mn one shifts intensity between 0.637 and 0.649 keV, and the placeholder flag
governs whether Ti L is trusted enough to fit at all.
"""

PLACEHOLDER_Z = (20, 21, 22, 23)     # Ca, Sc, Ti, V - La:Lb:Ll:Leta = 100:10:1:1

# Lb/La interpolated across the neighbours that do have measured values.
LB_LA_FIX = {25: 0.203}              # Mn: was 0.301


def apply(db, fix_mn=True, verbose=True):
    """Rescale Lb to the corrected Lb/La, preserving the La+Lb total."""
    if not fix_mn:
        return
    for Z, target in LB_LA_FIX.items():
        I = db.lineI.get(Z)
        if not I or 'La_' not in I or 'Lb_' not in I:
            continue
        a, b = I['La_'], I['Lb_']
        tot = a + b
        na = tot / (1.0 + target)
        nb = tot - na
        if verbose:
            print('  lfix: %s Lb/La %.3f -> %.3f  (La %.5f->%.5f, Lb %.5f->%.5f)'
                  % (db.sym[Z], b / a, target, a, na, b, nb))
        I['La_'], I['Lb_'] = na, nb
        for nm, val in (('La1', na), ('Lb1', nb)):
            if nm in I:
                I[nm] = val
