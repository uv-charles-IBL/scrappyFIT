"""Line data from xraylib, used to correct the Elam rebuild where they differ.

Cross-check outcome (see the session notes):

  fluorescence yields   AGREE. Identical Ca-Ag, within 1-9% for Sn/Pb. Elam is
                        fine here and needs no change.

  Coster-Kronig f23     DISAGREE, and Elam is wrong. Elam gives f23 = 0.25
                        (Ti), 0.42 (Fe), 0.45 (Ni); xraylib gives 0.000. The
                        L2-L3 Coster-Kronig transition is energetically
                        FORBIDDEN below about Z=30 - the L2-L3 splitting is
                        smaller than the M-shell binding energy, so there is
                        no channel. xraylib is right.

                        This matters directly: f23 sets the floor in the
                        Coster-Kronig screening test, so Elam's value made the
                        floor far too high and caused real elements to be
                        rejected.

  L3 branching          DISAGREE for low Z, and Elam is the weaker source.
                        Elam's Ll fraction is a constant 0.099 from Ti to Fe -
                        interpolated, not per-element. xraylib varies smoothly
                        and physically: Ll rises as Z falls (0.096 Fe, 0.118
                        Mn, 0.137 Cr, 0.201 V, 0.291 Ti) because Lalpha
                        weakens as the 3d shell empties. Ll is 2-3x larger
                        than Elam says for Ti-Cr, and those lines sit in the
                        O-F region where it matters most.

xraylib bundles Campbell & Wang's Dirac-Fock L emission rates, which is why it
is the better source for L specifically.
"""
import xraylib as xl

SHELL = {'K': xl.K_SHELL, 'L1': xl.L1_SHELL, 'L2': xl.L2_SHELL,
         'L3': xl.L3_SHELL, 'M1': xl.M1_SHELL, 'M2': xl.M2_SHELL,
         'M3': xl.M3_SHELL, 'M4': xl.M4_SHELL, 'M5': xl.M5_SHELL}

# GeoPIXE mnemonic -> xraylib line macro, grouped by originating subshell
LINES = {
    'L1': [('Lb3', xl.L1M3_LINE), ('Lb4', xl.L1M2_LINE),
           ('Lg2', xl.L1N2_LINE), ('Lg3', xl.L1N3_LINE)],
    'L2': [('Leta', xl.L2M1_LINE), ('Lb1', xl.L2M4_LINE),
           ('Lg1', xl.L2N4_LINE)],
    'L3': [('Ll', xl.L3M1_LINE), ('La2', xl.L3M4_LINE),
           ('La1', xl.L3M5_LINE), ('Lb2', xl.L3N5_LINE),
           ('Lb5', xl.L3O4_LINE), ('Lb6', xl.L3N1_LINE)],
    'M3': [('Mg_', xl.M3N5_LINE)],
    'M4': [('Mb_', xl.M4N6_LINE), ('Mz_', xl.M4N2_LINE)],
    'M5': [('Ma1', xl.M5N7_LINE), ('Ma2', xl.M5N6_LINE)],
}
CK = {('L1', 'L2'): xl.FL12_TRANS, ('L1', 'L3'): xl.FL13_TRANS,
      ('L2', 'L3'): xl.FL23_TRANS}


def _safe(fn, *a):
    try:
        v = fn(*a)
        return v if v and v > 0 else 0.0
    except Exception:
        return 0.0


def lines(Z, shell):
    """[(mnemonic, energy_keV, branching_within_shell)], normalised to 1."""
    out = []
    for nm, mac in LINES.get(shell, []):
        r = _safe(xl.RadRate, Z, mac)
        e = _safe(xl.LineEnergy, Z, mac)
        if r > 0 and e > 0:
            out.append([nm, e, r])
    t = sum(o[2] for o in out)
    if t > 0:
        for o in out:
            o[2] /= t
    return [tuple(o) for o in out]


def omega(Z, shell):
    return _safe(xl.FluorYield, Z, SHELL[shell])


def ck(Z, frm, to):
    m = CK.get((frm, to))
    return _safe(xl.CosKronTransProb, Z, m) if m else 0.0


def edge(Z, shell):
    return _safe(xl.EdgeEnergy, Z, SHELL[shell])
