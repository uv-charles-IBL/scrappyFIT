"""Generate MAC_FFAST.txt from the Chantler (FFAST) tables bundled in xraydb.

Why FFAST is worth adding
-------------------------
Heirwegh (2014, ch. 3-5) tested XCOM against FFAST on silicate glass and
mineral standards and found that neither wins outright. The right answer is a
rule:

    FFAST below the K edges, and everywhere below 1 keV
    XCOM above the K edges

That mixed selection took the pure-element to silicate-oxide efficiency
discrepancy from 6-8% down to under 2%, and under 1% for Mg and Si.

Two facts make FFAST unavoidable for light-element work:
  * below 1 keV, XCOM and Scofield supply nothing at all, so FFAST is the only
    theoretical option there;
  * below the K edges of Mg, Al and Si, Chantler's DHS photo-absorption cross
    sections are strongly favoured over Scofield's by experiment (ch. 4).

Photoabsorption only, to match the other tabulations in this directory - the
absorption correction in a thick-target yield uses the photoelectric term.

Edges are encoded the same way as the Henke table: two rows sharing an
abscissa bracket a discontinuity, and 'absco_table' never interpolates across
such a pair.
"""
import sys
import numpy as np
import xraydb

OUT = (r'C:\Users\Charles\Desktop\GeoPIXE-main\Workspace\main\database\dat'
       r'\MAC_FFAST.txt')
E_LO, E_HI = 0.010, 30.0          # keV, matched to the Henke table's span
EDGE_SPLIT = 1.0e-6               # keV, how far to straddle an edge


def element_grid(Z):
    """Chantler's own abscissae inside our range, plus straddled edges."""
    e = np.asarray(xraydb.chantler_energies(Z), dtype=float) / 1000.0   # keV
    e = e[(e >= E_LO) & (e <= E_HI)]
    if len(e) < 4:
        return None
    # duplicate abscissae mark edges in the source grid; keep them so the
    # discontinuity survives, but separate them by a hair so a reader that
    # sorts strictly still sees two distinct rows bracketing the jump
    out = []
    for i, v in enumerate(e):
        if i and abs(v - e[i - 1]) < 1e-12:
            out.append(v + EDGE_SPLIT)
        else:
            out.append(v)
    return np.array(out)


def main():
    rows = []
    n_el = 0
    for Z in range(1, 93):
        try:
            sym = xraydb.atomic_symbol(Z)
            A = xraydb.atomic_mass(Z)
            g = element_grid(Z)
            if g is None:
                continue
            mu = np.array([float(xraydb.mu_chantler(Z, x * 1000.0, photo=True))
                           for x in g])
        except Exception as ex:
            sys.stderr.write('Z=%d skipped: %s\n' % (Z, ex))
            continue
        ok = np.isfinite(mu) & (mu > 0)
        g, mu = g[ok], mu[ok]
        if len(g) < 4:
            continue
        n_el += 1
        rows.append((Z, sym, A, g, mu))

    with open(OUT, 'w') as fh:
        fh.write('; GeoPIXE tabulated mass attenuation coefficients.\n')
        fh.write('; Chantler FFAST photo-absorption, via the xraydb package.\n')
        fh.write('; Source: NIST FFAST, C.T. Chantler et al.\n')
        fh.write('; Photoabsorption only; excludes coherent/incoherent scattering.\n')
        fh.write('; Unlike XCOM and Scofield this set extends below 1 keV, which\n')
        fh.write('; is the whole light-element range. Requests outside the\n')
        fh.write('; per-element Range must be refused, not extrapolated.\n')
        fh.write(';\n')
        fh.write('Dataset    FFAST\n')
        fh.write('Reference  C.T. Chantler, J. Phys. Chem. Ref. Data 24 (1995) 71; '
                 '29 (2000) 597; NIST FFAST\n')
        fh.write('Quantity   photoelectric mass attenuation coefficient\n')
        fh.write('Units      keV cm2/g\n')
        fh.write('Elements   %d\n' % n_el)
        fh.write(';\n')
        fh.write('; Edge discontinuities are encoded as repeated abscissae:\n')
        fh.write('; two rows with near-identical energy bracket a jump. Never\n')
        fh.write('; interpolate across such a pair.\n')
        fh.write(';\n')
        for Z, sym, A, g, mu in rows:
            fh.write('Element    %-5s %-5d %.6f\n' % (sym, Z, A))
            fh.write('Range      %.6E %.6E\n' % (g[0], g[-1]))
            fh.write('Npoints    %d\n' % len(g))
            fh.write('Data\n')
            for x, y in zip(g, mu):
                fh.write('  %.6E  %.6E\n' % (x, y))
    print('wrote %s' % OUT)
    print('elements: %d   total points: %d' % (n_el, sum(len(r[3]) for r in rows)))


if __name__ == '__main__':
    main()
