"""
Secondary fluorescence, ported from secondary_fluorescence.pro.

A characteristic X-ray produced at depth x' can itself ionise another element
at depth x, provided its energy exceeds that element's absorption edge. The
second element's measured yield is then enhanced above what primary ion
ionisation alone would give.

GeoPIXE follows Reuter et al. (1975), integrating over depth with the kernel

    delta(a, a+b) = exp(-a) - exp(-(a+b)) - a*E1(a) + (a+b)*E1(a+b)

where a is the optical depth (in mux) between source and destination and b is
the half-slice increment. E1 is the exponential integral, GeoPIXE's
Elliptical_1. The enhancement for destination element j from source line i is

    0.5 * dY_i * omega_j * delta * f_j(E_i) * (r_j - 1)/r_j * branch * escape

with f_j(E_i) = mu_j(E_i)*w_j / mu_matrix(E_i), the fraction of the source
line's absorption that happens on element j. The factor 0.5 is the isotropic
half-space.

Why it matters here rather than being an optional refinement: in NaCl, Cl Ka
at 2.62 keV sits above the Na K edge at 1.07 keV, so sodium is enhanced and
chlorine is not. Leaving it out biases the Na/Cl ratio - the exact quantity a
thick NaCl target is used to check.
"""

import numpy as np
from scipy.special import exp1 as _exp1


def E1(x):
    """Exponential integral E1, matching GeoPIXE's Elliptical_1 for x > 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    q = x > 0
    out[q] = _exp1(x[q])
    return out


def enhancement(db, zlist, wfrac, xs, es, dest_Z, dest_line_E, mac,
                cos_out, sigma_of, omega_of, source_elements=None,
                n_depth=160):
    """Secondary-fluorescence enhancement factor for one destination element.

    Returns the ADDITIVE yield contribution in the same arbitrary units as the
    primary yield, so the caller adds it to the primary.

    db          Database
    zlist,wfrac matrix composition, mass fractions
    xs, es      depth (mg/cm^2) and beam energy (MeV) profile
    dest_Z      destination element
    dest_line_E its major line energy, keV
    sigma_of    callable (Z, E_MeV) -> ionisation cross section
    omega_of    callable (Z) -> fluorescence yield
    """
    edge_j = db.edge.get((dest_Z, 'K'), 0.0)
    if edge_j <= 0:
        return 0.0

    mu_j_out = db.mu_compound(zlist, wfrac, dest_line_E, mac)
    if mu_j_out is None:
        return 0.0
    mu_j_out *= 1.0e-3                      # cm^2/g -> cm^2/mg

    omega_j = omega_of(dest_Z)
    jr = db.jump.get((dest_Z, 'K'), 1.0)
    if jr <= 1.0:
        return 0.0
    jump_factor = (jr - 1.0) / jr

    xmax = xs[-1]
    xg = np.linspace(0.0, xmax, n_depth)
    Eg = np.interp(xg, xs, es)

    if source_elements is None:
        source_elements = list(zlist)

    total = 0.0
    for Zs, ws in zip(zlist, wfrac):
        if Zs not in source_elements or ws <= 0:
            continue
        Es = db.line_energy(Zs)
        if Es <= edge_j:                     # cannot ionise the destination
            continue
        mu_src = db.mu_compound(zlist, wfrac, Es, mac)
        if mu_src is None:
            continue
        mu_src *= 1.0e-3

        # fraction of the source line's absorption that lands on element j
        mu_j_at_src = db.mu(dest_Z, Es, mac)
        if mu_j_at_src is None:
            continue
        f_j = (mu_j_at_src * 1.0e-3) * ws_dest(db, zlist, wfrac, dest_Z) / mu_src
        if f_j <= 0:
            continue

        # primary production of the source line per unit depth
        omega_s = omega_of(Zs)
        prod = np.array([sigma_of(Zs, E) for E in Eg]) * omega_s * ws
        if prod.max() <= 0:
            continue

        # depth-transport kernel, evaluated pairwise
        db_step = np.gradient(xg)
        contrib = 0.0
        for k in range(n_depth):
            a = mu_src * np.abs(xg - xg[k])
            bstep = 0.5 * mu_src * db_step
            a_lo = np.maximum(a - bstep, 0.0)
            a_hi = a + bstep
            delta = (np.exp(-a_lo) - np.exp(-a_hi)
                     - a_lo * E1(a_lo) + a_hi * E1(a_hi))
            src = 0.5 * np.sum(prod * delta) * db_step[k]
            contrib += src * np.exp(-mu_j_out * xg[k] / cos_out)
        total += contrib * omega_j * f_j * jump_factor

    return total * db.N_A_over_A(dest_Z)


def ws_dest(db, zlist, wfrac, Z):
    """Mass fraction of the destination element in the matrix. Trace elements
    not in the matrix formula are enhanced per unit concentration, so return 1
    and let the caller's per-concentration convention carry it."""
    for z, w in zip(zlist, wfrac):
        if z == Z:
            return 1.0
    return 1.0
