"""Predicted detector artefacts: escape peaks and sum peaks.

Why predict them rather than only fit them
------------------------------------------
Both put real, sharp features into a spectrum at energies where no element
emits. A fitter offered a peak with no correct explanation will reach for the
nearest element, and the nearest element to an artefact is arbitrary - the
2.24 keV Si+O sum in a quartz spectrum was being handed to mercury.

Fitting them is the cure; knowing where they are BEFORE fitting is the
prevention. Marking them on the plot means an operator choosing elements can
see that a feature is already accounted for and decline to invent something.

Escape peaks
------------
A photon absorbed in the crystal can eject the crystal's own K X-ray, which
then leaves it. The event records low by exactly the escape energy:

    E_escape = E - 1.740 keV         (Si Ka_, from the database)
    E_escape = E - 9.885 keV         (Ge Ka1, from the database)

Only possible above the crystal's K edge - 1.839 keV on Si, so there are no
escape peaks from carbon, nitrogen, oxygen or fluorine at all, which is worth
knowing: an unexplained feature in the light-element region is NOT an escape
peak and something else is going on. On Ge the edge is ~11 keV (db).

As in GeoPIXE's escape_energy.pro, only Si and Ge crystals have escapes; any
other crystal gets none rather than a silicon guess. Ge also escapes its Kb1
(db, 10.98 keV; branching 0.133, escape_fraction.pro:29), which GeoPIXE's
display adds to the alpha escapes (spectrum_display_eventcb.pro:3769-3776);
it is marked here as '<parent> esc Kb'. Si has no beta escape.

Each element is one parent here, at its major line with the element's whole
area, so the escapes of its minor lines (Mo Kb's, say) are not marked;
GeoPIXE's display shifts the whole measured spectrum instead.

Sum peaks
---------
Two photons arriving inside the shaping time record as one:

    E_sum = E_i + E_j

Intensity scales as the product of the two rates, with a factor 2 for the
cross terms because either photon can arrive first. The sum peak is also
WIDER than a real line - the variances add - which is a useful check: a
feature about sqrt(2) times too wide for its energy is probably a sum.
"""

import numpy as np

# Fallbacks only: crystal_escape() reads both from the database, as
# EscapeModel does, so the markers sit on the fitted escape lines.
SI_KA = 1.740           # keV, db Si Ka_ (escape_energy.pro:21)
SI_K_EDGE = 1.8389      # keV, below which no escape is possible


def crystal_escape(db, crystal_Z):
    """(escape energy, K edge) in keV for the detector crystal.

    Follows escape_energy.pro: Ge escapes its Ka1, Si its Ka_ (:21); every
    other crystal returns (0, 0), meaning no escape peaks. The lookups are
    EscapeModel's, so markers and the fitted escape lines cannot disagree.
    """
    Z = int(crystal_Z)
    if Z == 14:
        return (db.lineE.get(14, {}).get('Ka_', SI_KA),
                db.edge.get((14, 'K'), SI_K_EDGE))
    if Z == 32:
        # same lookup and fallback as EscapeModel, so markers and the
        # fitted escape lines cannot disagree
        return (db.lineE.get(32, {}).get('Ka1', 9.886),
                db.edge.get((32, 'K'), 0.0))
    return 0.0, 0.0


def crystal_escape_beta(db, crystal_Z):
    """(K-beta escape energy keV, branching) for the crystal: Ge Kb1 and
    0.133 (escape_energy.pro:13, escape_fraction.pro:29), else (0, 0).
    Same lookup and fallback as EscapeModel.e_beta / ba_beta."""
    if int(crystal_Z) == 32:
        return db.lineE.get(32, {}).get('Kb1', 10.982), 0.133
    return 0.0, 0.0


class Artefact:
    __slots__ = ('kind', 'energy', 'label', 'parents', 'intensity', 'fwhm_eV')

    def __init__(self, kind, energy, label, parents, intensity, fwhm_eV=None):
        self.kind = kind                 # 'escape' or 'sum'
        self.energy = energy
        self.label = label
        self.parents = parents
        self.intensity = intensity       # predicted counts, if rates known
        self.fwhm_eV = fwhm_eV

    def __repr__(self):
        return '<%s %s %.4f keV  %.0f counts>' % (
            self.kind, self.label, self.energy, self.intensity or 0)


def _rough_fraction(E, k_edge):
    """escape_peaks' default escape fraction (see there)."""
    return 0.010 * (k_edge / E) ** 2.5


def escape_peaks(lines, e_low=0.2, e_high=6.6, escape_fraction=None,
                 min_counts=1.0, escape_energy=SI_KA, k_edge=SI_K_EDGE,
                 suffix='esc'):
    """Escape peaks for a set of (label, energy, counts).

    escape_energy, k_edge: the crystal's, from crystal_escape(); silicon by
    default. An escape energy of zero means a crystal with no escapes.
    suffix names the marker, '<label> <suffix>' ('esc Kb' for a beta escape).

    escape_fraction: callable(E) -> fraction, or None for a rough default.
    The default is a crude 1/E-ish falloff anchored near 1% just above the
    crystal edge (anchored for Si: on Ge its counts are far too low, the
    positions are still right); it is there so the marker positions are usable
    without a detector model, and it is NOT good enough to subtract with. Pass
    a real one from physics.escape when quantifying.
    """
    out = []
    if escape_energy < 0.01:           # escape_fraction.pro: no escape line
        return out
    for label, E, counts in lines:
        if E <= k_edge:
            continue
        Ee = E - escape_energy
        if not (e_low <= Ee <= e_high):
            continue
        if escape_fraction is not None:
            f = float(escape_fraction(E))
        else:
            f = _rough_fraction(E, k_edge)
        n = (counts or 0.0) * f
        if n < min_counts:
            continue
        out.append(Artefact('escape', Ee, '%s %s' % (label, suffix),
                            (label,), n))
    return sorted(out, key=lambda a: a.energy)


def sum_peaks(lines, e_low=0.2, e_high=6.6, pileup_fraction=None,
              total_counts=None, min_counts=1.0, resolution=None):
    """Sum peaks from every pair of the given lines.

    pileup_fraction: the probability that a second photon lands inside the
    shaping time, roughly (count rate) x (shaping time). If not given it is
    estimated from the observed data by the caller and passed in; there is no
    sensible universal default because it depends entirely on how hard the
    detector was being driven.
    """
    out = []
    n = len(lines)
    tot = total_counts or sum(c for _, _, c in lines) or 1.0
    p = pileup_fraction
    for i in range(n):
        li, Ei, Ci = lines[i]
        for j in range(i, n):
            lj, Ej, Cj = lines[j]
            Es = Ei + Ej
            if not (e_low <= Es <= e_high):
                continue
            # N_ij = p * C_i * C_j / total, doubled for i != j
            if p is None:
                counts = 0.0
            else:
                counts = p * Ci * Cj / tot * (1.0 if i == j else 2.0)
                if counts < min_counts:
                    continue
            w = None
            if resolution is not None:
                # variances add, so a sum peak is sqrt(2) wider than a line of
                # comparable energy - a useful tell when identifying one
                w = float(np.hypot(resolution(Ei), resolution(Ej)))
            lab = '%s+%s' % (li, lj) if i != j else '2x%s' % li
            out.append(Artefact('sum', Es, lab, (li, lj), counts, w))
    return sorted(out, key=lambda a: a.energy)


def estimate_pileup_fraction(spectrum, cal, lines, e_low=0.2, e_high=6.6):
    """Infer the pile-up probability from a sum peak that is actually there.

    Uses the strongest predicted sum peak whose energy lands in a region with
    no tabulated element line nearby, measures what is there, and solves for
    the fraction. Returns None when no clean sum peak exists, which is the
    honest answer rather than a guessed constant.
    """
    y = np.asarray(spectrum, dtype=float)
    a, b = cal
    E = a * np.arange(len(y)) + b
    tot = y.sum() or 1.0
    best = None
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            Es = lines[i][1] + lines[j][1]
            if not (e_low <= Es <= e_high):
                continue
            w = lines[i][2] * lines[j][2] * (1.0 if i == j else 2.0)
            if best is None or w > best[0]:
                best = (w, Es)
    if best is None:
        return None
    weight, Es = best
    q = (E > Es - 0.09) & (E < Es + 0.09)
    if q.sum() < 5:
        return None
    side = ((E > Es - 0.28) & (E < Es - 0.12)) | ((E > Es + 0.12) & (E < Es + 0.28))
    if side.sum() < 5:
        return None
    net = y[q].sum() - y[side].mean() * q.sum()
    if net <= 0 or not weight:
        return None
    # N_ij = p * C_i * C_j / total, so p = N_ij * total / (C_i * C_j).
    # 'weight' already carries the factor 2 for the i != j cross terms.
    return float(net * tot / weight)


def predict(db, spectrum, cal, elements, e_low=0.2, e_high=6.6,
            areas=None, resolution=None, escape_fraction=None, crystal_Z=14,
            escape_fraction_beta=None):
    """Everything at once, for the GUI.

    elements: [(Z, shell)]. areas: {name: counts} from a fit, or None to use
    the raw spectrum around each line. escape_fraction: callable(E) ->
    fraction, normally the fit's own EscapeModel.fraction, or None for the
    rough default in escape_peaks. crystal_Z: the detector crystal
    (options.crystal_Z), which sets the escape energy and edge.
    escape_fraction_beta: callable(E) for the Ge K-beta escape, normally
    lambda E: EscapeModel.fraction(E, beta=True); None takes the branching
    0.133 times the alpha fraction, which ignores the small change of the
    crystal's attenuation between Ka1 and Kb1.

    Returns (escapes, sums, pileup_fraction).
    """
    y = np.asarray(spectrum, dtype=float)
    a, b = cal
    E = a * np.arange(len(y)) + b

    lines = []
    for Z, sh in elements:
        e = db.line_energy(Z, sh)
        if not (e_low <= e <= e_high):
            continue
        nm = db.sym[Z] + {1: '', 2: 'L', 3: 'M'}[sh]
        if areas and nm in areas:
            c = max(areas[nm], 0.0)
        else:
            q = (E > e - 0.08) & (E < e + 0.08)
            c = float(y[q].sum())
        if c > 0:
            lines.append((nm, e, c))
    if not lines:
        return [], [], None

    p = estimate_pileup_fraction(y, cal, lines, e_low, e_high)
    e_esc, k_edge = crystal_escape(db, crystal_Z)
    esc = escape_peaks(lines, e_low, e_high, escape_fraction=escape_fraction,
                       escape_energy=e_esc, k_edge=k_edge)
    e_beta, ba = crystal_escape_beta(db, crystal_Z)
    if e_beta > 0:
        fb = escape_fraction_beta
        if fb is None:
            fa = escape_fraction or (lambda E: _rough_fraction(E, k_edge))
            fb = lambda E: ba * fa(E)
        esc = sorted(esc + escape_peaks(
            lines, e_low, e_high, escape_fraction=fb, escape_energy=e_beta,
            k_edge=k_edge, suffix='esc Kb'), key=lambda a: a.energy)
    sums = sum_peaks(lines, e_low, e_high, pileup_fraction=p,
                     total_counts=y.sum(), resolution=resolution)
    return esc, sums, p
