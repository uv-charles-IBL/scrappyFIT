"""The detector's low-energy tail, as tail_amplitude.pro computes it.

This package previously used a flat tail: one amplitude and one length for
every line, whatever its energy. GeoPIXE does not, and the difference is not
cosmetic.

tail_amplitude.pro derives the amplitude from where in the crystal the photon
was absorbed. Charge collection is incomplete in a thin FRONT region and in a
BACK region near the rear contact; an event landing in either loses some
charge and is recorded below its true energy. So the tail fraction is simply
the share of absorption happening in those two regions:

    U     = mu of the crystal at E, cm^2/mg
    D     = crystal thickness, mg/cm^2
    F, B  = front and back region thicknesses
    total = 1 - exp(-U D)
    front = 1 - exp(-U F)
    back  = exp(-U (D - B)) (1 - exp(-U B))
    amp   = tail.amp x (front + back) / total

That is strongly energy dependent. A soft X-ray is absorbed within microns of
the entrance, so nearly every event lands in the front region and the tail is
at its maximum. A hard X-ray penetrates deep into good material and the tail
falls away.

tail_length.pro is simpler and linear:

    length = max(L + S (E - 6.4), 0)

in units of the peak width, hinged at Fe Ka.

What the flat tail was doing to a fit, measured on GeoPIXE's own donut2x
example: the model came out 35% ABOVE the data at the Fe Ka peak (848k
against 630k counts, -274 sigma) while sitting +20 sigma BELOW it 0.4-1.0 keV
down the tail. With too little tail out in the wings the fit inflates the
total area to compensate, and the surplus lands on the peak. Fe's area came
out 1.29x GeoPIXE's for that reason alone.
"""

import numpy as np

#: Where tail_length.pro hinges its linear term.
TAIL_HINGE_KEV = 6.4


def _crystal_mu(db, det, E, mac='mixed'):
    """Crystal mu in cm^2/mg - atten() in GeoPIXE's units, not cm^2/g."""
    cry = det['crystal']
    v = db.mu_compound(cry['Z'], cry['F'], float(E), mac)
    return 0.0 if v is None else float(v) * 1e-3


def tail_amplitude(det, db, E, mac='mixed'):
    """Tail amplitude at energy E, following tail_amplitude.pro."""
    tail = det.get('tail') or {}
    amp0 = float(tail.get('amp', 0.0))
    if amp0 <= 1e-10:
        return 0.0
    D = float(det['crystal']['thick'])
    dens = float(det.get('density') or 0.0)
    if D <= 0 or dens <= 0:
        return amp0
    # F and B are recorded in microns; density/10 converts to mg/cm^2
    F = min(float(tail.get('F', 0.0)) * dens / 10.0, D)
    B = min(float(tail.get('B', 0.0)) * dens / 10.0, max(D - F, 0.0))

    U = _crystal_mu(db, det, E, mac) if E > 1.0 else 10000.0
    if U <= 0:
        return 0.0
    total = 1.0 - np.exp(-U * D)
    if total <= 0:
        return 0.0
    front = 1.0 - np.exp(-U * F)
    back = np.exp(-U * (D - B)) * (1.0 - np.exp(-U * B))
    frac = (front + back) / total
    if not np.isfinite(frac):
        return 0.0
    return amp0 * min(frac, 1.0)


def tail_length(det, E):
    """Tail length at energy E, following tail_length.pro."""
    tail = det.get('tail') or {}
    return max(float(tail.get('L', 0.0))
               + float(tail.get('S', 0.0)) * (float(E) - TAIL_HINGE_KEV), 0.0)


def tail_functions(det, db, mac='mixed', cache=True):
    """(amp_fn, len_fn) ready to attach to a Component.

    Cached per energy: the amplitude needs a MAC lookup and the same line
    energies recur on every iteration of the fit.
    """
    memo = {}

    def amp_fn(E):
        if not cache:
            return tail_amplitude(det, db, E, mac)
        key = round(float(E), 5)
        if key not in memo:
            memo[key] = tail_amplitude(det, db, E, mac)
        return memo[key]

    def len_fn(E):
        return tail_length(det, E)

    return amp_fn, len_fn


def describe(det, db, energies=(1.0, 2.0, 5.0, 6.4, 10.0, 20.0, 40.0),
             mac='mixed'):
    rows = ['%-8s %12s %12s' % ('keV', 'tail amp', 'tail length')]
    for e in energies:
        rows.append('%-8.2f %12.5f %12.5f'
                    % (e, tail_amplitude(det, db, e, mac), tail_length(det, e)))
    return '\n'.join(rows)
