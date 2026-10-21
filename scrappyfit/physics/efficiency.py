"""Detector efficiency curves.

What this is for
----------------
A fitted peak area is not a concentration. Between the two sit the ionisation
cross section, the fluorescence yield, self-absorption in the sample, and the
probability that the photon actually reached the detector and was counted.
That last term is the efficiency curve, and at low energy it is brutal: on an
Amptek C1 window the transmission at carbon Ka is a few percent, so a carbon
peak of a thousand counts represents far more carbon than a thousand counts of
silicon does.

Getting it wrong scales concentrations directly. It is also the term that is
hardest to know, because it depends on the exact window and contact stack of
the individual detector.

Reading versus computing
------------------------
Two modes:

  from_geopixe()  reads a curve GeoPIXE has already computed and exported.
                  This is the trustworthy route today, because GeoPIXE's
                  curve was built from the real detector definition.

  from_layers()   computes a curve from an explicit absorber stack plus the
                  crystal. Useful for asking what-if questions - what would
                  this look like on a C2 window - and as a check on the
                  exported curve, but it only knows what you tell it. The
                  detector's own dead layer and contact structure are not
                  published per unit and are not modelled here.

The exported file carries TWO curves. GeoPIXE calls the second one
'skip_abs': it is the efficiency with the sample self-absorption term left
out. Which one you want depends on the calculation - the yield model here
applies its own absorption, so it wants the plain detector term. Passing the
wrong one is a silent error, which is why both are kept and named.
"""

import numpy as np


class Efficiency:
    """Efficiency as a function of photon energy in keV.

    Callable: eff(1.74) returns a number. Outside the tabulated range the
    endpoint value is held rather than extrapolated, because extrapolating an
    exponential absorption edge produces nonsense quietly.
    """

    def __init__(self, energy, values, skip_abs=None, name='', source=''):
        idx = np.argsort(np.asarray(energy, dtype=float))
        self.energy = np.asarray(energy, dtype=float)[idx]
        self.values = np.asarray(values, dtype=float)[idx]
        self.skip_abs = (None if skip_abs is None
                         else np.asarray(skip_abs, dtype=float)[idx])
        self.name = name or 'efficiency'
        self.source = source

    def __call__(self, E, skip_abs=False):
        y = self.skip_abs if (skip_abs and self.skip_abs is not None) else self.values
        e = np.clip(np.asarray(E, dtype=float), self.energy[0], self.energy[-1])
        out = np.interp(e, self.energy, y)
        return float(out) if np.isscalar(E) or np.ndim(E) == 0 else out

    @property
    def range(self):
        return float(self.energy[0]), float(self.energy[-1])

    def __repr__(self):
        return ('<Efficiency %s: %d points, %.3f-%.1f keV%s>'
                % (self.name, len(self.energy), self.energy[0],
                   self.energy[-1], '' if self.skip_abs is None
                   else ', with skip_abs'))


def from_geopixe(path, name=''):
    """Read a curve exported by GeoPIXE.

    Two layouts are in circulation and both appear in the wild:

        EFF3   <keV>  <efficiency>  <efficiency_skip_abs>
        <keV>  <efficiency>  <efficiency_skip_abs>

    Anything else on a line is ignored, so headers like DETOK do not need
    stripping first.
    """
    E, a, b = [], [], []
    for line in open(path, errors='ignore'):
        t = line.split()
        if not t:
            continue
        if t[0].upper() == 'EFF3':
            t = t[1:]
        if len(t) < 2:
            continue
        try:
            vals = [float(x) for x in t[:3]]
        except ValueError:
            continue
        E.append(vals[0])
        a.append(vals[1])
        b.append(vals[2] if len(vals) > 2 else np.nan)
    if not E:
        raise ValueError('no efficiency rows found in %s' % path)
    b = None if np.all(np.isnan(b)) else b
    import os
    return Efficiency(E, a, b, name=name or os.path.basename(path),
                      source=str(path))


def from_layers(db, absorbers, crystal=('Si', 500.0), energies=None,
                mac='mixed', name='computed'):
    """Compute a curve from an explicit stack.

    absorbers : sequence of (formula_or_symbol, thickness_um, density_g_cm3)
                traversed before the crystal - windows, contacts, air paths.
    crystal   : (symbol, thickness_um) - what actually absorbs and counts.

    efficiency(E) = product(exp(-mu_i rho_i t_i)) * (1 - exp(-mu_c rho_c t_c))

    The first factor is what gets through the windows; the second is what
    then stops in the crystal. Below about 1 keV the first factor collapses
    and dominates everything.

    This deliberately does NOT model the detector's dead layer or incomplete
    charge collection - those are not window absorption, they are response,
    and they live in scrappyfit.fitting.peakshape. Mixing them here would
    double-count.
    """
    if energies is None:
        energies = np.geomspace(0.1, 20.0, 400)
    energies = np.asarray(energies, dtype=float)

    def mu_of(spec, E):
        if isinstance(spec, str):
            Z = db.z.get(spec.lower())
            if Z is None:
                raise ValueError('unknown absorber %r' % spec)
            return db.mu(Z, E, mac)
        zl, wf = spec
        return db.mu_compound(zl, wf, E, mac)

    trans = np.ones_like(energies)
    for spec, t_um, rho in absorbers:
        areal = t_um * 1e-4 * rho                     # g/cm2
        m = np.array([(mu_of(spec, e) or 0.0) for e in energies])
        trans *= np.exp(-m * areal)

    csym, ct_um = crystal
    crho = db.density.get(db.z[csym.lower()], 2.33) if hasattr(db, 'density') else 2.33
    careal = ct_um * 1e-4 * crho
    mc = np.array([(mu_of(csym, e) or 0.0) for e in energies])
    stop = 1.0 - np.exp(-mc * careal)

    return Efficiency(energies, trans * stop, None, name=name,
                      source='computed from %d absorbers + %s %.0f um'
                             % (len(absorbers), csym, ct_um))


# The Amptek C series windows, from the X-123 manual. Given here so a computed
# curve can be compared against an exported one, and so the C1/C2 difference
# can be asked about directly.
AMPTEK_C1 = [('Si3N4', 0.090, 3.17), ('Al', 0.250, 2.70)]
AMPTEK_C2 = [('Si3N4', 0.040, 3.17), ('Al', 0.030, 2.70)]
