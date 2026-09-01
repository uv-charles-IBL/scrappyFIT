"""
Layered thick-target yields, and the surface-sensitivity diagnostic.

Why this exists
---------------
For a soft line the X-ray escape depth can be far shorter than the proton
range. Carbon Ka in NaCl attenuates over ~0.02 mg/cm^2 against a 3.8 mg/cm^2
range at 1 MeV - so 99.5% of the ionisation the beam produces never reaches
the detector. The measurement is then almost entirely a surface measurement.

That is NOT the same as saying the element IS a surface film, and the
distinction matters:

  * element genuinely in the bulk   -> bulk model is right, you just sample
                                       the top of it
  * element is a surface film       -> bulk model inflates the concentration
                                       by roughly range/escape_depth

A single spectrum cannot distinguish them. The concentration you report is
model-dependent, and by a large factor. What the code CAN do is refuse to
report a bulk number silently when the ratio is extreme - hence
`surface_sensitivity` below, which every caller should check.

Two ways to break the degeneracy experimentally, both worth doing:
  * vary the beam energy - changes the range, not the escape depth
  * vary the take-off angle - changes the escape path, not the range
If the derived bulk concentration moves with either, the element is not
uniformly distributed.
"""

import math
import numpy as np


class Layer:
    def __init__(self, zlist, wfrac, thick_mgcm2, name=''):
        s = sum(wfrac)
        self.z = list(zlist)
        self.w = [x / s for x in wfrac]
        self.thick = float(thick_mgcm2)
        self.name = name


def surface_sensitivity(db, layer_z, layer_w, E_line, mac, range_mgcm2,
                        cos_out):
    """escape_depth / proton_range for a line. Small means surface-dominated.

    Returns (ratio, escape_depth_mgcm2). A ratio below ~0.05 means a bulk
    concentration is being inferred from the top few percent of the excited
    volume and is highly sensitive to how the element is actually distributed.
    """
    mu = db.mu_compound(layer_z, layer_w, E_line, mac)
    if mu is None or mu <= 0:
        return float('nan'), float('nan')
    mu = mu * 1.0e-3                       # cm^2/g -> cm^2/mg
    escape = cos_out / mu
    return escape / range_mgcm2, escape


class LayeredYieldModel:
    """Yields for a stack of layers, surface first.

    The beam slows continuously through the stack; each layer's own
    composition sets its stopping power. X-rays produced at depth x are
    attenuated by every layer between x and the surface, each with its own
    mass attenuation coefficient - which is the part a single-layer model
    cannot represent when the surface layer is chemically different.
    """

    def __init__(self, db):
        self.db = db

    def _profile(self, layers, Z1, A1, E0, n_steps, e_stop=0.02):
        """(x, E, layer_index) through the stack."""
        db = self.db
        xs, es, li = [0.0], [E0], [0]
        E, x = E0, 0.0
        edges = np.cumsum([l.thick for l in layers])
        dE = (E0 - e_stop) / n_steps
        for _ in range(n_steps):
            k = int(np.searchsorted(edges, x, side='right'))
            if k >= len(layers):
                break
            L = layers[k]
            s = db.dedx_compound(Z1, A1, L.z, L.w, E)
            if s <= 0:
                break
            x += dE / s
            E -= dE
            if E <= e_stop:
                break
            xs.append(x); es.append(E); li.append(min(k, len(layers) - 1))
        return np.array(xs), np.array(es), np.array(li)

    def _mux(self, layers, E_line, mac, xs, li, cos_out):
        """Cumulative optical depth from the surface to each x, using each
        layer's own MAC."""
        db = self.db
        mus = []
        for L in layers:
            m = db.mu_compound(L.z, L.w, E_line, mac)
            mus.append(None if m is None else m * 1.0e-3)
        if any(m is None for m in mus):
            return None
        mux = np.zeros(len(xs))
        for i in range(1, len(xs)):
            k = min(int(li[i]), len(layers) - 1)
            mux[i] = mux[i - 1] + mus[k] * (xs[i] - xs[i - 1])
        return mux / cos_out

    def yields(self, layers, elements, E0=1.0, Z1=1, A1=1.0, theta_deg=135.0,
               mac='henke1993', fy='krause', elam_zmax=10, n_steps=1500,
               per_major_line=True, in_layer=None):
        """Yields per major line, per unit mass fraction IN ITS OWN LAYER.

        in_layer: {Z: layer_index} - which layer each element is taken to sit
        in. Defaults to every element in every layer, weighted by that layer's
        own composition, which is what a bulk model does.
        """
        db = self.db
        cos_out = abs(math.cos(math.radians(180.0 - theta_deg
                                            if theta_deg > 90 else theta_deg)))
        cos_out = max(cos_out, 1e-3)

        xs, es, li = self._profile(layers, Z1, A1, E0, n_steps)
        out, detail = {}, {}

        for Z in elements:
            eline = db.line_energy(Z)
            if eline <= 0:
                out[Z] = 0.0
                detail[Z] = dict(reason='no line energy')
                continue
            mux = self._mux(layers, eline, mac, xs, li, cos_out)
            if mux is None:
                out[Z] = float('nan')
                detail[Z] = dict(reason='MAC out of range at %.4f keV' % eline)
                continue

            klayer = None if in_layer is None else in_layer.get(Z)
            om = db.fluor_yield(Z, 'K', fy, elam_zmax)

            tot = 0.0
            for i in range(1, len(xs)):
                if klayer is not None and int(li[i]) != klayer:
                    continue
                E = 0.5 * (es[i] + es[i - 1])
                sig = db.sigma_K(Z, E, A1)
                if sig <= 0:
                    continue
                atten = math.exp(-0.5 * (mux[i] + mux[i - 1]))
                tot += sig * atten * (xs[i] - xs[i - 1])

            y = tot * om * db.N_A_over_A(Z)
            if per_major_line:
                L = db.line_list(Z, 1)
                if L:
                    y *= max(i for _, i in L)

            host = layers[klayer] if klayer is not None else layers[0]
            ratio, escape = surface_sensitivity(db, host.z, host.w, eline,
                                                mac, xs[-1], cos_out)
            out[Z] = y
            detail[Z] = dict(e_line=eline, omega=om, range_mgcm2=xs[-1],
                             escape_mgcm2=escape, surface_ratio=ratio,
                             layer=klayer)
        self.last_detail = detail
        return out
