"""
Thick-target PIXE yield calculation, ported from GeoPIXE's model.

Reproduces the chain in geo_yield2 -> calc_slices2 -> calc_abs -> calc_yield:
the beam slows through the target, K-shell ionisation is integrated over
depth, and the emitted X-rays are attenuated back out to the detector.

    Y_i  =  (N_A / A_i) * INTEGRAL_0^R  sigma_K(E(x)) * omega_K * b
                                        * exp(-mu_i * x / cos(theta)) dx

with x in mg/cm^2 and dE/dx from the Andersen-Ziegler parameterization.

Secondary fluorescence is NOT included - it is a few percent for the cases
here and its omission is stated rather than hidden. Detector efficiency and
solid angle are also excluded, so absolute yields carry an arbitrary constant;
RATIOS between elements are the meaningful output and are what composition
depends on.
"""

import math


class YieldModel:
    def __init__(self, db):
        self.db = db

    def slow_down(self, Z1, A1, E0, zlist, wfrac, n_steps=400, e_stop=0.02):
        """Return (x, E) with x the areal density in mg/cm^2 and E the beam
        energy there, integrating dE/dx until the beam stops."""
        db = self.db
        xs, es = [0.0], [E0]
        E = E0
        x = 0.0
        # step in energy, which keeps resolution near the end of range
        dE = (E0 - e_stop) / n_steps
        for _ in range(n_steps):
            s = db.dedx_compound(Z1, A1, zlist, wfrac, E)
            if s <= 0:
                break
            dx = dE / s
            x += dx
            E -= dE
            if E <= e_stop:
                break
            xs.append(x)
            es.append(E)
        return xs, es

    @staticmethod
    def _interp_E(xs, es, x):
        """Beam energy at areal density x, linear in x."""
        if x <= xs[0]:
            return es[0]
        if x >= xs[-1]:
            return 0.0
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        f = (x - xs[lo]) / (xs[hi] - xs[lo])
        return es[lo] + f * (es[hi] - es[lo])

    def yields(self, zlist, wfrac, E0=1.0, Z1=1, A1=1.0, theta_deg=135.0,
               mac='henke1993', fy='krause', elam_zmax=10,
               elements=None, n_steps=400, per_major_line=True):
        """Thick-target yields for the requested elements.

        zlist/wfrac define the matrix (mass fractions, normalised here).
        'elements' defaults to the matrix elements.
        Returns {Z: yield} in arbitrary but consistent units.
        """
        db = self.db
        s = sum(wfrac)
        wfrac = [w / s for w in wfrac]
        if elements is None:
            elements = list(zlist)

        xs, es = self.slow_down(Z1, A1, E0, zlist, wfrac, n_steps=n_steps)
        cos_t = math.cos(math.radians(180.0 - theta_deg)) \
            if theta_deg > 90 else math.cos(math.radians(theta_deg))
        cos_t = abs(cos_t)
        if cos_t < 1e-3:
            cos_t = 1e-3

        out = {}
        detail = {}
        for Z in elements:
            eline = db.line_energy(Z)
            if eline <= 0.0:
                out[Z] = 0.0
                detail[Z] = dict(reason='no line energy')
                continue
            mu = db.mu_compound(zlist, wfrac, eline, mac)
            if mu is not None:
                # UNITS: mu_compound returns cm^2/g but the depth x below is in
                # mg/cm^2 (GeoPIXE's convention throughout - note that its own
                # 'atten' returns cm^2/mg for exactly this reason). Without the
                # 1e-3 the exponent is 1000x too large, absorption swamps the
                # integral, and the yield collapses to sigma(E0)*attenuation
                # length - which is nearly flat in Z because the attenuation
                # length rises almost as fast as the cross section falls.
                mu = mu * 1.0e-3          # cm^2/g -> cm^2/mg
            if mu is None:
                out[Z] = float('nan')
                detail[Z] = dict(reason='MAC out of range at %.4f keV' % eline)
                continue
            om = db.fluor_yield(Z, 'K', fy, elam_zmax)
            # Integrate on a depth grid refined near the surface.
            #
            # The absorption length for a soft line can be far shorter than
            # the step you get from uniform energy stepping: carbon Ka in NaCl
            # attenuates over ~0.02 mg/cm^2, while a 400-step energy walk gives
            # a first step of ~0.014 mg/cm^2. Integrating on that grid badly
            # over-predicts the soft lines. A log-spaced depth grid, refined to
            # a small fraction of the attenuation length, removes it.
            xmax = xs[-1]
            atten_len = cos_t / mu if mu > 0 else xmax
            x0 = min(atten_len / 200.0, xmax / 1.0e5)
            x0 = max(x0, 1.0e-9)
            ngrid = 4000
            tot = 0.0
            prevx = 0.0
            prevf = db.sigma_K(Z, es[0], A1)      # x = 0, no attenuation
            for k in range(ngrid):
                x = x0 * (xmax / x0) ** (k / float(ngrid - 1))
                E = self._interp_E(xs, es, x)
                if E <= 0.0:
                    f = 0.0
                else:
                    f = db.sigma_K(Z, E, A1) * math.exp(-mu * x / cos_t)
                tot += 0.5 * (f + prevf) * (x - prevx)
                prevx, prevf = x, f
            y = tot * om * db.N_A_over_A(Z)

            # GeoPIXE reports the yield PER MAJOR LINE, not per element. Below
            # Z = 27 the major line is the unresolved Ka group (beta ~ 0.88);
            # above it, the group splits and the major line becomes Ka1 alone
            # (beta ~ 0.58). Comparing an element-total yield against GeoPIXE's
            # therefore shows a spurious 1.5x step at Z = 28. Same convention
            # applies to the 'area' field in .pfr fit-result files.
            if per_major_line:
                lines = db.line_list(Z, 1)
                if lines:
                    y = y * max(i for _, i in lines)

            out[Z] = y
            detail[Z] = dict(e_line=eline, mu=mu, omega=om,
                             range_mgcm2=xs[-1] if xs else 0.0)
        self.last_detail = detail
        return out
