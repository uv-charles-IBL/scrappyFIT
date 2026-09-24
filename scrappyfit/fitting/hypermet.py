"""Measured (empirical) detector response: Hypermet line shape with energy-dependent
parameters taken from single-line standards, as an alternative to GeoPIXE's shape.

GeoPIXE's line.pro models every line as an area-normalised Gaussian plus ONE low-side
exponential tail whose amplitude and length come from the .detector file
(tail_amplitude.pro, tail_length.pro). On the Amptek C1 SDD with the 2026 DPP settings
that shape leaves structured residuals on every strong peak: a missing flat shelf
below the peak, a tail that is too long below the Si K edge and too short just above
it, and nothing for the Al Ka fluorescence the detector itself produces.

Here each line is

    f(x) = beta * [ (1 - ft - fs) G(x) + ft T(x) + fs S(x) ]  (+ Al Ka artefact)

    G   area-normalised Gaussian, width and centroid from the fit (ShapePars)
    T   exponential of slope b (keV) convolved with G - partial charge collection
    S   flat shelf from 0 keV up to the peak, rolled off by G - events that lost an
        arbitrary part of their charge

so beta is the TOTAL recorded intensity of the line and ft, fs are fractions of it.
ft(E), b(E), fs(E) are interpolated from measured points in two branches, below and
above the Si K edge: the absorption depth, and with it the incomplete charge
collection, jumps discontinuously there (1/mu_Si falls from 13 um at Si Ka to 3 um at
Cl Ka), and the measured tail fraction jumps with it (2 % -> 9 %).

Al Ka artefact: photons above the Al K edge (1.5596 keV) fluoresce aluminium inside
the detector housing (Amptek's multilayer collimator ends in Al). Measured at 0.33 %
of Si Ka on three unrelated samples; scaled to other energies by mu_Al(E)/mu_Al(ref).
"""

import json
import numpy as np
from scipy.special import erfc

SQRT2 = np.sqrt(2.0)
KW = 5.5451                      # 8 ln 2: FWHM^2 = KW sigma^2
SI_K_EDGE = 1.8389
AL_K_EDGE = 1.5596
AL_KA = 1.4867


def _interp_log(E, xs, ys, floor=1e-6):
    """log-log interpolation, clamped to the end values outside the table."""
    xs = np.asarray(xs, float); ys = np.maximum(np.asarray(ys, float), floor)
    if len(xs) == 1:
        return float(ys[0])
    return float(np.exp(np.interp(np.log(E), np.log(xs), np.log(ys))))


def _interp_lin(E, xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    return float(ys[0]) if len(xs) == 1 else float(np.interp(E, xs, ys))


class EmpiricalResponse:
    """Energy-dependent Hypermet parameters for one detector + DPP setting.

    below, above   {'E': [...], 'f_tail': [...], 'tail_slope_keV': [...], 'f_shelf': [...]}
                   measured points either side of the Si K edge, E ascending
    al_frac_ref    Al Ka artefact intensity as a fraction of a line at al_ref_E
    """

    def __init__(self, below, above, al_frac_ref=0.0, al_ref_E=1.7398, name='',
                 source='', edge=SI_K_EDGE):
        self.below, self.above = below, above
        self.al_frac_ref, self.al_ref_E = float(al_frac_ref), float(al_ref_E)
        self.name, self.source, self.edge = name, source, float(edge)
        self.mu_al = None            # callable E -> mu_Al(E), set by Session
        self._memo = {}

    def params(self, E):
        """(f_tail, tail_slope_keV, f_shelf) at energy E, keV."""
        key = round(float(E), 5)
        if key not in self._memo:
            br = self.below if E < self.edge or not self.above.get('E') else self.above
            if not br.get('E'):
                br = self.below
            ft = _interp_log(E, br['E'], br['f_tail'])
            b = max(_interp_lin(E, br['E'], br['tail_slope_keV']), 1e-3)
            fs = _interp_log(E, br['E'], br['f_shelf'])
            s = ft + fs
            if s > 0.9:                           # keep the Gaussian fraction positive
                ft, fs = ft * 0.9 / s, fs * 0.9 / s
            self._memo[key] = (ft, b, fs)
        return self._memo[key]

    def al_fraction(self, E):
        if self.al_frac_ref <= 0 or E <= AL_K_EDGE or self.mu_al is None:
            return 0.0
        ref = self.mu_al(self.al_ref_E)
        return 0.0 if not ref else self.al_frac_ref * self.mu_al(E) / ref

    # -- persistence ------------------------------------------------------
    def to_dict(self):
        return dict(name=self.name, source=self.source, edge=self.edge, below=self.below,
                    above=self.above, al_frac_ref=self.al_frac_ref, al_ref_E=self.al_ref_E)

    @classmethod
    def from_dict(cls, d):
        return cls(d['below'], d['above'], d.get('al_frac_ref', 0.0), d.get('al_ref_E', 1.7398),
                   d.get('name', ''), d.get('source', ''), d.get('edge', SI_K_EDGE))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# -- shapes, in channels, each of unit area -----------------------------------
def gauss(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def tail(x, c, s, b):
    """Low-side exponential of slope b (channels) convolved with the Gaussian."""
    b = max(b, 1e-6)
    z = np.clip((x - c) / b + s * s / (2.0 * b * b), -700.0, 700.0)
    return np.exp(z) * erfc((x - c) / (SQRT2 * s) + s / (SQRT2 * b)) / (2.0 * b)


def shelf(x, c, s, c0):
    """Flat from channel c0 (0 keV) up to the peak, rolled off by the Gaussian;
    nothing below c0 - a line cannot lose more than all of its charge."""
    return np.where(x >= c0, erfc((x - c) / (SQRT2 * s)) / (2.0 * max(c - c0, 1.0)), 0.0)


def line_profile(E, beta, pars, n_channels, resp, with_artefact=True):
    """Unit-area (times beta) Hypermet profile of one line over the whole spectrum."""
    f = np.zeros(n_channels)
    if beta == 0:
        return f
    c = float(pars.centroid(E))
    if c < 0 or c > n_channels - 1:
        return f
    w = float(pars.fwhm_channels(E))
    s = w / np.sqrt(KW)
    ft, b_keV, fs = resp.params(E)
    # Optional per-spectrum scaling of the measured tail and shelf, carried in
    # GeoPIXE's tail parameters a5 and a6 (0 means unscaled). Refined when
    # FitOptions.response_scale is set - the same freedom line.pro's tail has.
    k5, k6 = float(pars.a[5]), float(pars.a[6])
    if k5 != 0.0 or k6 != 0.0:
        ft = ft * (k5 * k5 if k5 != 0.0 else 1.0)
        fs = fs * (k6 * k6 if k6 != 0.0 else 1.0)
        if ft + fs > 0.95:
            r = 0.95 / (ft + fs); ft, fs = ft * r, fs * r
    chpk = float(pars.a[3])                        # channels per keV
    b = b_keV * chpk
    c0 = max(float(pars.centroid(0.0)), 0.0)
    lo = max(int(min(c0, c - 8 * s - 12 * b)), 0)
    hi = min(int(c + 6 * s) + 1, n_channels)
    x = np.arange(lo, hi, dtype=float)
    f[lo:hi] = beta * ((1.0 - ft - fs) * gauss(x, c, s) + ft * tail(x, c, s, b)
                       + fs * shelf(x, c, s, c0))
    if with_artefact:
        fa = resp.al_fraction(E)
        if fa > 0:
            f += line_profile(AL_KA, beta * fa, pars, n_channels, resp, with_artefact=False)
    return f


def fit_line(E_axis, counts, lo, hi, E0, neighbours=(), subthr_width=0.5, fix_subthr_width=True):
    """Local Hypermet fit to one isolated line in keV space - the measurement a response
    is built from.

    neighbours  [(label, E_keV)] lines sharing the window (Kb, escape, Al Ka ...), each
                with a free area and a +-30 eV free shift, sharing the main line's shape
    Returns dict(E, centroid, fwhm_eV, f_tail, tail_slope_keV, f_shelf, f_subthr, chi2,
                 neighbours={label: dict(area_ratio, shift_eV)}).
    """
    from scipy.optimize import least_squares
    a = float(np.median(np.diff(E_axis)))
    m = (E_axis >= lo) & (E_axis <= hi)
    xx, yy = E_axis[m], np.asarray(counts, float)[m]

    def hyp(x, A, c, s, ft, b, fs, fp, wp):
        box = (erfc((c - x) / (SQRT2 * s)) - erfc((c + wp - x) / (SQRT2 * s))) / (2 * wp)
        return A * ((1 - ft - fs) * gauss(x, c, s) + ft * tail(x, c, s, b)
                    + fs * erfc((x - c) / (SQRT2 * s)) / 2 / max(c, 1e-3) + fp * box)

    nb = list(neighbours)
    A0 = yy.sum() * a
    wlo, whi = (subthr_width - 1e-4, subthr_width + 1e-4) if fix_subthr_width else (0.05, 0.8)
    p0 = [A0, E0, 0.045, 0.03, 0.03, 0.01, 0.005, subthr_width, max(np.percentile(yy, 5), .1), 0.0]
    lb = [0, E0 - .03, .015, 0, .003, 0, 0, wlo, 0, -np.inf]
    ub = [np.inf, E0 + .03, .12, .6, .5, .3, .1, whi, np.inf, np.inf]
    for _ in nb:
        p0 += [A0 * .02, 0.0]; lb += [0, -.03]; ub += [np.inf, .03]

    def model(p):
        A, c, s, ft, b, fs, fp, wp, b0, b1 = p[:10]
        f = hyp(xx, A / a, c, s, ft, b, fs, fp, wp) * a + b0 + b1 * (xx - E0)
        for k, (lab, En) in enumerate(nb):
            sn = s if 'esc' in lab else s * np.sqrt(max(En, .1) / E0)
            f += hyp(xx, p[10 + 2 * k] / a, En + (c - E0) + p[11 + 2 * k], sn, ft, b, 0, fp, wp) * a
        return f

    def res(p):
        mm = np.maximum(model(p), 1e-6)
        return (yy - mm) / np.sqrt(np.maximum(mm, 1.0))

    r = least_squares(res, p0, bounds=(lb, ub), max_nfev=20000, x_scale='jac')
    p = r.x
    return dict(E=E0, centroid=float(p[1]), fwhm_eV=float(2.3548 * p[2] * 1e3), f_tail=float(p[3]),
                tail_slope_keV=float(p[4]), f_shelf=float(p[5]), f_subthr=float(p[6]),
                chi2=float(np.sum(r.fun ** 2) / max(len(xx) - len(p), 1)),
                neighbours={lab: dict(area_ratio=float(p[10 + 2 * k] / p[0]), shift_eV=float(p[11 + 2 * k] * 1e3))
                            for k, (lab, En) in enumerate(nb)},
                model=(xx, yy, model(p)))


def build(measurements, al_frac_ref=0.0, name='', source='', edge=SI_K_EDGE):
    """Average repeated measurements at the same line and split them at the Si K edge."""
    by = {}
    for d in measurements:
        by.setdefault(round(d['E'], 3), []).append(d)
    br = {'below': dict(E=[], f_tail=[], tail_slope_keV=[], f_shelf=[]),
          'above': dict(E=[], f_tail=[], tail_slope_keV=[], f_shelf=[])}
    for E in sorted(by):
        ds = by[E]; t = br['below' if E < edge else 'above']
        t['E'].append(E)
        for k in ('f_tail', 'tail_slope_keV', 'f_shelf'):
            t[k].append(float(np.mean([d[k] for d in ds])))
    return EmpiricalResponse(br['below'], br['above'], al_frac_ref, 1.7398, name, source, edge)
