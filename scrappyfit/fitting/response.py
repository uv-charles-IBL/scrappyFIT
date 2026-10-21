"""Extended detector response for the Amptek X-123 SDD, C1 window, 500 um Si.

GeoPIXE models a peak as an area-normalised Gaussian plus one low-side
exponential tail. That is adequate above ~1 keV and demonstrably not adequate
below it: fitting the light-element region leaves 1-6% of each peak's area
unaccounted for, ~50 eV below the centroid, and the fit hands that intensity
to whatever element has a line nearby. That is how a chondrite acquired
titanium it does not contain.

The physical reason is incomplete charge collection. A 0.28 keV photon is
absorbed within ~100 nm of the entrance contact, inside or adjacent to the
dead layer, where the field is weak and part of the charge cloud is lost.
A 1.74 keV photon penetrates microns into fully depleted silicon and is
collected cleanly. So the defect is strongly energy dependent and confined to
the light elements - exactly what the data shows.

Three terms, the standard Hypermet decomposition:

    Gaussian    the fully collected events
    tail        exponential on the low side, convolved with the Gaussian -
                partial charge collection
    shelf       a flat step extending to zero energy - events losing an
                arbitrary fraction of their charge

    f(x) = A * [ (1 - ft - fs) * G(x) + ft * T(x) + fs * S(x) ]

with T and S each normalised to unit area over the peak's own support, so ft
and fs are directly the fraction of the peak's counts in each component.

Reference for the functional form:
    Campbell & Maenhaut style Hypermet, as used across PIXE codes; the shelf
    term is what GeoPIXE's shape omits.
"""
import numpy as np
from scipy.special import erfc

SQRT2 = np.sqrt(2.0)


def gauss(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def tail(x, c, s, beta):
    """Exponential low-side tail convolved with the Gaussian. Unit area."""
    b = max(beta, 1e-6)
    z = (x - c) / b + s * s / (2 * b * b)
    z = np.clip(z, -700, 700)
    out = np.exp(z) * erfc((x - c) / (SQRT2 * s) + s / (SQRT2 * b))
    return out / (2.0 * b)


def shelf(x, c, s, width):
    """Flat step below the peak, rolled off by the Gaussian. Unit area over
    'width' channels below the centroid."""
    w = max(width, 1.0)
    return erfc((x - c) / (SQRT2 * s)) / (2.0 * w)


def peak(x, area, c, s, ft, beta, fs, shelf_width):
    core = (1.0 - ft - fs) * gauss(x, c, s)
    return area * (core + ft * tail(x, c, s, beta) + fs * shelf(x, c, s, shelf_width))


def fit_peak(ch, y, c0, s0, lo, hi, fit_shelf=True, fit_tail=True):
    """Fit one isolated peak over channels [lo, hi] with a linear background.

    Returns dict of fitted parameters and the reduced chi2. Weights are
    Poisson on the model, iterated, as elsewhere in this package.
    """
    from scipy.optimize import least_squares
    m = (ch >= lo) & (ch <= hi)
    x, yy = ch[m].astype(float), y[m].astype(float)
    width = hi - lo

    def unpack(p):
        area, c, s, ft, beta, fs, b0, b1 = p
        return area, c, s, ft, beta, fs, b0, b1

    def model(p):
        area, c, s, ft, beta, fs, b0, b1 = unpack(p)
        return peak(x, area, c, s, ft, beta, fs, width) + b0 + b1 * (x - c0)

    def resid(p):
        mm = np.maximum(model(p), 1e-9)
        return (yy - mm) / np.sqrt(np.maximum(mm, 1.0))

    a0 = max(yy.sum() - len(yy) * np.median(yy), 1.0)
    p0 = [a0, c0, s0, 0.05 if fit_tail else 0.0, max(s0, 1.0),
          0.01 if fit_shelf else 0.0, max(np.median(yy), 0.1), 0.0]
    lob = [0, c0 - 15, 0.3 * s0, 0.0, 0.2, 0.0, 0.0, -np.inf]
    upb = [np.inf, c0 + 15, 3.0 * s0,
           0.6 if fit_tail else 1e-9, 50.0,
           0.3 if fit_shelf else 1e-9, np.inf, np.inf]
    r = least_squares(resid, p0, bounds=(lob, upb), max_nfev=4000)
    area, c, s, ft, beta, fs, b0, b1 = r.x
    dof = max(len(x) - len(p0), 1)
    return dict(area=area, centroid=c, sigma=s, f_tail=ft, beta=beta,
                f_shelf=fs, b0=b0, b1=b1,
                chi2=float(np.sum(r.fun ** 2)) / dof, n=len(x))
