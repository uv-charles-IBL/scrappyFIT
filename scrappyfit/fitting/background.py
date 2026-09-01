"""
SNIP background, ported from GeoPIXE's strip_clip.pro.

Reference for the method:
    C.G. Ryan, E. Clayton, W.L. Griffin, S.H. Sie, D.R. Cousens (1988),
    "SNIP, a Statistics Sensitive Background Treatment for the Quantitative
    Analysis of PIXE Spectra in Geoscience Applications",
    Nucl. Instr. Meth. B34, 396-402.

This is a port, not a reimplementation: the transform, the clipping kernel,
the curvature weighting and the pass structure follow strip_clip.pro line for
line, so that a spectrum stripped here matches one stripped by GeoPIXE.

Included:
  * low_stats_filter  - ported from image_lib.f, see lowstats.py
  * boost_back        - transmission pre-correction; GeoPIXE calls this
                        UNCONDITIONALLY, the 'boost' flag only selects whether
                        filter and sample terms join the detector term

Still not included, and stated rather than hidden:
  * trim_seb          - secondary electron bremsstrahlung hump removal,
                        only applied when explicitly requested
"""

import numpy as np

from .lowstats import low_stats_filter, af_bf


# strip_clip.pro defaults when no detector struct supplies them
W1_DEFAULT = 0.003
W0_DEFAULT = 0.16 * 0.16 - W1_DEFAULT * 5.895

CURVE_DEFAULT = 1.0003
SAMPLE_WIDTH_SCALING = 2.0        # clip FWHM scaling width
N_WIGGLES_TRIM = 4                # reduce wiggles on the last passes

# half_curve, PIXE range
C_LOW = -1.5
C_HIGH = 1.1


def _half_curve(x, n_curve, curve):
    """Curvature weighting applied to the two sampling points."""
    half = np.full(len(x), 0.5, dtype=float)
    if n_curve > 0:
        c_mid = 0.5 * (C_LOW + C_HIGH)
        c_range = 0.5 * (C_HIGH - C_LOW)
        r = x.astype(float) / float(n_curve)
        q = (r > C_LOW) & (r < C_HIGH)
        if q.any():
            y = (r[q] - c_mid) / c_range
            half[q] = 0.5 * (1.0 + (curve - 1.0) * (1.0 - y * y))
    else:
        half = half * curve
    return half


def _simple_strip(spec, low, high, cal_a, cal_b, w0, w1, curve,
                  width_scaling, passes, n_curve, xmin):
    """The SNIP clipping kernel. Operates in place on 'spec'.

    Two clips per pass, exactly as strip_clip.pro does - the second working on
    the result of the first. That double clip is not incidental; halving it
    changes the background level.
    """
    n = len(spec)
    x = np.arange(low, high + 1)
    e = cal_a * x + cal_b
    w = np.sqrt(np.maximum(w0 + w1 * e, 1e-12)) / cal_a
    off = width_scaling * w

    if (high - low) <= 2.0 * off.max():
        low = max(int(low - off.max()), 0)
        high = min(int(high + off.max()), n - 1)
        x = np.arange(low, high + 1)
        e = cal_a * x + cal_b
        w = np.sqrt(np.maximum(w0 + w1 * e, 1e-12)) / cal_a
        off = width_scaling * w

    half = _half_curve(x, n_curve, curve)

    if n_curve == 0:
        rf, f = 1.0, 0.0
    else:
        s = max(width_scaling, 2.0)
        f = 2.0 ** (-((2.0 * s) ** 2))
        rf = 1.0 / (1.0 - 2.0 * f)

    ioff = off.astype(int)
    lo_i = np.maximum(x - ioff, xmin)
    hi_i = np.minimum(x + ioff, n - 1)

    t = spec.copy()
    for _ in range(passes):
        y = half * (spec[lo_i] + spec[hi_i])
        y = rf * (y - 2.0 * f * spec[x])
        t[x] = np.minimum(y, spec[x])

        y = half * (t[lo_i] + t[hi_i])
        y = rf * (y - 2.0 * f * t[x])
        spec[x] = np.minimum(y, t[x])


def _extend_ends(spec, low, high):
    """Anchor the ends: log-linear ramp below 'low', flat above 'high'."""
    n = len(spec)
    ns = int(min(max((low - 1) // 10, 2), 4))
    if low > 0 and low + 2 * ns + 1 < n:
        a = spec[low + 1:low + ns + 1].mean()
        b = spec[low + ns + 1:low + 2 * ns + 1].mean()
        if a > 0 and b > 0:
            slope = (np.log(a) - np.log(b)) / float(ns)
            x = np.arange(low)
            base = max(spec[low], 0.0)
            with np.errstate(over='ignore', invalid='ignore'):
                ramp = np.exp(np.log(base if base > 0 else 1e-10)
                              + slope * (low - x))
            ramp[~np.isfinite(ramp)] = spec[low]
            spec[x] = np.maximum(spec[x], np.maximum(ramp, spec[low]))
    if high + 1 < n:
        x = np.arange(high + 1, n)
        spec[x] = np.maximum(spec[x], max(spec[max(high - 10, 0):high + 1].mean(), 0.0))
    spec[~np.isfinite(spec)] = 0.0
    q = np.nonzero(spec > 0)[0]
    return int(q[0]) if len(q) else 2


def boost_correction(E, eff=None, filter_trans=None, sample_trans=None,
                     cal_a=0.0017, smooth_kev=0.16):
    """Transmission factor applied before stripping, from boost_back.pro.

    SNIP assumes a smooth continuum, but detector efficiency collapses at low
    energy - 0.03 at C Ka on a C1 window - which bends the observed continuum
    sharply downward and makes SNIP under-estimate the background there. This
    divides the distortion out, strips, then puts it back.

    'eff' is det_eff(/skip_abs). The filter and sample terms are the ones the
    'boost' flag switches on; the detector term is always applied.

    The smoothing width is 0.16 keV converted to channels, as in boost_back.
    """
    trans = np.ones_like(E, dtype=float)
    if eff is not None:
        trans = trans * np.asarray(eff, dtype=float)
    if filter_trans is not None:
        trans = trans * np.asarray(filter_trans, dtype=float)
    if sample_trans is not None:
        trans = trans * np.asarray(sample_trans, dtype=float)
    good = np.isfinite(trans) & (trans > 0)
    if not good.any():
        return np.ones_like(E, dtype=float)
    trans[~good] = np.interp(E[~good], E[good], trans[good])
    k = max(int(smooth_kev / cal_a), 2)
    kern = np.ones(k) / k
    trans = np.convolve(trans, kern, mode='same')
    return np.maximum(trans, 1e-6)


def snip(counts, cal_a, cal_b, elow, ehigh, passes=8, def_passes=None,
         w0=None, w1=None, curve=CURVE_DEFAULT, hybrid=True,
         e_trans=None, use_low_stats=True, trans=None):
    """Return the SNIP background for a spectrum.

    counts        spectrum, counts per channel
    cal_a, cal_b  energy = cal_a*channel + cal_b, keV
    elow, ehigh   fitting range, keV. elow is floored at 0.15 keV as in GeoPIXE.
    passes        SNIP passes below the transition energy
    def_passes    passes above it (GeoPIXE default 2*passes)
    w0, w1        peak width model, FWHM^2 = w0 + w1*E (keV^2)
    hybrid        more passes at high energy, as GeoPIXE does by default
    e_trans       transition energy for the hybrid split. GeoPIXE derives this
                  from the filter inflection ('e_inflection'); with no filter,
                  pass it explicitly or leave None to strip in one range.
    """
    if w1 is None:
        w1 = W1_DEFAULT
    if w0 is None:
        w0 = W0_DEFAULT
    if def_passes is None:
        def_passes = 2 * passes

    elow = max(elow, 0.15)
    ehigh = max(ehigh, elow + 2.0)

    n = len(counts)
    t = np.asarray(counts, dtype=float).copy()

    low = max(int((elow - cal_b) / cal_a), 1)
    high = min(int((ehigh - cal_b) / cal_a), n - 2)
    if n < 50 or high < low + 50:
        raise ValueError('spectrum or fit range too short for SNIP')

    # veto_ends: pull low/high in past any leading or trailing empty channels
    low = max(min(low, n - 1), 5)
    high = max(min(high, n - 5), low)
    q = np.nonzero(t[low:high + 1] > 0.1)[0]
    if not len(q):
        raise ValueError('no non-zero channels in the fit range')
    low = max(low, low + int(q[0]))
    high = min(high, low + int(q[-1]))

    # Statistics-sensitive smoothing, before anything else, as strip_clip does.
    if use_low_stats:
        AF, BF = af_bf(w0, w1, cal_a, cal_b)
        t = low_stats_filter(t, low, high, AF, BF)

    xmin = _extend_ends(t, low, high)

    # Transmission pre-correction (boost_back). Divide out here, multiply back
    # after stripping, so SNIP sees an undistorted continuum.
    if trans is not None:
        t = t / trans

    # LLS double-log transform
    t = np.log(np.maximum(np.log(np.maximum(t, 0.0) + 1.0), 0.0) + 1.0)

    n_curve = 0
    if e_trans is not None:
        n_curve = int((e_trans - cal_b) / cal_a)

    if hybrid and e_trans is not None:
        n_trans_low = max(int(0.8 * n_curve), low)
        n_trans_high = min(int(1.2 * n_curve), high)
        if n_trans_high > low + 50:
            _simple_strip(t, low, n_trans_high, cal_a, cal_b, w0, w1, curve,
                          SAMPLE_WIDTH_SCALING, passes, n_curve, xmin)
        if high > n_trans_low + 50:
            _simple_strip(t, n_trans_low, high, cal_a, cal_b, w0, w1, curve,
                          SAMPLE_WIDTH_SCALING, def_passes, n_curve, xmin)
    else:
        _simple_strip(t, low, high, cal_a, cal_b, w0, w1, curve,
                      SAMPLE_WIDTH_SCALING, passes, n_curve, xmin)

    # wiggle-reduction passes, narrowing each time
    nm = 4 if N_WIGGLES_TRIM >= 1 else 2
    wiggle_passes = max((N_WIGGLES_TRIM + 1) // 4, 1)
    r = 1.0
    for _ in range(nm):
        _simple_strip(t, low, high, cal_a, cal_b, w0, w1, 1.0,
                      r * SAMPLE_WIDTH_SCALING, wiggle_passes, 0, xmin)
        r *= 0.7

    back = np.exp(np.exp(t) - 1.0) - 1.0
    if trans is not None:
        back = back * trans
    return np.maximum(back, 0.1)
