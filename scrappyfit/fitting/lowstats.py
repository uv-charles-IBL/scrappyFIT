"""
Statistics-sensitive smoothing filter, ported from low_stats_filter_b in
Workspace/Fortran/image_lib.f (GeoPIXE calls it through call_external).

The filter widens a symmetric box until the enclosed sum is statistically
adequate, then shrinks it back wherever widening would smear a peak. So flat,
low-count regions get heavily averaged while peaks keep their shape.

This is not cosmetic. GeoPIXE runs it before SNIP, and skipping it made my
background 20-45% low on a real spectrum - an error that lands almost entirely
on the weak peaks, which is exactly where trace analysis lives.

Constants are the Fortran ones, unchanged:
    BIG_A = 75   desired sum slope        RATIO = 1.3  maximum side-slope
    BIG_B = 10   desired sum minimum      SCALE = 1.5  FWHM scaling
    TOO_SMALL = 10                        minimum sum
"""

import numpy as np

BIG_A = 75.0
BIG_B = 10.0
TOO_SMALL = 10.0
RATIO = 1.3
SCALE = 1.5


def low_stats_filter(spec, low, high, AF, BF):
    """Return the filtered spectrum.

    spec      counts per channel
    low, high channel range to filter
    AF, BF    FWHM^2 = AF*channel + BF, in channels^2

    AF and BF come from the energy-space width model:
        AF = w1*cal_a / cal_a^2      BF = (w0 + w1*cal_b) / cal_a^2
    exactly as strip_clip.pro computes them.
    """
    spec = np.asarray(spec, dtype=float)
    n = len(spec)
    out = spec.copy()

    low = int(min(max(low, 0), n - 2))
    high = int(min(high, n - 1))
    if low > high - 2:
        return out

    idx = np.arange(n)

    for i in range(low, high + 1):
        fwhm = np.sqrt(max(AF * i + BF, 0.0))

        filt = spec[i]
        big = BIG_A * np.sqrt(abs(filt))
        if big < BIG_B:
            big = BIG_B

        nw = int(round(SCALE * fwhm))
        if nw < 2:
            nw = 2

        # full side sums out to +/- nw
        lo_i = np.clip(i - np.arange(1, nw + 1), 0, n - 1)
        hi_i = np.clip(i + np.arange(1, nw + 1), 0, n - 1)
        left = float(spec[lo_i].sum())
        right = float(spec[hi_i].sum())
        w = 1.0 + 2.0 * nw

        y = filt + left + right

        # shrink the span back while the statistics allow it. The two Z tests
        # compare the left and right sums against each other in units of their
        # own Poisson error - a large asymmetry means a peak edge is inside the
        # window, so the window must not be this wide.
        for j in range(nw, 0, -1):
            sr = abs(right) + 1.0
            sl = abs(left) + 1.0
            z1 = (sr - np.sqrt(sr)) / (sl + np.sqrt(sl))
            z2 = (sl - np.sqrt(sl)) / (sr + np.sqrt(sr))
            if (y <= big) and (z1 <= RATIO) and (z2 <= RATIO):
                break

            left -= spec[min(max(i - j, 0), n - 1)]
            right -= spec[min(max(i + j, 0), n - 1)]
            if filt + left + right <= TOO_SMALL:
                break

            y = filt + left + right
            w -= 2.0

        out[i] = y / w

    return out


def af_bf(w0, w1, cal_a, cal_b):
    """Convert the energy-space width model to the channel-space AF, BF that
    low_stats_filter expects, as strip_clip.pro does."""
    AF = (w1 * cal_a) / (cal_a * cal_a)
    BF = (w0 + w1 * cal_b) / (cal_a * cal_a)
    return AF, BF
