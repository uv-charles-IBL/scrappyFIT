"""Elemental maps from an OMDAQ list-mode file.

Each event carries (x, y, energy), so a map is just a 2-D histogram of the
events falling in an element's line window. The only real subtlety is the
background: at these energies the continuum under a weak line can exceed the
line itself, and an un-subtracted map shows the continuum's spatial structure
rather than the element's.

So each window is paired with flanking windows, interpolated linearly under the
peak and subtracted per pixel. That is the mapping analogue of what SNIP does
to a spectrum - cruder, but it is per-pixel and it removes the dominant term.
"""

import numpy as np


def window(e_lo, e_hi, cal_a, cal_b):
    return (int(round((e_lo - cal_b) / cal_a)),
            int(round((e_hi - cal_b) / cal_a)))


def element_map(x, y, e, ch_lo, ch_hi, bg=None, nbin=256, binning=1):
    """2-D counts map for channels [ch_lo, ch_hi], optionally background
    subtracted using bg = ((lo1,hi1),(lo2,hi2)) flanking windows."""
    n = nbin // binning
    xi = (x // binning).astype(np.int32)
    yi = (y // binning).astype(np.int32)

    def hist(sel):
        return np.bincount(yi[sel] * n + xi[sel],
                           minlength=n * n).reshape(n, n).astype(float)

    peak = hist((e >= ch_lo) & (e <= ch_hi))
    if bg is None:
        return peak

    (l1, h1), (l2, h2) = bg
    b1 = hist((e >= l1) & (e <= h1))
    b2 = hist((e >= l2) & (e <= h2))
    w1, w2, wp = (h1 - l1 + 1), (h2 - l2 + 1), (ch_hi - ch_lo + 1)
    # linear interpolation of the continuum across the peak window
    c1, c2 = 0.5 * (l1 + h1), 0.5 * (l2 + h2)
    cp = 0.5 * (ch_lo + ch_hi)
    f = (cp - c1) / (c2 - c1) if c2 != c1 else 0.5
    back = wp * ((1 - f) * b1 / w1 + f * b2 / w2)
    return peak - back


def smooth(m, sigma=1.0):
    """Small Gaussian, for maps that are counting-limited per pixel.

    Normalised by the kernel weight that lands inside the field, so a pixel
    on the edge is the Gaussian-weighted mean of its in-field neighbours
    (fewer of them, so edge pixels are noisier). np.convolve 'same'
    zero-pads: a flat map came out 30 % low on the edge rows and 51 % low in
    the corners (smooth(np.ones((64, 64)), 1.0), 2026-10-20), a false dark
    rim that percentile masks and colour limits then saw.
    """
    if sigma <= 0:
        return m
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()

    def conv(a):
        a = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 0, a)
        return np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 1, a)

    m = np.asarray(m, dtype=float)
    return conv(m) / conv(np.ones(m.shape))
