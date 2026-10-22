"""Element association (correlation) plots: one map against another.

Each pixel is a point (value in map A, value in map B). Phases show up as
clusters, mixing lines as streaks between them, and a region drawn round a
cluster picks out every pixel of that phase wherever it lies on the map -
which a shape drawn on the map itself cannot do.

GeoPIXE's version (corr.pro, corr_routines.pro; geopixe-ref 2026-10-21):
negatives clipped to 0 (corr_routines.pro:556-593), a 2-D histogram over
0 .. 1.05 x the maximum with a 2 % border ignored (:74-91), optional log axes
5 decades deep (:109-130), and a 10-point spline on the histogram that keeps
every pixel whose BIN it encloses (:728-735). Here the test is on each pixel's
own point, not its bin, so the selection does not depend on the binning.
"""

import numpy as np
from matplotlib.path import Path

LOG_DECADES = 5       # as GeoPIXE's log axes, corr_routines.pro:109-130


def _interior(m, border=0.02):
    """The map without a 2 % border, as GeoPIXE takes its range."""
    ny, nx = m.shape
    by, bx = int(round(border * ny)), int(round(border * nx))
    return m[by:ny - by or None, bx:nx - bx or None]


def axis_range(m, log=False):
    """(lo, hi) of one axis: 0 .. 1.05 x the interior maximum, or for log
    5 decades below that."""
    m = np.asarray(m, float)
    top = np.nanmax(_interior(m)) if m.size else 0.0
    if not np.isfinite(top) or top <= 0:
        top = 1.0
    hi = 1.05 * top
    return (hi * 10.0 ** -LOG_DECADES, hi) if log else (0.0, hi)


def _coords(m, log, lims):
    """Pixel values as plot coordinates: negatives clipped to 0, anything
    over the top (a hot pixel in the 2 % border) put on it, and on a log
    axis log10 with everything under the bottom put on the bottom. So every
    pixel is on the plot."""
    v = np.clip(np.asarray(m, float), 0, lims[1])
    if not log:
        return v
    return np.log10(np.maximum(v, lims[0]))


def histogram(a, b, bins=128, log=(False, False), within=None):
    """2-D histogram of map b against map a.

    Returns (H, xedges, yedges, lims): H[i, j] counts pixels in x-bin i and
    y-bin j; edges are in plot coordinates (log10 on a log axis); lims are
    the two axes' (lo, hi) in data units. within restricts it to a mask.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError('maps differ in shape: %s vs %s' % (a.shape, b.shape))
    lims = (axis_range(a, log[0]), axis_range(b, log[1]))
    x, y = _coords(a, log[0], lims[0]), _coords(b, log[1], lims[1])
    rng = [np.log10(l) if lg else l for l, lg in zip(lims, log)]
    sel = np.ones(a.shape, bool) if within is None else np.asarray(within, bool)
    H, xe, ye = np.histogram2d(x[sel], y[sel], bins=bins, range=rng)
    return H, xe, ye, lims


def select(a, b, vertices, log=(False, False)):
    """Pixels whose (a, b) point lies inside a polygon drawn on the plot.

    vertices are in plot coordinates (as histogram's edges: log10 on a log
    axis). Returns a bool map shaped like a.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(vertices) < 3:
        return np.zeros(a.shape, bool)
    lims = (axis_range(a, log[0]), axis_range(b, log[1]))
    pts = np.column_stack([_coords(a, log[0], lims[0]).ravel(),
                           _coords(b, log[1], lims[1]).ravel()])
    inside = Path(np.asarray(vertices, float)).contains_points(pts)
    return inside.reshape(a.shape)


def pearson(a, b, within=None):
    """Pearson r of the two maps' pixel values as plotted, negatives
    clipped to 0 (over a mask, if given)."""
    a = np.clip(np.asarray(a, float), 0, None)
    b = np.clip(np.asarray(b, float), 0, None)
    sel = np.ones(a.shape, bool) if within is None else np.asarray(within, bool)
    x, y = a[sel], b[sel]
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])
