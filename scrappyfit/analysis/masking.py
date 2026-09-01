"""Region selection on an elemental map.

A rectangle is enough to test an idea, and a percentile threshold is enough to
isolate a phase that is brighter than its surroundings. Neither is enough for
a real mineral grain, which is an irregular shape whose intensity overlaps the
matrix at its edges. Hence two more tools:

    lasso        an arbitrary polygon the user draws
    flood        pick a pixel, grow outward through connected pixels that are
                 similar to it

Flood fill is the more useful of the two, because it follows the grain rather
than the operator's hand, and because "connected and similar" is close to what
a mineral phase actually is. Its tolerance is expressed as a fraction of the
seed value rather than an absolute, so the same setting behaves sensibly on a
bright grain and a faint one.

All of these return a boolean array in MAP coordinates. Session.set_mask
upsamples to the raster.
"""

import numpy as np


def rectangle(shape, x0, y0, x1, y1):
    m = np.zeros(shape, dtype=bool)
    y0, y1 = sorted((max(int(y0), 0), min(int(y1), shape[0] - 1)))
    x0, x1 = sorted((max(int(x0), 0), min(int(x1), shape[1] - 1)))
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def threshold(data, percentile=90.0, above=True):
    d = np.asarray(data, dtype=float)
    t = np.nanpercentile(d, percentile)
    return (d >= t) if above else (d <= t)


def polygon(shape, vertices):
    """Even-odd point-in-polygon test over the map grid.

    Written out rather than pulled from matplotlib.path so that masking works
    headlessly and in a batch, where importing a GUI toolkit would be wrong.
    """
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 3:
        return np.zeros(shape, dtype=bool)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    px, py = xx.ravel() + 0.5, yy.ravel() + 0.5
    inside = np.zeros(px.shape, dtype=bool)
    x0, y0 = verts[:, 0], verts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        crosses = ((ay > py) != (by > py))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (bx - ax) * (py - ay) / (by - ay) + ax
        inside ^= crosses & (px < xint)
    return inside.reshape(shape)


def flood(data, seed_xy, tolerance=0.25, connectivity=4, max_pixels=None):
    """Grow a region from a seed pixel through connected similar pixels.

    tolerance is RELATIVE to the seed value: 0.25 accepts anything within
    +-25% of it. Relative rather than absolute so the same setting works on a
    bright grain and a faint one, which matters when the same batch runs over
    samples of different concentration.

    connectivity 4 (edges) or 8 (edges and corners). Four is the safer
    default: eight will leak through a single-pixel diagonal touch, which on
    a noisy map happens more often than you would like.
    """
    d = np.asarray(data, dtype=float)
    ny, nx = d.shape
    x, y = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= x < nx and 0 <= y < ny):
        return np.zeros(d.shape, dtype=bool)
    v0 = d[y, x]
    if not np.isfinite(v0):
        return np.zeros(d.shape, dtype=bool)
    span = abs(v0) * float(tolerance)
    if span <= 0:
        span = max(np.nanstd(d) * tolerance, 1e-9)
    lo, hi = v0 - span, v0 + span

    ok = np.isfinite(d) & (d >= lo) & (d <= hi)
    out = np.zeros(d.shape, dtype=bool)
    if not ok[y, x]:
        return out

    steps = ([(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else
             [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              if (dx, dy) != (0, 0)])
    stack = [(x, y)]
    out[y, x] = True
    count = 1
    while stack:
        cx, cy = stack.pop()
        for dx, dy in steps:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny and ok[ny_, nx_] \
                    and not out[ny_, nx_]:
                out[ny_, nx_] = True
                count += 1
                if max_pixels and count >= max_pixels:
                    return out
                stack.append((nx_, ny_))
    return out


def grow(mask, n=1):
    """Dilate by n pixels. A grain's edge pixels are mixed with the matrix, so
    growing INTO a grain is wrong but growing a seed region out to catch the
    full grain before eroding back is sometimes what you want."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(n)):
        g = m.copy()
        g[1:, :] |= m[:-1, :]
        g[:-1, :] |= m[1:, :]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def shrink(mask, n=1):
    """Erode by n pixels - the usual way to drop mixed edge pixels before
    quantifying a phase."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(n)):
        g = m.copy()
        g[1:, :] &= m[:-1, :]
        g[:-1, :] &= m[1:, :]
        g[:, 1:] &= m[:, :-1]
        g[:, :-1] &= m[:, 1:]
        m = g
    return m


def invert(mask):
    return ~np.asarray(mask, dtype=bool)


def combine(a, b, how='or'):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {'or': a | b, 'and': a & b, 'sub': a & ~b, 'xor': a ^ b}[how]


def describe(mask):
    m = np.asarray(mask, dtype=bool)
    return '%d of %d pixels (%.1f%%)' % (m.sum(), m.size, 100 * m.mean())
