"""Concentration profiles along user-specified geometry.

An element map answers "where is it". The questions that actually get asked
of one are different: how thick is that rim, does the zoning step or grade,
is the core depleted, what is the mean concentration inside this grain and
how far does it run. Those are all questions about a PATH or a REGION
through the map, not about a pixel.

Four shapes cover almost everything:

    line        two points; the workhorse, for traverses across a boundary
    polyline    a chain of points, for a traverse that follows a feature
    radial      mean at each radius about a centre, for anything concentric
    region      a mask; mean, spread and area inside an arbitrary shape

Two decisions worth stating, because they change the numbers:

Width. A one-pixel-wide traverse across a noisy map is mostly noise. Every
path here takes a width and averages ACROSS the path at each step, which
trades spatial resolution for precision in a way the caller controls and can
report. width=1 is a true single-pixel cut.

Interpolation. Samples land between pixels. Bilinear is the default because
a profile is usually read for the position and width of a transition, and
nearest-neighbour turns a smooth edge into a staircase whose step positions
are an artefact of the grid. Use order=0 when the map is already coarse and
you want to see the actual pixel values rather than a smoothed version.
"""

import numpy as np


def _bilinear(img, r, c):
    """Sample img at fractional (row, col), clamped at the edges."""
    h, w = img.shape
    r = np.clip(r, 0, h - 1.000001)
    c = np.clip(c, 0, w - 1.000001)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr = r - r0
    fc = c - c0
    return (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c1] * (1 - fr) * fc +
            img[r1, c0] * fr * (1 - fc) + img[r1, c1] * fr * fc)


def _nearest(img, r, c):
    h, w = img.shape
    return img[np.clip(np.round(r).astype(int), 0, h - 1),
               np.clip(np.round(c).astype(int), 0, w - 1)]


def line_profile(image, p0, p1, width=1, samples=None, order=1,
                 pixel_size=1.0, reduce='mean'):
    """Profile along the segment p0 -> p1, as (distance, value, spread).

    p0, p1      (x, y) in pixel coordinates, x across and y down, matching
                how the map is displayed
    width       pixels to average ACROSS the path. Odd values are symmetric
                about it; 1 is a single-pixel cut
    samples     points along the path; defaults to one per pixel of length,
                which is the finest that carries real information
    order       1 bilinear, 0 nearest-neighbour
    pixel_size  physical size of a pixel, so distance comes back in microns
                (or whatever unit is passed) rather than pixels
    reduce      'mean' or 'median' across the width. Median resists a hot
                pixel or a dead one; mean is what a concentration should be
                averaged with when the statistics are Poisson

    Returns (distance, value, spread) where spread is the standard deviation
    across the width, zero when width == 1. Reporting it matters: a profile
    with a large cross-path spread is sampling a feature that is not
    perpendicular to the path, and its "edge" is not an edge.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError('line_profile needs a 2-D map')
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < 1e-9:
        raise ValueError('the two points of the line are the same')
    n = int(samples) if samples else max(int(round(length)) + 1, 2)

    t = np.linspace(0.0, 1.0, n)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    # unit normal, to step across the path
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux

    w = max(int(width), 1)
    offsets = np.arange(w, dtype=float) - (w - 1) / 2.0
    samp = _bilinear if order else _nearest

    stack = np.empty((w, n), dtype=float)
    for i, off in enumerate(offsets):
        stack[i] = samp(img, ys + off * ny, xs + off * nx)

    if reduce == 'median':
        value = np.median(stack, axis=0)
    else:
        value = stack.mean(axis=0)
    spread = stack.std(axis=0) if w > 1 else np.zeros(n)
    dist = t * length * float(pixel_size)
    return dist, value, spread


def polyline_profile(image, points, **kw):
    """Profile along a chain of points, distance accumulating across joins.

    Duplicated join samples are dropped so a vertex does not appear twice and
    put a false plateau in the profile.
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError('a polyline needs at least two points')
    dists, vals, spreads = [], [], []
    base = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d, v, s = line_profile(image, a, b, **kw)
        if dists:                      # drop the repeated vertex sample
            d, v, s = d[1:], v[1:], s[1:]
        dists.append(d + base)
        vals.append(v)
        spreads.append(s)
        base = dists[-1][-1] if len(dists[-1]) else base
    return (np.concatenate(dists), np.concatenate(vals),
            np.concatenate(spreads))


def radial_profile(image, centre=None, nbins=None, rmax=None,
                   pixel_size=1.0, mask=None):
    """Mean value in annuli about a centre, as (radius, mean, std, count).

    For anything concentric - a droplet, an inclusion, a diffusion halo, the
    donut in GeoPIXE's own example - this says in one line what a traverse
    only samples in one direction, and it averages over the whole annulus so
    the statistics are far better than any single cut.
    """
    img = np.asarray(image, dtype=float)
    h, w = img.shape
    if centre is None:
        centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    cx, cy = float(centre[0]), float(centre[1])
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)

    good = np.isfinite(img)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    if rmax is None:
        rmax = r[good].max() if good.any() else 0.0
    if nbins is None:
        nbins = max(int(round(rmax)), 1)

    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.digitize(r[good], edges) - 1
    v = img[good]
    ok = (idx >= 0) & (idx < nbins)
    idx, v = idx[ok], v[ok]

    count = np.bincount(idx, minlength=nbins).astype(float)
    total = np.bincount(idx, weights=v, minlength=nbins)
    sq = np.bincount(idx, weights=v * v, minlength=nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        var = np.where(count > 0, sq / np.maximum(count, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    centres = 0.5 * (edges[:-1] + edges[1:]) * float(pixel_size)
    return centres, mean, std, count


def region_stats(image, mask, pixel_size=1.0):
    """Summary of one region of a map.

    'area' is in the square of whatever pixel_size is given in, so passing a
    micron pixel size gives an area in square microns rather than in pixels,
    which is what anyone reporting a grain actually wants.
    """
    img = np.asarray(image, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if m.shape != img.shape:
        raise ValueError('mask shape %s does not match map shape %s'
                         % (m.shape, img.shape))
    v = img[m & np.isfinite(img)]
    if v.size == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'),
                    std=float('nan'), min=float('nan'), max=float('nan'),
                    sum=0.0, area=0.0)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std()), min=float(v.min()), max=float(v.max()),
                sum=float(v.sum()),
                area=float(v.size) * float(pixel_size) ** 2)


def profile_table(maps, p0, p1, elements=None, **kw):
    """One traverse, every element at once.

    Returns (distance, {element: values}). Sampling every map on the SAME
    path is the point: concentrations can then be compared step by step, and
    a ratio taken along the traverse, which is usually more informative than
    any single element because it divides out the beam and the thickness.
    """
    names = list(elements) if elements else list(maps)
    dist = None
    out = {}
    for nm in names:
        if nm not in maps:
            continue
        d, v, _ = line_profile(maps[nm], p0, p1, **kw)
        dist = d if dist is None else dist
        out[nm] = v
    if dist is None:
        raise ValueError('none of the requested elements are in the maps')
    return dist, out


def to_csv(path, distance, series, unit='pixels'):
    """Write a profile to CSV, so it can leave for a plotting package."""
    names = list(series)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('distance_%s,%s\n' % (unit, ','.join(names)))
        for i, d in enumerate(distance):
            fh.write('%.6g,%s\n'
                     % (d, ','.join('%.6g' % series[n][i] for n in names)))
    return path
