"""Elemental maps built with GeoPIXE's own Dynamic Analysis matrix.

This is what DA is for. A window-and-flank map fails wherever a weak line sits
between strong neighbours - the flanking regions land on the neighbours' tails
and the subtraction goes negative, which is exactly what Al and Na do between
Mg and Si in the chondrite. The DA matrix instead carries, for every channel, a
signed weight per element derived from the full fitted peak-shape model, so
overlaps are resolved rather than approximated.

Applying it is one pass over the event list: each event of energy ch adds
matrix[el, ch] to element el at that pixel.
"""
import numpy as np


def da_maps(dam, x, y, e, nbin=256, binning=1, elements=None):
    n = nbin // binning
    xi = (x // binning).astype(np.int64)
    yi = (y // binning).astype(np.int64)
    pix = yi * n + xi
    size = dam['size']
    ok = (e >= 0) & (e < size)
    pix, ee = pix[ok], e[ok].astype(np.int64)

    out = {}
    for j, el in enumerate(dam['el']):
        if elements is not None and el not in elements:
            continue
        if el in ('Back', 'sum'):
            continue
        w = dam['matrix'][j][ee]
        img = np.bincount(pix, weights=w, minlength=n * n).reshape(n, n)
        # each pixel receives charge/npix, not the whole run's charge
        out[el] = img * (n * n) / dam['charge'] / 1e4      # wt%
    return out
