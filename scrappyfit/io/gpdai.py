"""Reader for GeoPIXE Dynamic Analysis image files (.dai).

A .dai is an XDR (big-endian) dump written by write_geopixe_image.pro: a long
versioned header, then one float32 frame per element, then, when the file
carries errors, one half-size error frame per element. The header grew a new
field with almost every version, and read_geopixe_image.pro walks all of
them, so the honest way to find the images is not to parse the header but to
count back from the END of the file:

    images start at  len(file) - n_el * (xsize*ysize + xesize*yesize) * 4

with n_el the element count, xsize/ysize the map size and xesize/yesize the
error-map size, all of which sit near the front of the header at fixed
places. On 287427 (version -56) that puts the image block at byte 161008.

The first attempt at this reader guessed the block at byte 596, straight
after the element names, and read 40,103 floats of header and flux maps as
the start of the images. Every map came out as the bottom half of one
element glued to the top half of the next, the correlation with our own maps
was 0.05 for F and 0.001 for Al, and the conclusion drawn - that GeoPIXE's
DA matrix for this sample was degenerate - was wrong. Read from the right
offset the same maps correlate at 0.79 (F), 0.96 (Al), 0.98 (Si) and 0.996
(Cl) with ours. The lesson is recorded here so that it is not relearned.
"""

import re

import numpy as np


def _names(b):
    """Element names: XDR strings in the block that starts with 'Back'."""
    start = b.find(b'Back')
    if start < 0:
        raise ValueError('no element list found')
    names, p = [], start - 8
    while len(names) < 400:
        n = int(np.frombuffer(b[p:p + 4], '>i4')[0])
        if not (0 <= n <= 16):
            break
        p += 4
        if n and int(np.frombuffer(b[p:p + 4], '>i4')[0]) == n:
            p += 4
        nm = b[p:p + n].decode('latin-1') if n else ''
        p += (n + 3) // 4 * 4 if n else 0
        names.append(nm)
        if nm == 'sum':
            break
    return names


def read_dai(path):
    """Return dict(version, names, xsize, ysize, images{name: (ny, nx) float},
    charge, source, dam)."""
    b = open(path, 'rb').read()
    i4 = np.frombuffer(b[:128], '>i4')
    version = int(i4[0])
    # the size words follow two XDR strings (source and label); scan for the
    # first run of four plausible sizes: xsize ysize xesize yesize
    sizes = None
    for k in range(3, 28):
        xs, ys, xe, ye = (int(v) for v in i4[k:k + 4])
        if 1 <= xs <= 8192 and 1 <= ys <= 8192 and xe in (xs // 2, (xs + 1) // 2) \
                and ye in (ys // 2, (ys + 1) // 2):
            sizes = (xs, ys, xe, ye)
            break
    if sizes is None:
        raise ValueError('map sizes not found in header')
    xs, ys, xe, ye = sizes
    names = _names(b)
    n_el = len(names)
    n_img = xs * ys
    n_err = xe * ye
    # count back from the end: images then error maps, or images alone
    start = len(b) - n_el * (n_img + n_err) * 4
    has_err = True
    if start < 596:
        start = len(b) - n_el * n_img * 4
        has_err = False
    if start < 596:
        raise ValueError('file too short for %d maps of %dx%d' % (n_el, xs, ys))
    images = {}
    for k, nm in enumerate(names):
        v = np.frombuffer(b, '>f4', n_img, start + k * n_img * 4)
        images[nm] = v.reshape(ys, xs).astype(np.float64)
    strings = [m.decode('latin-1') for m in re.findall(rb'[ -~]{4,}', b[:4096])]
    def _path(t):
        # the XDR length prefix decodes as a stray character before the path
        m = re.search(r'[A-Za-z]:[\\/].*', t)
        return m.group(0) if m else t
    dam = [_path(s) for s in strings if s.lower().endswith('.dam')]
    src = [_path(s) for s in strings if s.lower().endswith('.lmf')]
    return dict(version=version, names=names, xsize=xs, ysize=ys,
                images=images, has_errors=has_err, image_offset=start,
                dam=dam[0] if dam else '', source=src[0] if src else '')
