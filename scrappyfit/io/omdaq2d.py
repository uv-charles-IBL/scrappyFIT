"""Reader for OMDAQ / GeoPIXE .2D window maps.

A .2D is a single element (or detector) map exported from an OMDAQ list-mode
run: the counts falling in one energy window, per pixel. The header is the
same 4096-byte block the LMF carries - run number, date, sample label, and a
window label such as 'Ti Ka1 PIXE0 (708 - 726)' giving the channel range.

The map itself is run-length coded, one byte per token, decoded here from
the byte statistics of a map whose answer was already known (the same window
rebuilt from the LMF events):

    b < 0x20          skip b pixels (they are zero)
    0x20 <= b < 0xE0  pixel value = b >> 5 (1..4), repeated (b & 0x1F) times
    b >= 0xE0         one pixel, value = b & 0x1F (5..31)

so 0x21 is one pixel of 1, 0x22 two pixels of 1, 0x43 three pixels of 2.
The first attempt had value and repeat the other way round. Totals are
symmetric under that swap - value x repeat is the same product - so every
map summed correctly while every pixel was in the wrong place. The first
mismatch against the LMF rebuild (token 0x22 where the data read 1, 1) is
what settled it.
A value above 31 or a run above 7 needs an escape that these files did not
exercise; decode() reports the counts it placed so a caller can check the
total against the LMF and notice if one appears.

Pixels are written row-major from the top left, the same raster the LMF
events are binned into. Verified on run 380012's Ti Ka map: 40462 events in
the LMF window, and the decoded map places 40462 counts on 30242 pixels,
with every pixel equal to the LMF rebuild.
"""

import os
import re

import numpy as np

HEADER = 4096


def window(path):
    """(label, lo_channel, hi_channel) from the header, or None."""
    b = open(path, 'rb').read(HEADER + 256)
    m = re.search(rb'([A-Za-z][ -~]{2,}?)\s*\((\d+)\s*-\s*(\d+)\)', b)
    if not m:
        return None
    return (m.group(1).decode('latin-1').strip(), int(m.group(2)),
            int(m.group(3)))


def _data_start(b):
    """First byte of the RLE stream.

    After the 4096-byte header come two short groups of little-endian floats
    (scan geometry, then offsets) separated by zeros, and then the stream.
    Those floats decode as plausible RLE tokens and, taken as data, put a
    bright smear of ~350 spurious counts in the top-left corner of every map.
    The stream is found by skipping both float groups: from the header, pass
    two non-zero runs, and start at the third.
    """
    # The float groups occupy 4096-4139 and contain zero bytes of their own
    # (20.0 is 00 00 A0 41), so a run-based skip stops inside them. They end
    # by 4140 in every file seen; the stream is the first non-zero byte after.
    p = HEADER + 44
    n = len(b)
    while p < n and b[p] == 0:
        p += 1
    return p


def decode(path, nx=256, ny=256):
    """Return (image, info). image is (ny, nx) int32."""
    b = open(path, 'rb').read()
    start = _data_start(b)
    img = np.zeros(nx * ny, dtype=np.int32)
    i = 0
    placed = 0
    escapes = 0
    for tok in b[start:]:
        if tok < 0x20:
            i += tok
            continue
        if tok >= 0xE0:
            # escape: one pixel whose value is the low five bits. The
            # encoder uses this for every value of 5 and above - 0xA1 and
            # 0xC1 never occur - so the packed form only carries 1 to 4.
            val = tok & 0x1F
            rep = 1
        else:
            val = tok >> 5
            rep = tok & 0x1F
        if rep == 0:
            escapes += 1
            rep = 1
        for _ in range(rep):
            if i >= img.size:
                break
            img[i] = val
            placed += val
            i += 1
        if i >= img.size:
            break
    info = dict(start=start, filled_to=i, placed=int(placed),
                escapes=escapes, window=window(path),
                name=os.path.basename(path))
    return img.reshape(ny, nx), info


def rebuild_from_lmf(x, y, e, lo, hi, nx=256, ny=256):
    """The same map from LMF events, for checking a decode."""
    m = (e >= lo) & (e <= hi)
    img = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(img, (np.clip(y[m], 0, ny - 1), np.clip(x[m], 0, nx - 1)), 1)
    return img
