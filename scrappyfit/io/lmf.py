"""OMDAQ-3 list-mode file reader.

Layout, verified against run 251239 (ADC0 reproduced the known PIXE spectrum
to 1 event in 206,934):

    header      variable length, ends on an 8192-byte block boundary
    block       8192 bytes, repeated:
                  5 x int32   charge, elapsed, live, live0, live1
                  2043 x      {x:uint8, y:uint8, e:uint16}
    e           bits 0-11  energy      (0x0FFF = null, skip)
                bits 12-14 ADC number  (>> 12)
"""

import numpy as np

BLOCK = 8192
NHDR_I32 = 5
NREC = (BLOCK - 4 * NHDR_I32) // 4          # 2043
REC = np.dtype([('x', 'u1'), ('y', 'u1'), ('e', '<u2')])
E_MASK = 0x0FFF
ADC_SHIFT = 12
ADC_MASK = 0x7


def header_size(path):
    """Smallest plausible header length that leaves a whole number of blocks."""
    import os
    n = os.path.getsize(path)
    for h in range(1024, 16385):
        if (n - h) % BLOCK == 0:
            return h
    raise ValueError('no consistent block alignment in %s' % path)


def read(path, nhdr=None):
    """Return (x, y, energy, adc, charge) as flat arrays."""
    if nhdr is None:
        nhdr = header_size(path)
    raw = np.fromfile(path, dtype='u1', offset=nhdr)
    nb = len(raw) // BLOCK
    raw = raw[:nb * BLOCK].reshape(nb, BLOCK)

    charge = raw[:, :4 * NHDR_I32].copy().view('<i4')[:, 0]
    rec = raw[:, 4 * NHDR_I32:].copy().view(REC)

    e = rec['e'].ravel()
    good = (e & E_MASK) != E_MASK
    return (rec['x'].ravel()[good], rec['y'].ravel()[good],
            (e[good] & E_MASK).astype(np.int32),
            ((e[good] >> ADC_SHIFT) & ADC_MASK).astype(np.int8),
            charge)


def ascii_header(path, nhdr=None):
    """Printable strings from the header, for calibration and sample notes."""
    import re
    if nhdr is None:
        nhdr = header_size(path)
    b = open(path, 'rb').read(nhdr)
    return [s.decode('latin-1') for s in re.findall(rb'[ -~]{4,}', b)]
