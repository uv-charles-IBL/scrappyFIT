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


def clocks(path, nhdr=None):
    """The five per-block counters, and what they actually turned out to be.

    The block header is 5 x int32. Traced across four real runs (251239,
    253246, 287424, 287427):

      field 0   a DOSE COUNTER. Starts at 0 or 1, rises monotonically, ends
                in the thousands. This is pulses from the current digitiser,
                NOT microcoulomb - the conversion depends on the digitiser
                range in use during the run, which the file does not record.
                So it gives relative charge exactly and absolute charge only
                once someone supplies uC per count.

      fields 1-4  four free-running clocks in MICROSECONDS, wrapping through
                32 bits. They advance together and sit only ~300-600 us apart
                from one another, which rules out the obvious reading of
                'elapsed vs live': a live clock would fall behind an elapsed
                one by percent, not by microseconds. They look like four
                timestamps latched at slightly different points while the
                block header is written.

    The consequence is worth being blunt about: DEAD TIME IS NOT RECOVERABLE
    FROM THIS FILE. It has to come from OMDAQ's own run log, or from the
    input and output count rates. Returning a fabricated live fraction of 1.0
    would silently bias every absolute concentration low, so this returns
    None instead and lets the caller decide.

    Returns a dict; 'duration_s' is from the clock span and is reliable,
    'charge_counts' is field 0, and 'live_fraction' is always None.
    """
    if nhdr is None:
        nhdr = header_size(path)
    raw = np.fromfile(path, dtype='u1', offset=nhdr)
    nb = len(raw) // BLOCK
    if nb == 0:
        return dict(charge_counts=0, duration_s=None, live_fraction=None,
                    blocks=0)
    hdr = raw[:nb * BLOCK].reshape(nb, BLOCK)[:, :4 * NHDR_I32]
    hdr = hdr.copy().view('<i4')

    charge = int(hdr[-1, 0]) - int(hdr[0, 0])
    # unsigned, because the clock wraps through 2^32 mid-run
    t = hdr[:, 1].astype(np.uint32)
    span = (int(t[-1]) - int(t[0])) % (1 << 32)
    return dict(charge_counts=charge,
                charge_total=int(hdr[-1, 0]),
                duration_s=span * 1e-6,
                live_fraction=None,
                blocks=nb)


def ascii_header(path, nhdr=None):
    """Printable strings from the header, for calibration and sample notes."""
    import re
    if nhdr is None:
        nhdr = header_size(path)
    b = open(path, 'rb').read(nhdr)
    return [s.decode('latin-1') for s in re.findall(rb'[ -~]{4,}', b)]
