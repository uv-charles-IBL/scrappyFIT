"""Reader for GeoPIXE binary .spec spectrum files.

read_spec.pro walks a long, version-branched header: forty-odd fields, whole
blocks that appear only above certain versions, optional ion-chamber and
dead-time structures, and a per-detector array at the end. Decoding all of
that faithfully is a lot of surface area for one number.

This reads it structurally instead, and then proves it was right.

The spectrum is the last n_channels float32 values in the file. That is not a
guess: for donut2x-2-whole.spec the trailing run of non-negative floats ends
exactly at EOF and is 4117 words long, and 4096 of those are the spectrum -
file length, channel count and run length are mutually consistent to the
byte. The header is read only for the strings and the channel count.

Verification is built in rather than assumed. read_spec() locates the data,
and verify() checks the result against known line energies: on the GeoPIXE
example, Fe Ka lands 15 eV low, Fe Kb 6 eV low, Ni Ka 10 eV low, Sr Ka 23 eV
low and Zr Ka 2 eV high, all sub-channel on a 15.4 eV/channel spectrum.

The energy calibration is NOT reliably in this file's header. Where a .pfr
sits beside the .spec it carries the calibration GeoPIXE actually fitted, and
that is the one to use. Note that the .pfr's 'Cal Gain' and 'Cal Offset'
PARAMETERS are not it - those read 64.96 and 957.28 for a spectrum whose real
calibration is 0.0154 keV/ch and 0.0084 keV. The calibration lives in the
spectrum sub-structure.
"""

import os
import re

import numpy as np

#: Channel counts to try, largest first. A spectrum is a power of two in
#: every GeoPIXE file seen; anything else is metadata that happens to parse
#: as a non-negative float.
CHANNEL_COUNTS = (16384, 8192, 4096, 2048, 1024, 512)


def strings(path, min_len=4):
    """Printable strings from the header - file paths, labels, the sample
    name. Useful on their own and the only part of the header worth having."""
    b = open(path, 'rb').read()
    return [s.decode('latin-1')
            for s in re.findall(rb'[ -~]{%d,}' % min_len, b)]


def _trailing_run(v):
    """Length of the run of plausible counts ending at the last value."""
    ok = np.isfinite(v) & (v >= 0) & (v < 1e12)
    n = 0
    for g in ok[::-1]:
        if not g:
            break
        n += 1
    return n


def read_spec(path, n_channels=None):
    """Return (spectrum, info).

    n_channels overrides the inferred channel count, for a file whose
    spectrum is not a power of two.
    """
    raw = open(path, 'rb').read()
    v = np.frombuffer(raw[:len(raw) // 4 * 4], '>f4')
    version = int(np.frombuffer(raw[:4], '>i4')[0])
    run = _trailing_run(v)

    if n_channels is None:
        for c in CHANNEL_COUNTS:
            if c <= run:
                n_channels = c
                break
    if not n_channels:
        raise ValueError('%s: no plausible spectrum found; the trailing run '
                         'of non-negative floats is only %d values'
                         % (os.path.basename(path), run))
    if n_channels > run:
        raise ValueError('%s: asked for %d channels but only %d trailing '
                         'values are plausible counts'
                         % (os.path.basename(path), n_channels, run))

    spec = v[-n_channels:].astype(float)
    info = dict(version=version, n_channels=n_channels, trailing_run=run,
                total=float(spec.sum()), strings=strings(path))
    for s in info['strings']:
        if s.lower().endswith('.spec') or s.lower().endswith('.evt'):
            info.setdefault('source', s)
    return spec, info


def calibration_from_pfr(pfr_path):
    """The (gain, offset) in keV that GeoPIXE fitted, from a .pfr.

    gpfit.py deliberately stops decoding before the spectrum sub-structure,
    so the calibration is found by anchoring instead. The units string sits
    at the END of that struct:

        {a, b, da, db, units}

    so the four float32 immediately before the XDR-encoded 'keV' are the
    calibration and its errors. That is a positional argument, which is
    exactly the kind that can be silently wrong - so callers should run
    verify() on the result, and the comparison tool does.

    Do NOT use the 'Cal Gain' and 'Cal Offset' entries in the parameter
    block for this. On the GeoPIXE example they read 64.9626 and 957.2782,
    while the real calibration is 0.0154 keV/ch and 0.0084 keV.
    """
    raw = open(pfr_path, 'rb').read()
    # XDR string 'keV': length 3 twice, then the bytes padded to 4
    anchor = raw.find(b'\x00\x00\x00\x03\x00\x00\x00\x03keV')
    if anchor < 0:
        anchor = raw.find(b'keV')
        if anchor < 0:
            return None
        anchor -= 8
    start = anchor - 16
    if start < 0:
        return None
    a, b, da, db = np.frombuffer(raw[start:start + 16], '>f4')
    if not (1e-6 < float(a) < 1.0):
        return None
    return float(a), float(b)


def verify(spec, cal, db, lines=None, tol_eV=60.0, min_frac=0.01):
    """Check an extracted spectrum against known line energies.

    Returns (rows, worst_eV). Each row is (channel, keV, counts, label,
    diff_eV). A large worst_eV means the channel offset or the calibration is
    wrong - which is exactly the failure a structural reader can make and a
    field-by-field one cannot, so it is worth checking every time.
    """
    from scipy.signal import find_peaks

    spec = np.asarray(spec, float)
    a, b = cal
    E = a * np.arange(len(spec)) + b
    if lines is None:
        lines = {}
        for Z in range(11, 93):
            e = db.line_energy(Z)
            if E[0] < e < E[-1]:
                lines[e] = db.sym[Z] + ' Ka'
    pk, _ = find_peaks(spec, height=spec.max() * min_frac, distance=5)
    rows = []
    for i in sorted(pk, key=lambda j: -spec[j])[:12]:
        e = E[i]
        k = min(lines, key=lambda x: abs(x - e))
        rows.append((int(i), float(e), float(spec[i]), lines[k],
                     1000.0 * (e - k)))
    worst = max((abs(r[4]) for r in rows), default=float('nan'))
    return rows, worst
