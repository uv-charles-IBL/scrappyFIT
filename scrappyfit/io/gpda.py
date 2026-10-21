"""Reader for GeoPIXE .dam files - the Dynamic Analysis matrix.

Format follows write_da.pro / read_da.pro: an IDL XDR (big-endian) stream.
The useful parts for cross-checking a fit are

    el      element/line names, one per matrix row
    yield   counts_per_ppm_uC - GeoPIXE's yield factor per element
    mdl     minimum detection limit at the stored charge
    matrix  the DA projection itself, one row per element
    cal     energy calibration of the matrix rows

IDL writes strings in XDR as the length repeated as two int32s, followed by
the bytes zero-padded to a 4-byte boundary. IDL INT (16-bit) is promoted to 4
bytes by XDR, which is why n_pure/on/n_det are read as int32 here even though
read_da.pro declares them with 16-bit literals.
"""

import numpy as np


class _R:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def i4(self):
        v = int(np.frombuffer(self.b, '>i4', 1, self.p)[0]); self.p += 4; return v

    def f4(self, n=None):
        if n is None:
            v = float(np.frombuffer(self.b, '>f4', 1, self.p)[0]); self.p += 4; return v
        v = np.frombuffer(self.b, '>f4', n, self.p).astype(float); self.p += 4 * n
        return v

    def s(self):
        # IDL XDR writes a string as its length TWICE, then the bytes padded
        # to a 4-byte boundary.
        n = self.i4()
        n2 = self.i4()
        if n != n2:
            raise ValueError('string length mismatch %d/%d at %d' % (n, n2, self.p))
        raw = self.b[self.p:self.p + n]
        self.p += (n + 3) // 4 * 4
        return raw.decode('latin-1')

    def strarr(self, n):
        return [self.s() for _ in range(n)]


def read_dam(path):
    r = _R(open(path, 'rb').read())
    version = r.i4()
    nda_extra = r.i4()
    out = []
    for _ in range(nda_extra + 1):
        d = {}
        d['label'] = r.s()
        n = r.i4()
        d['n_el'] = n
        d['cal_orig'] = (r.f4(), r.f4())
        d['cal'] = (r.f4(), r.f4())
        d['charge'] = r.f4()
        d['el'] = r.strarr(n)
        d['ecompress'] = r.i4() if version <= -3 else 1
        d['mdl'] = r.f4(n)
        d['yield'] = r.f4(n) if version <= -2 else None
        size = r.i4()
        d['size'] = size
        d['matrix'] = r.f4(size * n).reshape(n, size)   # IDL fltarr(size,n)
        d['station'] = r.i4() if version <= -6 else 0
        d['density0'] = r.f4() if version <= -5 else 0.0
        d['thick'] = r.f4() if version <= -8 else 0.0
        if version <= -4:
            d['mu_zero'] = r.f4(n) if r.i4() else None
        if version <= -2:
            npure = r.i4()
            d['n_pure'] = npure
            d['pure'] = r.f4(size * npure).reshape(npure, size) if npure > 0 else None
        if version <= -7:
            on = r.i4()
            d['array_on'] = on
            if on > 0:
                nd = r.i4()
                d['rGamma'] = r.f4(nd * n).reshape(n, nd) if nd > 0 else None
        d['E_beam'] = r.f4() if version <= -9 else 0.0
        d['version'] = version
        out.append(d)
    return out
