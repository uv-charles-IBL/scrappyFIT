"""Reader for GeoPIXE's hubbell.dat - the Berger & Hubbell (XCOM) tables.

XCOM is the Berger-Hubbell evaluation, so this file IS the XCOM dataset;
nothing needs importing to pair FFAST with XCOM the way Heirwegh does.

Layout from init_hubbell.pro, IDL XDR:

    ntot, nmax                       two int16 (4 bytes each in XDR)
    then ntot copies of:
        el         string   (length written twice, then padded bytes)
        Z, n       int16
        e          float32[nmax]      keV
        units      string
        Rayleigh, Compton, Photo, Pair, Atten_All, Atten
                   float32[nmax] each

Only the Photo column is used here: the absorption correction in a
thick-target yield is photoelectric, and the other tabulations in this package
are photoelectric-only too, so mixing them stays consistent.
"""
import os
import numpy as np

_PAD = 4


class _R:
    def __init__(self, b):
        self.b, self.p = b, 0

    def i4(self):
        v = int(np.frombuffer(self.b, '>i4', 1, self.p)[0]); self.p += 4; return v

    def f4(self, n):
        v = np.frombuffer(self.b, '>f4', n, self.p).astype(float); self.p += 4 * n
        return v

    def s(self):
        n = self.i4()
        if n == 0:
            return ''
        save = self.p
        n2 = self.i4()
        if n2 != n:
            self.p = save
        raw = self.b[self.p:self.p + n]
        self.p += (n + _PAD - 1) // _PAD * _PAD
        return raw.decode('latin-1').strip()


def read_hubbell(dat_dir):
    """{Z: (energies_keV, photo_cm2_per_g)} for every tabulated element."""
    r = _R(open(os.path.join(dat_dir, 'hubbell.dat'), 'rb').read())
    ntot, nmax = r.i4(), r.i4()
    out = {}
    for _ in range(ntot):
        el = r.s()
        Z, n = r.i4(), r.i4()
        e = r.f4(nmax)
        r.s()                                    # units
        r.f4(nmax)                               # Rayleigh
        r.f4(nmax)                               # Compton
        photo = r.f4(nmax)
        r.f4(nmax)                               # Pair
        r.f4(nmax)                               # Atten_All
        r.f4(nmax)                               # Atten
        if 1 <= Z <= 92 and n > 1:
            ee, pp = e[:n], photo[:n]
            ok = np.isfinite(ee) & np.isfinite(pp) & (ee > 0) & (pp > 0)
            if ok.sum() > 3:
                out[Z] = (ee[ok], pp[ok])
    return out
