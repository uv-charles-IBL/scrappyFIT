"""Reader for GeoPIXE .yield files (write_yield.pro / read_yield.pro).

IDL XDR again, but note the string convention differs from .dam: an empty
string is a single zero-length int32, while a non-empty one is written as its
length twice followed by the padded bytes. Both forms are handled.

What matters here is the layer stack - formula, thickness, density - plus the
beam energy and the take-off geometry, because those are the inputs a yield
calculation needs and they are not recorded anywhere else.
"""
import numpy as np


class _R:
    def __init__(self, b):
        self.b, self.p = b, 0

    def i4(self, n=None):
        if n is None:
            v = int(np.frombuffer(self.b, '>i4', 1, self.p)[0]); self.p += 4; return v
        v = np.frombuffer(self.b, '>i4', n, self.p).astype(int); self.p += 4 * n; return v

    def f4(self, n=None):
        if n is None:
            v = float(np.frombuffer(self.b, '>f4', 1, self.p)[0]); self.p += 4; return v
        v = np.frombuffer(self.b, '>f4', n, self.p).astype(float); self.p += 4 * n; return v

    def s(self):
        n = self.i4()
        if n == 0:
            return ''
        save = self.p
        n2 = self.i4()
        if n2 != n:                      # not the doubled form; rewind
            self.p = save
        raw = self.b[self.p:self.p + n]
        self.p += (n + 3) // 4 * 4
        return raw.decode('latin-1').strip()

    def strarr(self, n):
        return [self.s() for _ in range(n)]


def read_yield(path):
    r = _R(open(path, 'rb').read())
    version = r.i4()
    ny = r.i4() if version <= -6 else 1
    out = []
    for _ in range(ny):
        n, nk, nl = r.i4(), r.i4(), r.i4()
        d = dict(version=version, n=n, nk=nk, nl=nl, title=r.s())
        d['z2'] = r.i4(n); d['shell'] = r.i4(n); d['n_lines'] = r.i4(n)
        d['lines'] = r.i4(nk * n)
        d['intensity'] = r.f4(nk * n); d['e'] = r.f4(nk * n)
        d['yield'] = r.f4(n * nl).reshape(nl, n)
        layers = []
        for _ in range(nl):
            N = r.i4()
            Z = r.i4(32); Frac = r.f4(32)
            thick = r.f4(); name = r.s()
            layers.append(dict(N=N, Z=list(Z[:max(N, 0)]), F=list(Frac[:max(N, 0)]),
                               thick=thick, name=name))
        d['layers'] = layers
        d['unknown'] = r.i4(); d['e_beam'] = r.f4()
        d['theta'] = r.f4(); d['phi'] = r.f4()
        d['alpha'] = r.f4(); d['beta'] = r.f4()
        if version <= -2:
            d['formula'] = r.strarr(nl)
            d['weight'] = r.i4(nl); d['thick'] = r.f4(nl)
            d['microns'] = r.i4(nl); d['density'] = r.f4(nl)
            d['z1'] = r.i4(); d['a1'] = r.i4(); d['state'] = r.f4()
        out.append(d)
    return out
