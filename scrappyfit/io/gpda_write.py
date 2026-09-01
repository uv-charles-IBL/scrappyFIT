"""Writing GeoPIXE Dynamic Analysis matrices.

Why bother
----------
Reading GeoPIXE's files lets scrappyFIT check itself against GeoPIXE. Writing
them lets the traffic go the other way: a DA matrix built here - with the
rebuilt line data, the mixed MAC rule and the fuller detector response - can
be loaded into GeoPIXE and used to project real-time maps with its own
machinery. That is the single most useful interop feature, because it means
improvements here reach the existing workflow instead of forking it.

Format follows write_da.pro, XDR big-endian, version -9. The one trap is the
string encoding: IDL writes a string as its length TWICE, then the bytes
padded to a four-byte boundary. Get that wrong and GeoPIXE reports a corrupt
file with no further detail.

What a DA matrix is
-------------------
One row per element. Each row is a set of per-channel weights such that

    concentration = sum over channels of ( matrix[element, channel] x counts )

so applying it is a dot product per pixel, which is why GeoPIXE can project
maps live. The weights come from inverting the fitted pure-element spectra
against each other, so the matrix carries the peak overlap resolution with it.
"""

import numpy as np

XDR_PAD = 4


class _W:
    def __init__(self):
        self.buf = bytearray()

    def i4(self, v):
        self.buf += np.array([int(v)], dtype='>i4').tobytes()
        return self

    def f4(self, v):
        self.buf += np.asarray(np.atleast_1d(v), dtype='>f4').tobytes()
        return self

    def s(self, text):
        """IDL XDR string: length, length again, bytes padded to 4."""
        b = str(text).encode('latin-1', errors='replace')
        n = len(b)
        self.i4(n).i4(n)
        self.buf += b + b'\x00' * ((-n) % XDR_PAD)
        return self

    def strarr(self, items):
        for t in items:
            self.s(t)
        return self


def build_matrix(pure_spectra, names, ppm_scale=None):
    """Least-squares DA weights from a set of pure-element unit spectra.

    pure_spectra : (n_elements, n_channels) - each row the modelled spectrum
                   of one element at unit concentration and unit charge
    returns      : (n_elements, n_channels) weights

    The weights are the pseudo-inverse of the pure spectra, which is exactly
    the statement "given a measured spectrum, what mixture of these produced
    it". Overlapping elements come out anti-correlated in their weights - a
    channel shared by two lines contributes positively to one and negatively
    to the other - and that is the overlap resolution.

    A small ridge term is added because a set of pure spectra with two nearly
    identical members is numerically singular, and a singular inverse produces
    enormous opposed weights that look fine until real noise hits them.
    """
    A = np.asarray(pure_spectra, dtype=float)
    n = A.shape[0]
    G = A @ A.T
    ridge = 1e-9 * np.trace(G) / max(n, 1)
    M = np.linalg.solve(G + ridge * np.eye(n), A)
    if ppm_scale is not None:
        M = M * np.asarray(ppm_scale, dtype=float)[:, None]
    return M


def write_dam(path, label, elements, matrix, cal, charge, mdl=None,
              yields=None, e_beam=1.0, thickness=0.0, density=0.0,
              station=0, cal_orig=None, ecompress=1, pure=None):
    """Write a version -9 .dam file GeoPIXE can read.

    elements : list of row names, e.g. ['Back', 'C', 'O', 'Si', 'sum']
    matrix   : (n_elements, size) float
    cal      : (gain_keV_per_channel, offset_keV)
    mdl      : minimum detection limits at this charge, one per row
    yields   : counts per ppm per microcoulomb, one per row
    """
    M = np.asarray(matrix, dtype=float)
    n, size = M.shape
    if len(elements) != n:
        raise ValueError('%d element names for %d matrix rows'
                         % (len(elements), n))
    mdl = np.ones(n) if mdl is None else np.asarray(mdl, dtype=float)
    yields = np.ones(n) if yields is None else np.asarray(yields, dtype=float)
    cal_orig = cal_orig or cal

    w = _W()
    w.i4(-9)                    # version
    w.i4(0)                     # nda_extra: no chained matrices
    w.s(label)
    w.i4(n)
    w.f4(cal_orig[0]).f4(cal_orig[1])
    w.f4(cal[0]).f4(cal[1])
    w.f4(charge)
    w.strarr(elements)
    w.i4(ecompress)
    w.f4(mdl)
    w.f4(yields)
    w.i4(size)
    # IDL stores fltarr(size, n) with the FIRST index varying fastest, which
    # is our row-major (n, size) laid out flat. Writing M directly is correct.
    w.f4(M.ravel())
    w.i4(station)
    w.f4(density)
    w.f4(thickness)
    w.i4(0)                     # use_mu_zero = 0
    w.i4(0 if pure is None else len(pure))
    if pure is not None:
        w.f4(np.asarray(pure, dtype=float).ravel())
    w.i4(0)                     # array.on = 0
    w.f4(e_beam)

    with open(path, 'wb') as fh:
        fh.write(bytes(w.buf))
    return path


def from_session(session, path, extra_rows=True, quantify=True):
    """Build and write a DA matrix from a completed Session fit.

    The fitted component profiles ARE the pure-element unit spectra - that is
    what a component is - so the matrix follows directly from a fit already
    checked in the GUI.

    Scaling matters and is easy to get wrong. The raw pseudo-inverse recovers
    peak AREAS; a DA matrix is expected to yield CONCENTRATIONS in ppm per
    microcoulomb. Each row is therefore multiplied by that element's
    concentration-per-unit-area, taken from the same yield and efficiency
    chain Session.quantify uses. Without an efficiency curve that conversion
    does not exist, so the matrix is written in area units and the label says
    so rather than pretending otherwise.

    A 'Back' row is prepended and a 'sum' row appended because GeoPIXE expects
    them; the sum row is what its normalise-to-100 option acts on.
    """
    r = session.fit
    if r is None:
        raise RuntimeError('fit the spectrum first')
    prof = np.asarray(r.profiles, dtype=float)          # (channels, ncomp)
    size = prof.shape[0]
    rows, names = [], []

    if extra_rows:
        bk = session.background
        b = np.zeros(size)
        nb = min(size, len(bk))
        b[:nb] = bk[:nb]
        tot = b.sum()
        rows.append(b / tot if tot > 0 else b)
        names.append('Back')

    for i, nm in enumerate(r.names):
        rows.append(prof[:, i])
        names.append(nm)

    A = np.array(rows)

    # concentration per unit fitted area, element by element
    scale = np.ones(len(names))
    units = 'area'
    if quantify and session.efficiency is not None:
        try:
            conc = session.quantify()
            areas = session.areas()
            for i, nm in enumerate(names):
                Z = session.db.z.get(nm.lower())
                if Z in conc and areas.get(nm, 0) > 0:
                    scale[i] = conc[Z] * 1e4 / areas[nm]     # wt% -> ppm
            units = 'ppm'
        except Exception:
            pass

    # GeoPIXE applies a DA matrix as  matrix . spectrum / charge  and expects
    # ppm out. Our per-element scale converts a fitted area straight to ppm,
    # so the stored weights must carry a factor of charge for that division to
    # cancel. Getting this wrong is invisible - the maps look right and every
    # concentration is out by the run's charge.
    M = build_matrix(A, names, ppm_scale=scale * float(session.charge or 1.0))

    if extra_rows:
        M = np.vstack([M, M[1:].sum(axis=0)])
        names = names + ['sum']

    label = '%s [scrappyFIT %s]' % (session.path or session.label, units)
    return write_dam(path, label, names, M, session.cal,
                     session.charge or 1.0, e_beam=1.0)
