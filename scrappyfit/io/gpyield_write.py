"""Writing GeoPIXE .yield files.

What a yield file is
--------------------
For every element and every layer, the number of X-rays produced per unit
concentration per unit charge, after self-absorption and detector efficiency.
It is the bridge between a fitted peak area and a concentration, and GeoPIXE
keeps it as a file so the expensive integration is done once and reused.

Which version to write, and why not the newest
----------------------------------------------
The format has grown to version -12. From version -8 onward it embeds an IDL
`beam` structure - a nested record produced by `define(/source)` - whose exact
byte layout would have to be reproduced field for field. Getting one field
wrong there yields a file that loads without complaint and is subtly wrong,
which is the worst possible failure for a calibration file.

So this writes **version -3**, the newest version whose content is fully
determined by things we actually know. Reading `read_yield.pro`, a -3 file
skips the beam struct, the detector-array block, and the mu_zero block, and
is otherwise complete: lines, energies, intensities, yields, layers,
composition, thickness, density, beam energy and geometry all round-trip.

The one thing lost is the MAC provenance field, added at version -12, which
records which attenuation dataset was used. GeoPIXE will therefore report
'Mayer' for these files regardless of what actually produced them. Rather than
let that stand as a silent falsehood, a sidecar `.provenance.json` is written
alongside carrying the real settings.
"""

import json
import pathlib

import numpy as np

XDR_PAD = 4
NK = 20             # lines per element, matching what GeoPIXE writes


class _W:
    def __init__(self):
        self.buf = bytearray()

    def i4(self, v):
        self.buf += np.asarray(np.atleast_1d(v), dtype='>i4').tobytes()
        return self

    def f4(self, v):
        self.buf += np.asarray(np.atleast_1d(v), dtype='>f4').tobytes()
        return self

    def s(self, text, pad_to=None):
        b = str(text).encode('latin-1', errors='replace')
        if pad_to:
            b = b[:pad_to].ljust(pad_to)
        n = len(b)
        self.i4(n).i4(n)
        self.buf += b + b'\x00' * ((-n) % XDR_PAD)
        return self

    def strarr(self, items):
        for t in items:
            self.s(t)
        return self


def _layer_record(w, zlist, wfrac, thick_mgcm2, name):
    """make_layer's struct: {N, Z[32], F[32], thick, name}."""
    n = len(zlist)
    Z = np.zeros(32, dtype=int)
    F = np.zeros(32, dtype=float)
    Z[:n] = zlist
    F[:n] = wfrac
    w.i4(n)
    w.i4(Z)
    w.f4(F)
    w.f4(float(thick_mgcm2))
    w.s(name)


def write_yield(path, db, elements, layers, yields, e_beam=1.0,
                theta=135.0, phi=0.0, alpha=0.0, beta=0.0,
                z1=1, a1=1, state=1.0, title='', e_min=0.0, e_max=0.0,
                provenance=None):
    """Write a version -3 yield file.

    elements : [(Z, shell)] in the same order as the yields
    layers   : [dict(zlist=, wfrac=, thick=, density=, microns=, formula=,
                     name=)]
    yields   : array (n_layers, n_elements) - X-rays per unit concentration
               per unit charge
    """
    n = len(elements)
    nl = len(layers)
    Y = np.asarray(yields, dtype=float).reshape(nl, n)

    z2 = np.array([Z for Z, _ in elements], dtype=int)
    shell = np.array([sh for _, sh in elements], dtype=int)
    n_lines = np.zeros(n, dtype=int)
    line_idx = np.zeros((n, NK), dtype=int)
    inten = np.zeros((n, NK), dtype=float)
    energy = np.zeros((n, NK), dtype=float)
    for k, (Z, sh) in enumerate(elements):
        try:
            lines = db.line_list(Z, sh)
        except Exception:
            lines = []
        lines = lines[:NK]
        n_lines[k] = len(lines)
        for j, (e, i) in enumerate(lines):
            line_idx[k, j] = j + 1
            energy[k, j] = e
            inten[k, j] = i

    w = _W()
    w.i4(-3)                                  # version; NO ny at this version
    w.i4(n).i4(NK).i4(nl)
    w.s(title or 'scrappyFIT')
    w.i4(z2).i4(shell).i4(n_lines)
    # IDL stores intarr(nk,n) with the FIRST index fastest, matching our
    # (n, NK) row-major layout
    w.i4(line_idx.ravel())
    w.f4(inten.ravel())
    w.f4(energy.ravel())
    w.f4(Y.ravel())
    for L in layers:
        _layer_record(w, L['zlist'], L['wfrac'], L['thick'],
                      L.get('name', L.get('formula', 'layer')))
    w.i4(0)                                   # unknown
    w.f4(e_beam)
    w.f4(theta).f4(phi).f4(alpha).f4(beta)
    # version <= -2
    w.strarr([L.get('formula', '') for L in layers])
    w.i4([int(L.get('weight', 0)) for L in layers])
    w.f4([float(L['thick']) for L in layers])
    w.i4([int(L.get('microns', 0)) for L in layers])
    w.f4([float(L.get('density', 0.0)) for L in layers])
    w.i4(z1).i4(a1).f4(state)
    # version <= -3
    w.f4(e_min).f4(e_max)

    path = pathlib.Path(path)
    path.write_bytes(bytes(w.buf))

    # Provenance GeoPIXE's version -3 has nowhere to record. Written beside
    # the file rather than left implicit, because a yield file whose
    # attenuation dataset is unknown cannot be audited later.
    meta = dict(format_version=-3,
                note='GeoPIXE reports MAC dataset "Mayer" for pre-v12 yield '
                     'files regardless of what produced them. The real '
                     'settings are here.',
                e_beam_MeV=e_beam, theta_deg=theta, phi_deg=phi,
                alpha_deg=alpha, beta_deg=beta,
                elements=[[int(Z), int(sh)] for Z, sh in elements],
                layers=[{k: (list(v) if isinstance(v, (list, tuple, np.ndarray))
                             else v) for k, v in L.items()} for L in layers])
    if provenance:
        meta.update(provenance)
    path.with_suffix('.provenance.json').write_text(json.dumps(meta, indent=2))
    return str(path)


def from_session(session, path, matrix=None, thickness=None, beam_MeV=1.0,
                 theta_deg=135.0, title=''):
    """Compute yields for the session's fitted elements and write them.

    Uses the same yield model that Session.quantify() uses, so a yield file
    written here and a concentration computed in the GUI agree by
    construction rather than by coincidence.
    """
    from ..physics.layers import Layer, LayeredYieldModel

    if not session._meta:
        raise RuntimeError('fit first - the element list comes from the fit')
    zk = [(Z, sh) for Z, sh in session._meta if sh == 1]
    if not zk:
        raise RuntimeError('no K-shell components; the yield model has no L '
                           'or M ionisation cross sections')

    if matrix is None:
        matrix = session._bootstrap_matrix([Z for Z, _ in zk], session.areas())
    mz = list(matrix.keys())
    mw = [matrix[z] for z in mz]
    thick = thickness if thickness else 1e4

    lym = LayeredYieldModel(session.db)
    lay = Layer(mz, mw, thick, 'matrix')
    yy = lym.yields([lay], [Z for Z, _ in zk], E0=beam_MeV,
                    theta_deg=theta_deg, mac=session.options.mac,
                    fy=session.options.fluor_yield,
                    elam_zmax=session.options.fluor_elam_zmax, n_steps=900)

    eff = session.efficiency
    row = []
    for Z, sh in zk:
        v = yy.get(Z, 0.0)
        v = 0.0 if not np.isfinite(v) else float(v)
        if eff is not None:
            v *= eff(session.db.line_energy(Z))
        row.append(v)

    total = sum(mw) or 1.0
    formula = ''.join('(%s)%.4g' % (session.db.sym[z], w / total)
                      for z, w in zip(mz, mw))
    layer = dict(zlist=mz, wfrac=[w / total for w in mw], thick=thick,
                 density=0.0, microns=0, formula=formula, name='matrix')

    return write_yield(
        path, session.db, zk, [layer], np.array([row]),
        e_beam=beam_MeV, theta=theta_deg,
        title=title or (session.label or 'scrappyFIT'),
        e_min=session.options.e_low, e_max=session.options.e_high,
        provenance=dict(mac_dataset=session.options.mac,
                        fluorescence_yield=session.options.fluor_yield,
                        lines_file=session.options.lines_file,
                        efficiency=(eff.source if eff else None),
                        source_spectrum=session.path))
