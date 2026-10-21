"""Writing GeoPIXE .yield files.

What a yield file is
--------------------
For every element and every layer, the number of MAJOR-LINE X-rays produced
per ppm per uC per msr, after self-absorption but BEFORE detector efficiency.
It is the bridge between a fitted peak area and a concentration, and GeoPIXE
keeps it as a file so the expensive integration is done once and reused.

How GeoPIXE reads one (pixe_fit.pro, array_yield.pro, geo_yield2.pro)
---------------------------------------------------------------------
* ONE column only: `Yyield = (*peaks).yield[*, (*peaks).unknown-1]`. Every
  element is quantified as if it sat in the UNKNOWN layer, whose index is
  1-based. Writing 0 there indexes column -1.
* `counts_per_ppm_uc = omega[msr] * eff(E_major) * T * Y` and
  `conc = area / (counts_per_ppm_uc * charge)`, with `area` the amplitude of
  line 0. So Y is the major line alone, and GeoPIXE applies the detector
  efficiency itself - a Y that already carries it is double-counted.
* Line 0 is the major line with intensity 1; the others are relative to it
  (geo_yield2: `major_yield = total_yield[0]`, `rel_int = total_yield /
  major_yield`), absorption-weighted, in descending order. A sum-normalised
  list makes GeoPIXE over-report by 1/I0 (2.1x for I L).

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


def _atom_fractions(db, zlist, wfrac):
    """Weight fractions to atom fractions: divide by atomic mass, renormalise."""
    n = [w / db.A[z] for z, w in zip(zlist, wfrac)]
    t = sum(n)
    return [x / t for x in n] if t > 0 else list(wfrac)


def _layer_afrac(L):
    """The ATOM fractions to write into F: 'afrac', or the legacy 'wfrac'
    key, which callers have always filled with atom fractions despite its
    name (projects/perovskite-corrosion-pixe/runs/yield_v2_outputs.py)."""
    a = [float(x) for x in L.get('afrac', L.get('wfrac'))]
    t = sum(a)
    return [x / t for x in a] if t > 0 else a


def _major_first(lines):
    """Lines for the file: strongest first, intensities relative to it.

    GeoPIXE treats line 0 as the major line, the one Y refers to. The
    database's lists are sum-normalised and in table order (Ka_ then Kb_,
    La1/La2 before the Lb lines), which is usually but not always strongest
    first, so sort before dividing.
    """
    lines = sorted(lines, key=lambda t: -t[1])[:NK]
    if not lines or lines[0][1] <= 0:
        return []
    i0 = float(lines[0][1])
    return [(float(e), float(i) / i0) for e, i in lines]


def _relative_to_line0(lines):
    """A caller's table, kept in its order: line 0 is the major line by
    contract, so only the scaling changes. Absorption can make a minor line
    exceed it across an edge; GeoPIXE allows that, and so does this."""
    lines = list(lines)[:NK]
    if not lines or lines[0][1] <= 0:
        return []
    i0 = float(lines[0][1])
    return [(float(e), float(i) / i0) for e, i in lines]


def _layer_record(w, zlist, afrac, thick_mgcm2, name):
    """make_layer's struct: {N, Z[32], F[32], thick, name}.

    F is the ATOM fraction, as decode_formula.pro returns it - SiO2 is
    O 0.667, Si 0.333. Writing weight fractions here gives a file that loads
    without complaint and reads back with the wrong matrix.
    """
    n = len(zlist)
    Z = np.zeros(32, dtype=int)
    F = np.zeros(32, dtype=float)
    Z[:n] = zlist
    F[:n] = afrac
    w.i4(n)
    w.i4(Z)
    w.f4(F)
    w.f4(float(thick_mgcm2))
    w.s(name)


def write_yield(path, db, elements, layers, yields, e_beam=1.0,
                theta=135.0, phi=0.0, alpha=0.0, beta=0.0,
                z1=1, a1=1, state=1.0, title='', e_min=0.0, e_max=0.0,
                provenance=None, unknown=1, lines=None):
    """Write a version -3 yield file.

    elements : [(Z, shell)] in the same order as the yields
    layers   : [dict(zlist=, afrac=, thick=, density=, microns=, formula=,
                     name=)]
               'afrac' are ATOM fractions, written straight into GeoPIXE's
               layer F array. 'wfrac' is accepted in its place for old
               callers and must ALSO hold atom fractions - it is not
               converted. Weight fractions (what Layer() takes) go through
               _atom_fractions(db, zlist, w) first. A 'formula' should use
               atom proportions too: 'weight' defaults to 0, and
               decode_formula.pro then reads its numbers as atom counts.
    yields   : array (n_layers, n_elements) - MAJOR-line yields in GeoPIXE
               units: major-line photons per ppm (in that layer) per uC per
               msr, absorption included, detector efficiency EXCLUDED.
               LayeredYieldModel.yields(per_major_line=True) x
               physics.geometry.YIELD_TO_GEOPIXE is this. A total-shell
               yield T goes in as T x I_major / sum(I), the I being the
               intensities written for that element; Y x sum(intensity)
               read back is then T.
    unknown  : the UNKNOWN layer, 1-BASED as GeoPIXE counts it. GeoPIXE
               quantifies every element from this layer's column alone, so
               a multilayer file is right only for elements that live in
               it: write one file per UNKNOWN layer.
    lines    : optional {(Z, shell): [(keV, intensity), ...]} - per-element
               line tables, e.g. absorption-weighted relative intensities.
               Line 0 MUST be the major line, the one `yields` refers to;
               the table is kept in its order and divided by line 0's
               intensity. Elements not in it take db.line_list(Z, shell),
               sorted strongest first and divided by the strongest - those
               are emitted, not absorption-weighted, intensities.
    """
    n = len(elements)
    nl = len(layers)
    Y = np.asarray(yields, dtype=float).reshape(nl, n)
    unknown = int(unknown)
    if not 1 <= unknown <= nl:
        raise ValueError('unknown=%d: the UNKNOWN layer is 1-based and must '
                         'be one of the %d layer(s)' % (unknown, nl))
    lines = lines or {}

    z2 = np.array([Z for Z, _ in elements], dtype=int)
    shell = np.array([sh for _, sh in elements], dtype=int)
    n_lines = np.zeros(n, dtype=int)
    line_idx = np.zeros((n, NK), dtype=int)
    inten = np.zeros((n, NK), dtype=float)
    energy = np.zeros((n, NK), dtype=float)
    for k, (Z, sh) in enumerate(elements):
        if (Z, sh) in lines:
            ll = _relative_to_line0(lines[(Z, sh)])
        else:
            try:
                ll = _major_first(db.line_list(Z, sh))
            except Exception:
                ll = []
        n_lines[k] = len(ll)
        for j, (e, i) in enumerate(ll):
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
        _layer_record(w, L['zlist'], _layer_afrac(L), L['thick'],
                      L.get('name', L.get('formula', 'layer')))
    w.i4(unknown)                             # 1-based, as GeoPIXE reads it
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
    meta = dict(format_version=-3, unknown_layer_1based=unknown,
                yield_units='major-line photons per ppm per uC per msr; '
                            'absorption included, detector efficiency '
                            'excluded (GeoPIXE applies it)',
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


def _absorbed_line_tables(lym, layers, zk, db, kw):
    """{(Z, 1): [(keV, rel)]}: emitted intensities times each line's yield
    (transmission) relative to the major line's, strongest-emitted first.

    One yields() call per line rank, attenuating at that rank's energy; the
    ionisation and fluorescence factors cancel in the ratio. A line whose
    attenuation is unavailable (MAC out of range) keeps its emitted ratio.
    """
    emitted = {Z: sorted(db.line_list(Z, sh), key=lambda t: -t[1])[:NK]
               for Z, sh in zk}
    nmax = max((len(v) for v in emitted.values()), default=0)
    per_rank = []
    for j in range(nmax):
        el = {Z: v[j][0] for Z, v in emitted.items() if len(v) > j}
        per_rank.append(lym.yields(layers, list(el), per_major_line=False,
                                   e_lines=el, **kw))
    out = {}
    for Z, sh in zk:
        ll = emitted[Z]
        if not ll:
            continue
        y0 = per_rank[0].get(Z, float('nan'))
        tab = []
        for j, (e, i) in enumerate(ll):
            yj = per_rank[j].get(Z, float('nan'))
            t = yj / y0 if (np.isfinite(yj) and np.isfinite(y0)
                            and y0 > 0) else 1.0
            tab.append((e, i / ll[0][1] * t))
        out[(Z, sh)] = tab
    return out


def from_session(session, path, matrix=None, thickness=None, beam_MeV=1.0,
                 theta_deg=135.0, title=''):
    """Compute yields for the session's fitted elements and write them.

    Uses the same yield model that Session.quantify() uses, so a yield file
    written here and a concentration computed in the GUI agree by
    construction rather than by coincidence.

    Units: the yields are written as GeoPIXE stores them - major-line
    photons per ppm per uC per msr (x YIELD_TO_GEOPIXE), WITHOUT the
    detector efficiency, which GeoPIXE multiplies in itself
    (array_yield.pro). Before 12 Oct 2026 this wrote the model's raw output
    times efficiency: ~5e23 too small in scale and efficiency counted twice
    once GeoPIXE applied its own. Regenerate any file written before then.
    The efficiency curve's source is still recorded in the provenance.

    Lines: major first, each line's intensity weighted by its own
    transmission out of the matrix relative to the major line's, as
    geo_yield2.pro does.
    """
    from ..physics.geometry import YIELD_TO_GEOPIXE
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
    kw = dict(E0=beam_MeV, theta_deg=theta_deg, mac=session.options.mac,
              fy=session.options.fluor_yield,
              elam_zmax=session.options.fluor_elam_zmax, n_steps=900)
    yy = lym.yields([lay], [Z for Z, _ in zk], **kw)

    row = []
    for Z, sh in zk:
        v = yy.get(Z, 0.0)
        v = 0.0 if not np.isfinite(v) else float(v) * YIELD_TO_GEOPIXE
        row.append(v)
    tables = _absorbed_line_tables(lym, [lay], zk, session.db, kw)
    eff = session.efficiency

    # The matrix is in weight fractions; the file wants atom fractions in
    # both F and the formula (weight=0), or a hydrogen- or Pb-bearing matrix
    # reads back tens of percent off.
    total = sum(mw) or 1.0
    wfrac = [w / total for w in mw]
    afrac = _atom_fractions(session.db, mz, wfrac)
    formula = ''.join('(%s)%.4g' % (session.db.sym[z], a)
                      for z, a in zip(mz, afrac))
    layer = dict(zlist=mz, afrac=afrac, mass_frac=wfrac, thick=thick,
                 density=0.0, microns=0, weight=0, formula=formula,
                 name='matrix')

    return write_yield(
        path, session.db, zk, [layer], np.array([row]), lines=tables,
        e_beam=beam_MeV, theta=theta_deg,
        title=title or (session.label or 'scrappyFIT'),
        e_min=session.options.e_low, e_max=session.options.e_high,
        provenance=dict(mac_dataset=session.options.mac,
                        fluorescence_yield=session.options.fluor_yield,
                        lines_file=session.options.lines_file,
                        efficiency=(eff.source if eff else None),
                        efficiency_in_yields=False,
                        source_spectrum=session.path))
