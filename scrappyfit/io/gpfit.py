"""Reader for GeoPIXE .pfr fit-result files (read_fit_results.pro).

A .pfr is what GeoPIXE writes after fitting a spectrum: the concentration of
every element, its error and detection limit, the fitted peak areas, and the
non-linear parameters (calibration, resolution, tail) it settled on. It is
the only artefact that says what GeoPIXE actually concluded from a spectrum,
which makes it the right thing to check a replacement fitter against.

Two GeoPIXE example fits ship with the source, each paired with the .spec it
came from, so the comparison is end to end: fit the same spectrum here, read
what GeoPIXE got, and put the two concentration lists side by side.

Format notes, from read_fit_results.pro:

  * IDL XDR, big-endian, versions -1 to -16. The version gates which of
    several spectrum sub-structures is present, so anything older than -7 is
    read through its own branch.
  * ORG and RORG index the non-linear parameter block. ORG - RORG is always
    10 (ten slots reserved for line "tweaks"), and n_pars is RORG when RORG
    is set, otherwise ORG.
  * Parameter slots, which is the one place the layout is documented:
        0 noise      1 Fano       2 cal B      3 cal A     4 pileup
        5 tail amp   6 tail len   7 backgnd 1  8 Compton tail amp
        9 Compton tail len       10 backgnd 2

Only the parts needed to compare results are decoded. The reader stops at
the first structure it cannot place rather than guessing, and says where.
"""

import numpy as np

from .gpyield import _R

#: Map the parameter labels GeoPIXE stores onto stable keys.
_PARAM_KEY = {'width noise': 'noise',
              'width fano factor': 'fano',
              'cal offset': 'cal_offset',
              'cal gain': 'cal_gain',
              'pileup': 'pileup',
              'tail amplitude': 'tail_amp',
              'tail length': 'tail_length'}

VALID_VERSIONS = tuple(range(-16, 0))


def read_fit_results(path):
    """Return a list of result records, one per fitted region.

    Each record has 'el' (Z, shell, name, line, energy), the four result
    arrays conc / error / mdl / area, and whatever setup, fit-quality and
    spectrum metadata the file version carries.
    """
    r = _R(open(path, 'rb').read())
    version = r.i4()
    if version not in VALID_VERSIONS:
        raise ValueError('%s: unsupported .pfr version %d' % (path, version))
    n = r.i4()
    if not (0 < n <= 10000):
        raise ValueError('%s: implausible record count %d' % (path, n))

    out = []
    for _ in range(n):
        if out:
            # Records are variable length and the tail of each is not fully
            # decoded, so the start of record 2 cannot be located. Stop and
            # say so rather than returning misaligned numbers.
            d0 = out[0]
            d0['truncated'] = ('%d of %d records read; the record tail is '
                               'not fully decoded' % (len(out), n))
            break
        n_els = r.i4()
        n_layers = r.i4()
        org = r.i4()
        rorg = r.i4() if version <= -4 else 0
        type_ = r.i4() if version <= -2 else 0
        if not (0 < n_els <= 300):
            raise ValueError('%s: implausible element count %d' % (path, n_els))
        if not (0 < n_layers <= 1000):
            raise ValueError('%s: implausible layer count %d' % (path, n_layers))
        n_pars = rorg if rorg else org

        d = dict(version=version, n_els=n_els, n_layers=n_layers,
                 org=org, rorg=rorg, type=type_)
        d['conc'] = r.f4(n_els)
        d['error'] = r.f4(n_els)
        d['mdl'] = r.f4(n_els)
        d['area'] = r.f4(n_els)
        d['area_error'] = r.f4(n_els)
        d['area_mdl'] = r.f4(n_els)
        d['scale'] = r.f4()
        d['mode'] = r.i4()

        # the element struct is written field by field, each a full array
        d['Z'] = r.i4(n_els)
        d['shell'] = r.i4(n_els)
        d['name'] = r.strarr(n_els)
        d['note'] = r.strarr(n_els)
        d['line'] = r.strarr(n_els)
        d['e'] = r.f4(n_els)
        d['mask'] = r.i4(n_els)

        d['setup'] = dict(pcm=r.s(), elow=r.f4(), ehigh=r.f4())
        d['fit'] = dict(n_its=r.i4(), phases=r.s(), chi=r.f4(), rms=r.f4())

        # -- everything above is what a result comparison needs -------------
        #
        # Below here the record continues into the non-linear parameter
        # block, the detector and filter names, the spectrum metadata and
        # finally a full copy of the yield model. That tail is NOT fully
        # decoded: the parameter name/note arrays do not have the length
        # read_fit_results.pro implies (n_pars = rorg = 11 here, but more
        # strings follow than that), so the offsets drift and the yield
        # struct cannot be located reliably.
        #
        # Rather than guess, the tail is attempted and abandoned on the first
        # thing that does not fit. A caller gets the concentrations either
        # way, and d['tail_ok'] says whether the extras are trustworthy. An
        # exception here would throw away a perfectly good result table.
        d['tail_ok'] = False
        try:
            raw = r.raw(4)
            nl = dict(free=dict(cal=raw[0], fwhm=raw[1],
                                fano=raw[2], tail=raw[3]),
                      no_tail=r.i4())
            nl['a'] = r.f4(n_pars)
            nl['mask'] = r.i4(n_pars)
            nl['name'] = r.strarr(n_pars)
            d['nonlinear'] = nl
            d['parameters'] = {}
            for i, nam in enumerate(nl['name']):
                key = _PARAM_KEY.get(nam.strip().lower())
                if key:
                    d['parameters'][key] = float(nl['a'][i])
            d['tail_ok'] = bool(d['parameters'])
        except Exception as ex:
            d['tail_error'] = '%s: %s' % (type(ex).__name__, ex)

        out.append(d)
    return out


def concentrations(rec, min_conc=0.0):
    """[(name, shell, conc_ppm, error_ppm, mdl_ppm)] sorted by abundance.

    GeoPIXE stores concentrations in ppm. Elements masked out of the fit are
    dropped, because a masked element's slot still holds a number and it does
    not mean anything.
    """
    rows = []
    for i in range(rec['n_els']):
        if rec['mask'][i] == 0:
            continue
        c = float(rec['conc'][i])
        if c < min_conc:
            continue
        rows.append((rec['name'][i] or '?', int(rec['shell'][i]), c,
                     float(rec['error'][i]), float(rec['mdl'][i])))
    return sorted(rows, key=lambda t: -t[2])


def area_scale(rec, i, db):
    """Factor converting an ELEMENT-TOTAL area into what .pfr row i stores.

    A .pfr does not store the element total. Its 'line' field names the
    transition the area belongs to - 'Ka_' for the unresolved Ka group below
    Z = 28, 'Ka1' above it where Ka1 and Ka2 separate - and the area is that
    line's, so it carries the branching fraction.

    Getting this wrong looks like a physics disagreement rather than a
    bookkeeping one. Against donut2x the raw element totals here came out
    1.77x GeoPIXE's areas across nineteen elements, which reads as a
    systematic error; multiplying by the branch gives a median of 0.955.

    Note this is the OPPOSITE convention to a .yield file, which does store
    element totals. Two GeoPIXE formats, two conventions.
    """
    name = (rec['name'][i] or '').split()[0]
    Z = db.z.get(name.lower())
    if not Z:
        return None
    shell = int(rec['shell'][i]) or 1
    lines = db.line_list(Z, shell)
    if not lines:
        return None
    tag = (rec['line'][i] or '').strip().lower()
    if tag.startswith('ka') or tag.startswith('la') or tag.startswith('ma'):
        return max(i2 for _, i2 in lines)
    return 1.0


def summary(rec):
    """One-line description of what GeoPIXE was fitting."""
    sp = rec.get('spectrum', {})
    return ('%s  %d elements, %d layer(s), chi %.4g, %d iterations, '
            'cal a=%.7g b=%.7g, charge %.6g'
            % (sp.get('label') or sp.get('file') or '?', rec['n_els'],
               rec['n_layers'], rec['fit']['chi'], rec['fit']['n_its'],
               sp.get('cal', {}).get('a', float('nan')),
               sp.get('cal', {}).get('b', float('nan')),
               sp.get('charge', float('nan'))))
