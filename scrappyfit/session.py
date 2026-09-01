"""The analysis session: one object holding a dataset and everything derived
from it.

Why a facade
------------
The physics lives in small modules that each do one thing, and it should stay
that way. But a GUI - or a script, or an agent driving this from a terminal -
should not have to know that fitting needs a background computed first, that
the background needs a statistics-sensitive filter before it, that maps need
the event list rather than the summed spectrum, or which of five mass
attenuation datasets is valid below 1 keV. Those are decisions with right
answers, and encoding them once here means every caller gets them right.

So: Session owns the data, exposes verbs (load, map, mask, fit, export), and
keeps the results. The GUI is then a thin thing that calls verbs and draws
what comes back, which is what makes it replaceable.

Everything is lazy. Loading a 68 MB list-mode file fits nothing; asking for a
fit computes and caches one; changing an option invalidates only what depends
on it.
"""

import json
import os
import pathlib

import numpy as np

from . import config
from .analysis import damaps as _damaps
from .analysis import maps as _maps
from .fitting import fit as _fit
from .fitting.background import snip
from .io import gpda as _gpda
from .io import lmf as _lmf
from .physics import lfix
from .physics.gpdb import Database

SHELL_SUFFIX = {1: '', 2: 'L', 3: 'M'}

# Header strings that describe the instrument rather than the sample. OMDAQ
# writes these alongside the operator's sample description, and picking the
# wrong one leaves every window titled "Energy Range - 6.6 keV".
_INSTRUMENT_HINTS = ('energy range', 'coarse gain', 'ibmal', 'rbs -',
                     'null', 'amptek', 'stim', 'ibic', 'annular')


def _sample_name(strings):
    """The operator's description of the sample, from an LMF header."""
    best = ''
    for s in strings:
        t = s.strip()
        if len(t) < 8 or t[0].isdigit():
            continue
        if any(h in t.lower() for h in _INSTRUMENT_HINTS):
            continue
        if len(t) > len(best):
            best = t
    return best


class FitOptions:
    """Everything that changes a fit result, in one place so it can be saved
    beside the numbers. A result without its options is not reproducible."""

    def __init__(self, **kw):
        self.e_low = 0.20
        self.e_high = 6.50
        self.noise = 32.74
        self.fano = 28.05
        self.tail_amp = 0.08
        self.tail_len = 1.0
        self.snip_passes = 3
        self.mac = 'mixed'
        self.fluor_yield = 'krause'
        self.fluor_elam_zmax = 10
        self.lines_file = 'xray_lines_rebuilt.txt'
        # detector response terms GeoPIXE has no equivalent for
        self.use_escape_step = True
        self.use_contact_step = True
        self.use_window_step = False
        # 'strict' forbids negative areas. True reports them instead, which is
        # informative when deciding whether an element is present at all.
        self.nonneg = 'strict'
        self.refine = ('cal', 'width', 'tail')
        self.fix_mn_lb = True
        self.__dict__.update(kw)

    def refine_groups(self):
        g = list(self.refine)
        for flag, name in ((self.use_escape_step, 'shelf'),
                           (self.use_contact_step, 'contact'),
                           (self.use_window_step, 'window')):
            if flag and name not in g:
                g.append(name)
        return tuple(g)

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


class Session:
    """One dataset and its analysis."""

    def __init__(self, database=None, options=None):
        self.options = options or FitOptions()
        self._db_root = database
        self._db = None
        self.reset_data()

    # -- database -------------------------------------------------------

    @property
    def db(self):
        if self._db is None:
            self._db = Database(self._db_root,
                                lines_file=self.options.lines_file)
            if self.options.fix_mn_lb:
                lfix.apply(self._db, verbose=False)
        return self._db

    def set_lines_file(self, name):
        self.options.lines_file = name
        self._db = None
        self.invalidate()

    # -- state ----------------------------------------------------------

    def reset_data(self):
        self.path = None
        self.label = ''
        self.events = None          # (x, y, energy), list mode only
        self.spectrum = None        # the ACTIVE spectrum (may be masked)
        self.full_spectrum = None   # whole field, kept when a mask is applied
        self.cal = (0.0017, -0.375)
        self.charge = None
        self.adc = 0
        self.mask = None
        self.mask_name = ''
        self.dam = None
        self.invalidate()

    def invalidate(self):
        self._bk = None
        self._fit = None
        self._meta = []

    # -- loading --------------------------------------------------------

    def load(self, path, adc=0):
        """Open any format we understand.

        .lmf   OMDAQ list mode - carries event POSITIONS, so maps and region
               masking become available
        .dam   GeoPIXE Dynamic Analysis matrix; its calibration is adopted and
               the source list-mode file pulled in if it can be found
        .spec  GeoPIXE text export (SPEC/DATA records)
        other  two columns, channel and counts
        """
        p = pathlib.Path(path)
        self.reset_data()
        self.path = str(p)
        self.label = p.name
        ext = p.suffix.lower()
        if ext == '.lmf':
            self._load_lmf(p, adc)
        elif ext == '.dam':
            self._load_dam(p)
        elif ext == '.spec':
            self._load_spec_text(p)
        else:
            self._load_columns(p)
        self.full_spectrum = (None if self.spectrum is None
                              else self.spectrum.copy())
        self.invalidate()
        return self

    def _load_lmf(self, p, adc):
        x, y, e, a, q = _lmf.read(str(p))
        m = a == adc
        self.adc = adc
        self.events = (x[m].astype(np.int32), y[m].astype(np.int32),
                       e[m].astype(np.int32))
        self.spectrum = np.bincount(e[m], minlength=4096).astype(float)
        self.charge = float(q.sum())
        self.label = _sample_name(_lmf.ascii_header(str(p))) or p.name

    def _load_dam(self, p):
        self.dam = _gpda.read_dam(str(p))[0]
        self.cal = self.dam['cal']
        self.charge = self.dam['charge']
        self.label = os.path.basename(self.dam['label']) or p.name
        src = pathlib.Path(self.dam['label'])
        for cand in (src, p.with_suffix('.lmf'), p.parent / src.name):
            if cand.suffix.lower() == '.lmf' and cand.exists():
                self._load_lmf(cand, 0)
                break

    def _load_spec_text(self, p):
        cal, data = None, []
        for line in open(p, errors='ignore'):
            t = line.split()
            if t and t[0] == 'SPEC' and len(t) >= 3:
                cal = (float(t[2]), float(t[1]))
            elif t and t[0] == 'DATA' and len(t) >= 3:
                data.append(float(t[2]))
        if not data:
            raise ValueError('%s holds no SPEC/DATA records. GeoPIXE also '
                             'writes a binary .spec, which this does not read '
                             'yet.' % p.name)
        self.spectrum = np.array(data)
        if cal:
            self.cal = cal

    def _load_columns(self, p):
        d = np.loadtxt(p)
        self.spectrum = d[:, 1] if d.ndim == 2 else d

    # -- energy axis ----------------------------------------------------

    @property
    def energy(self):
        a, b = self.cal
        return a * np.arange(len(self.spectrum)) + b

    def set_calibration(self, gain, offset):
        self.cal = (float(gain), float(offset))
        self.invalidate()

    # -- maps and masking -----------------------------------------------

    def element_map(self, which, width=0.075, binning=4, subtract=True):
        """Window map for one line. Needs list-mode data.

        Flanking subtraction fails where a weak line sits between two strong
        ones - aluminium between magnesium and silicon is the standard case,
        and it goes negative. Use da_maps() where a DA matrix exists.
        """
        if self.events is None:
            raise RuntimeError('maps need a list-mode (.lmf) file')
        if isinstance(which, str):
            E0 = self.db.line_energy(self.db.z[which.lower()])
        else:
            E0 = float(which)
        a, b = self.cal
        lo, hi = _maps.window(E0 - width, E0 + width, a, b)
        d = hi - lo + 1
        bg = ((lo - 2 * d, lo - d), (hi + d, hi + 2 * d)) if subtract else None
        x, y, e = self.events
        return _maps.smooth(
            _maps.element_map(x, y, e, lo, hi, bg=bg, binning=binning), 1.0)

    def da_maps(self, binning=4, elements=None):
        """Concentration maps (wt%) through a loaded GeoPIXE DA matrix, which
        resolves peak overlaps properly where a window map cannot."""
        if self.dam is None:
            raise RuntimeError('load a .dam file first')
        if self.events is None:
            raise RuntimeError('DA maps need the list-mode file')
        x, y, e = self.events
        return _damaps.da_maps(self.dam, x, y, e, binning=binning,
                               elements=elements)

    def set_mask(self, mask, name=''):
        """Restrict the active spectrum to a region of the raster.

        This is the point of mapping: the bulk spectrum of a heterogeneous
        sample describes nowhere in it.
        """
        if self.events is None:
            raise RuntimeError('masking needs a list-mode file')
        x, y, e = self.events
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != 256:
            f = 256 // m.shape[0]
            m = np.repeat(np.repeat(m, f, 0), f, 1)
        sel = m[y, x]
        self.mask, self.mask_name = m, name
        self.spectrum = np.bincount(
            e[sel], minlength=len(self.full_spectrum)).astype(float)
        self.invalidate()
        return int(sel.sum())

    def clear_mask(self):
        self.mask, self.mask_name = None, ''
        if self.full_spectrum is not None:
            self.spectrum = self.full_spectrum.copy()
        self.invalidate()

    # -- fitting --------------------------------------------------------

    @property
    def background(self):
        if self._bk is None:
            o = self.options
            a, b = self.cal
            self._bk = snip(self.spectrum, a, b, o.e_low, o.e_high,
                            passes=o.snip_passes, use_low_stats=True)
        return self._bk

    def build_components(self, spec):
        """spec: element symbols, or (Z, shell) pairs. Shell 1 = K, 2 = L,
        3 = M. 3d-metal L components get the broadened band shape, because
        for those elements the M shell IS the valence band and the emission
        is a band rather than a set of lines."""
        out, meta = [], []
        for item in spec:
            if isinstance(item, (tuple, list)):
                Z, sh = item
            else:
                Z, sh = self.db.z[str(item).lower()], 1
            lines = self.db.line_list(Z, sh)
            if not lines:
                continue
            c = _fit.Component(self.db.sym[Z] + SHELL_SUFFIX[sh], lines)
            if sh == 2 and 21 <= Z <= 30:
                c.tail_amp_fn = lambda E: 0.70
                c.tail_len_fn = lambda E: 3.5
            else:
                c.tail_amp_fn = lambda E, a=self.options.tail_amp: a
                c.tail_len_fn = lambda E, l=self.options.tail_len: l
            out.append(c)
            meta.append((Z, sh))
        self._meta = meta
        return out

    def run_fit(self, elements):
        o = self.options
        a, b = self.cal
        comps = self.build_components(elements)
        if not comps:
            raise ValueError('no fittable components in %r' % (elements,))
        self._fit = _fit.fit_spectrum(
            self.spectrum, a, b, comps, o.e_low, o.e_high,
            noise=o.noise, fano=o.fano, tail_amp=o.tail_amp,
            tail_len=o.tail_len, background=self.background,
            refine=o.refine_groups(), nonneg=o.nonneg)
        return self._fit

    @property
    def fit(self):
        return self._fit

    def areas(self):
        return {} if self._fit is None else dict(
            zip(self._fit.names, self._fit.areas))

    def errors(self):
        return {} if self._fit is None else dict(
            zip(self._fit.names, self._fit.errors))

    # -- export ---------------------------------------------------------

    def export(self, folder, prefix=None):
        """Write everything about this analysis into one folder.

        Spectrum, background, model, every component separately, the residual,
        areas with uncertainties, the mask, and the complete option set. The
        folder is meant to be self-describing - someone opening it later
        should be able to see what was done, not only what came out.
        """
        out = pathlib.Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        pre = prefix or (pathlib.Path(self.label).stem or 'spectrum')
        if self.mask_name:
            pre += '_' + self.mask_name.replace(' ', '_')
        written = []

        def w(name, arr, header=''):
            f = out / ('%s_%s' % (pre, name))
            np.savetxt(f, arr, header=header, fmt='%.6g')
            written.append(f.name)

        E = self.energy
        w('spectrum.txt',
          np.c_[np.arange(len(self.spectrum)), E, self.spectrum],
          'channel  energy_keV  counts')
        if self._bk is not None:
            w('background.txt', np.c_[E, self._bk], 'energy_keV  background')

        r = self._fit
        if r is not None:
            n = min(len(E), len(r.model))
            w('fit_model.txt', np.c_[E[:n], r.model[:n]], 'energy_keV  model')
            sg = np.sqrt(np.maximum(r.model[:n], 1.0))
            w('residual.txt',
              np.c_[E[:n], (self.spectrum[:n] - r.model[:n]) / sg],
              'energy_keV  residual_sigma')
            cols, hdr = [E[:n]], ['energy_keV']
            for i, nm in enumerate(r.names):
                cols.append(r.profiles[:n, i] * r.areas[i])
                hdr.append(nm)
            w('components.txt', np.column_stack(cols), '  '.join(hdr))
            rows = ['# component      area        error   rel_percent']
            for nm, ar, er in zip(r.names, r.areas, r.errors):
                rel = 100 * er / ar if ar else float('nan')
                rows.append('%-10s %14.4f %12.4f %10.2f' % (nm, ar, er, rel))
            (out / ('%s_areas.txt' % pre)).write_text('\n'.join(rows) + '\n')
            written.append('%s_areas.txt' % pre)

        import scrappyfit
        meta = dict(
            source=self.path, label=self.label, adc=self.adc,
            calibration=dict(gain=self.cal[0], offset=self.cal[1]),
            charge=self.charge, mask=self.mask_name or None,
            chi2_reduced=(r.reduced_chi2 if r is not None else None),
            database=str(config.database_path(required=False)),
            options=self.options.to_dict(),
            scrappyfit_version=scrappyfit.__version__)
        (out / ('%s_analysis.json' % pre)).write_text(json.dumps(meta, indent=2))
        written.append('%s_analysis.json' % pre)

        if self.mask is not None:
            np.save(out / ('%s_mask.npy' % pre), self.mask)
            written.append('%s_mask.npy' % pre)
        return written
