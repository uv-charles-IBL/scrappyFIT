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
from .physics.efficiency import Efficiency, from_geopixe
from .physics.escape import EscapeModel
from .physics.gpdb import Database
from .physics.geometry import Geometry
from .physics.layers import Layer, LayeredYieldModel

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
        # Sum-peak pile-up. Two photons inside the shaping time are recorded
        # as one event at the sum of their energies. It is small - a few
        # tenths of a percent on a bright spectrum - but it lands in empty
        # regions where nothing else is, so it is exactly where a fit will
        # invent an element. On quartz the Si+O sum at 2.24 keV is 488 counts
        # and gets assigned to mercury or niobium if it is not modelled.
        self.use_pileup = True
        self.sum_deficit = 0.1
        """Fraction of sum-peak amplitude lost to finite time resolution.

        sum_peaks.pro's default. Two photons close enough in time to sum are
        sometimes close enough to be rejected instead, so the sum peak is
        always a little smaller than the raw product of the parent rates.
        """
        self.shaping_time_us = 1.0
        """Amplifier shaping time, which sets how much pile-up is possible.

        Pile-up probability is roughly (count rate) x (shaping time), so this
        is what bounds the pile-up component. Too generous a value lets it
        soak up model error; too mean a one forces real pile-up into the
        elements. 1 us is typical for a digital pulse processor on an SDD.
        """
        self.pileup_headroom = 5.0
        """How far above the first-order estimate the fit may still go.

        rate x tau is good to a factor of a few, not to a factor of a
        thousand, so the cap is deliberately loose. It exists to reject the
        1300x excess seen on 287424, not to pin pile-up to three figures.
        """
        # Silicon escape peaks, tied to their parent component rather than
        # fitted freely. Every line above the Si K edge produces one, so this
        # is not an exotic correction - it is part of the response of any
        # silicon detector, and a spectrum with calcium in it has a calcium
        # escape peak at 1.95 keV whether or not anyone modelled it.
        self.use_escape_peaks = True
        self.crystal_Z = 14
        self.crystal_thick_um = 500.0
        self.escape_gamma = 1.0
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
        self.efficiency = None
        self.geometry = Geometry()
        self.detector = None        # a GeoPIXE .detector model, if loaded
        self._tail_fns = None       # its energy-dependent tail functions
        self._escape = None
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

    @property
    def escape_model(self):
        """Escape model for the current crystal, built lazily.

        Thickness converts to areal density with silicon at 2.33 g/cm3, which
        is what EscapeModel expects.
        """
        o = self.options
        if not o.use_escape_peaks:
            return None
        key = (o.crystal_Z, o.crystal_thick_um, o.escape_gamma, o.mac)
        if self._escape is None or getattr(self, '_escape_key', None) != key:
            rho = 2.33 if o.crystal_Z == 14 else 5.32
            areal = o.crystal_thick_um * 1e-4 * rho * 1000.0     # mg/cm2
            self._escape = EscapeModel(self.db, o.crystal_Z, areal,
                                       gamma=o.escape_gamma, mac_dataset=o.mac)
            self._escape_key = key
        return self._escape

    def set_lines_file(self, name):
        self.options.lines_file = name
        self._db = None
        self._escape = None
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

    # -- detector efficiency --------------------------------------------

    def load_efficiency(self, path):
        """A fitted area is not a concentration until this is known. At
        carbon Ka a C1 window transmits about 3%, so the same peak area means
        thirty times more carbon than it would silicon."""
        self.efficiency = from_geopixe(path)
        self._conc = None
        return self.efficiency

    def builtin_efficiencies(self):
        import glob
        import pathlib
        d = pathlib.Path(__file__).parent / 'resources' / 'efficiency'
        return sorted(glob.glob(str(d / '*.txt')))

    def invalidate(self):
        self._bk = None
        self._fit = None
        self._conc = None
        self._meta = []

    # -- loading --------------------------------------------------------

    def load_spectrum(self, counts, cal=None, label=''):
        """Take a spectrum straight from an array.

        For data that arrived through a reader of its own rather than through
        load() - a GeoPIXE binary .spec, a detector API, a simulation. No
        event positions, so maps and masking stay unavailable.
        """
        a = np.asarray(counts, dtype=float).ravel()
        if a.size == 0:
            raise ValueError('empty spectrum')
        self.path = None
        self.events = None
        self.dam = None
        self.spectrum = a
        self.full_spectrum = a.copy()
        self.mask = None
        if cal:
            self.cal = (float(cal[0]), float(cal[1]))
        self.label = label or 'spectrum'
        self.invalidate()
        return self

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
            self._load_spec_any(p)
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

    def load_detector(self, path):
        """Load a GeoPIXE .detector file and adopt its physics.

        This sets four things that a fit gets badly wrong without them:

          * the crystal, so escape peaks are the right element. Canberra-34
            is germanium: escape at 9.886 keV, not silicon's 1.740, and a
            gamma_factor of 0.24 against silicon's 0.022.
          * the energy-DEPENDENT tail amplitude and length. The flat values
            used otherwise are the single largest lineshape error here.
          * the resolution, if the file carries w0 and w1.
          * the absorbers, for efficiency.
        """
        from .io import gpdetector as _gd
        from .physics import dettail as _dt

        det = _gd.read_detector(str(path))
        if not det.get('crystal'):
            raise ValueError('%s: no crystal layer found' % path)
        self.detector = det
        cry = det['crystal']
        self.options.crystal_Z = cry['Z'][0]
        dens = det.get('density') or 5.32
        self.options.crystal_thick_um = cry['thick'] / dens * 10.0
        if det.get('gamma_factor'):
            self.options.escape_gamma = 1.0     # the file's own prefactor
        if det.get('tail'):
            self._tail_fns = _dt.tail_functions(det, self.db,
                                                mac=self.options.mac)
        w0, w1 = det.get('w0'), det.get('w1')
        if w0 and w1 and self.cal and self.cal[0]:
            # GeoPIXE: FWHM_keV^2 = w0 + w1 E.  Here: FWHM_ch^2 = noise^2 +
            # fano^2 (E - e_low), and FWHM_keV = FWHM_ch x cal_a. Equating
            # them gives fano and noise directly.
            a = float(self.cal[0])
            fano = (w1 ** 0.5) / a
            noise2 = w0 / (a * a) + fano * fano * self.options.e_low
            if noise2 > 0:
                self.options.noise = noise2 ** 0.5
                self.options.fano = fano
        self._escape = None
        self.invalidate()
        return det

    def load_filter(self, path):
        """Add a GeoPIXE .filter to the absorbers in front of the crystal."""
        from .io import gpdetector as _gd
        if self.detector is None:
            raise RuntimeError('load a .detector first')
        self.detector.setdefault('external', [])
        self.detector['external'] += _gd.read_filter(str(path))
        self.invalidate()
        return self.detector['external']

    def load_dam(self, path, keep_data=None):
        """Attach a GeoPIXE Dynamic Analysis matrix.

        A DA matrix is built once from a good fit and then applied to other
        runs from the same session - that is the whole point of it, and it is
        why this is separate from load(). Two situations:

          keep_data=True   attach the matrix to whatever is already loaded.
                           Use this to apply one run's matrix to another
                           run's events, which is the normal case.
          keep_data=False  also load the list-mode file the matrix was built
                           from, if it can be found beside it.

        The default follows what is already open: if events are loaded, the
        matrix is attached to them and the calibration is left alone, because
        overwriting a working calibration with one from another file is a
        silent way to get wrong energies.
        """
        p = pathlib.Path(path)
        if keep_data is None:
            keep_data = self.events is not None
        self.dam = _gpda.read_dam(str(p))[0]
        if not keep_data:
            self.cal = self.dam['cal']
            self.charge = self.dam['charge']
            self.label = os.path.basename(self.dam['label']) or p.name
            src = pathlib.Path(self.dam['label'])
            for cand in (src, p.with_suffix('.lmf'), p.parent / src.name):
                if cand.suffix.lower() == '.lmf' and cand.exists():
                    self._load_lmf(cand, 0)
                    break
        self.invalidate()
        return self

    def dam_elements(self):
        """Element names the loaded matrix can project, in its own order."""
        if self.dam is None:
            return []
        return list(self.dam.get('el') or self.dam.get('names') or [])

    def _load_dam(self, p):
        self.load_dam(p, keep_data=False)

    def _load_spec_any(self, p):
        """A .spec is either GeoPIXE's text export or its binary XDR form.

        The text one starts with printable SPEC/DATA records; the binary one
        starts with a negative int32 version. Try text first and fall back,
        so a user opening a .spec does not have to know which they have.

        A binary .spec does not carry a usable energy calibration - it lives
        in the .pfr written beside it - so that is picked up too when present,
        and verified against known line energies rather than trusted.
        """
        try:
            self._load_spec_text(p)
            return
        except Exception as text_error:
            pass
        from .io import gpspec as _gs
        try:
            spec, info = _gs.read_spec(str(p))
        except Exception as ex:
            raise ValueError(
                '%s reads as neither a text .spec (%s) nor a binary one (%s)'
                % (p.name, text_error, ex))
        self.spectrum = np.asarray(spec, float)
        self.full_spectrum = self.spectrum.copy()
        self.events = None
        self.label = p.stem
        for cand in (p.with_name(p.stem + '-REF.pfr'),
                     p.with_suffix('.pfr')):
            if cand.exists():
                cal = _gs.calibration_from_pfr(str(cand))
                if cal:
                    self.cal = cal
                    self.label = '%s (cal from %s)' % (p.stem, cand.name)
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
            c = _fit.Component(self.db.sym[Z] + SHELL_SUFFIX[sh], lines,
                               escape=self.escape_model)
            if sh == 2 and 21 <= Z <= 30:
                c.tail_amp_fn = lambda E: 0.70
                c.tail_len_fn = lambda E: 3.5
            elif self._tail_fns is not None:
                # Real detector model: amplitude and length both vary with
                # energy, as tail_amplitude.pro and tail_length.pro do.
                c.tail_amp_fn, c.tail_len_fn = self._tail_fns
            else:
                # No detector model loaded, so a flat tail is all there is.
                # It is a poor approximation over a wide energy range - see
                # physics/dettail - but it is honest about being one.
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
        sump = None
        if o.use_pileup:
            sump = _fit.SumPeakComponent(
                comps, sum_deficit=o.sum_deficit, e_high=o.e_high,
                max_area=self.pileup_cap())
            comps = comps + [sump]

        def go():
            return _fit.fit_spectrum(
                self.spectrum, a, b, comps, o.e_low, o.e_high,
                noise=o.noise, fano=o.fano, tail_amp=o.tail_amp,
                tail_len=o.tail_len, background=self.background,
                refine=o.refine_groups(), nonneg=o.nonneg)

        res = go()
        if sump is not None:
            # Sum-line intensities are products of the parent AREAS, which are
            # not known until the elements have been fitted once. sum_peaks.pro
            # has the same dependency and resolves it the same way: fit,
            # rebuild the sum lines from the result, fit again. Two passes is
            # enough - the sum peaks are a fraction of a percent, so their
            # effect on the parent areas that generated them is negligible.
            for _ in range(2):
                if not sump.update(dict(zip(res.names, res.areas))):
                    break
                res = go()
        self._fit = res
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

    # -- quantification -------------------------------------------------

    def quantify(self, matrix=None, thickness=None, beam_MeV=1.0,
                 theta_deg=135.0, normalise=True, absolute=False):
        """Fitted areas to weight percent.

        Needs three things beyond the fit: an efficiency curve, an assumed
        matrix, and a thickness. The matrix matters because self-absorption
        is computed through it - which makes this mildly circular, so it is
        iterated once from a first-pass estimate unless you supply one.

        matrix     {Z: weight_fraction} or None to bootstrap from the areas
        thickness  mg/cm2; None means thick target (larger than the range)
        normalise  scale the result to sum to 100 wt%. Almost always what you
                   want, and it cancels the solid angle and charge constants
                   that are not otherwise known. Turn it off only if you have
                   an absolute calibration.

        Returns {Z: wt%}. K-shell elements only - the yield model has no L or
        M ionisation cross sections, so an L-fitted element is deliberately
        absent rather than silently wrong.
        """
        if self._fit is None:
            raise RuntimeError('fit first')
        if self.efficiency is None:
            raise RuntimeError('load a detector efficiency curve first; '
                               'without it areas cannot become concentrations')
        ar = self.areas()
        zk = [Z for (Z, sh) in self._meta if sh == 1]
        if not zk:
            raise RuntimeError('no K-shell components in this fit')

        if matrix is None:
            matrix = self._bootstrap_matrix(zk, ar)
        mz = list(matrix.keys())
        mw = [matrix[z] for z in mz]
        thick = thickness if thickness else 1e4      # thick target

        lym = LayeredYieldModel(self.db)
        lay = Layer(mz, mw, thick, 'matrix')
        yy = lym.yields([lay], zk, E0=beam_MeV, theta_deg=theta_deg,
                        mac=self.options.mac, fy=self.options.fluor_yield,
                        elam_zmax=self.options.fluor_elam_zmax, n_steps=900)
        self._yield_detail = getattr(lym, 'last_detail', {})

        out = {}
        for Z in zk:
            nm = self.db.sym[Z]
            a = ar.get(nm, 0.0)
            if a <= 0 or not np.isfinite(yy.get(Z, np.nan)) or yy[Z] <= 0:
                continue
            branch = max(i for _, i in self.db.line_list(Z, 1))
            eff = self.efficiency(self.db.line_energy(Z))
            if eff <= 0:
                continue
            out[Z] = a * branch / (yy[Z] * eff)
        if absolute:
            # counts = concentration x yield x charge x solid angle x norm.
            # No normalisation anywhere, so the SUM is a real measurement and
            # a bad fit shows up as a total that is not 100%.
            scale = self.geometry.scale()
            out = {Z: v / scale for Z, v in out.items()}
            self._conc_absolute = dict(out)
            self._conc_sum = sum(out.values())
        elif normalise and out:
            t = sum(out.values())
            out = {Z: 100.0 * v / t for Z, v in out.items()}
            self._conc_absolute = None
            self._conc_sum = None
        self._conc = out
        return out

    # -- charge, from the run log ---------------------------------------

    UC_PER_DOSE_COUNT = 1.0e-4
    """Microcoulomb per LMF dose count.

    Measured, not assumed: 21 runs of the 15-Dec-2023 session were regressed
    against the 'Q (nC)' column of their OMDAQ run log, giving 0.0001000 uC
    per count with a correlation of 1.000000 and 0.5% scatter that is entirely
    the log's three-decimal rounding. The digitiser therefore emits one count
    per 0.1 nC.

    This is a property of the digitiser range, so it is a default rather than
    a law. Confirm it for a session by calling
    io.runlog.calibrate_digitiser() whenever a run log is to hand.
    """

    def auto_charge(self, path=None, quantum=None):
        """Integrated charge in uC for the loaded run, preferring the log.

        Two sources, in order of trust:
          1. the OMDAQ run log beside the data, which records Q in nC
             directly - this is the measurement
          2. the LMF dose counter times the digitiser quantum - a very good
             proxy, but it inherits whatever range the digitiser was on

        Returns (charge_uC, source) so a caller can say which was used, and
        (None, reason) rather than guessing when neither is available.
        """
        from .io import runlog
        p = path or self.path
        if not p:
            return None, 'nothing loaded'
        try:
            q = runlog.charge_for(p)
        except Exception:
            q = None
        if q:
            return q, 'run log'
        try:
            from .io import lmf as _lmf
            c = _lmf.clocks(p)
        except Exception as ex:
            return None, 'no run log and no readable dose counter (%s)' % ex
        n = c.get('charge_counts') or 0
        if n <= 0:
            return None, 'no run log, and the dose counter is empty'
        return n * (quantum or self.UC_PER_DOSE_COUNT), 'LMF dose counter'

    def live_seconds(self):
        """Acquisition duration in seconds, from the LMF clocks. None if the
        file does not carry them - and then the pile-up cap cannot be set."""
        if not self.path or not str(self.path).lower().endswith('.lmf'):
            return None
        try:
            from .io import lmf as _l
            return _l.clocks(str(self.path)).get('duration_s')
        except Exception:
            return None

    def pileup_cap(self):
        """Largest pile-up area the measured count rate can justify.

        None when the duration is unknown, which leaves the amplitude free -
        the old behaviour, and the honest one when the rate cannot be
        measured. Whenever the duration IS known the cap applies, because an
        unbounded pile-up component is the single most effective way for this
        fitter to hide a modelling error.
        """
        if self.spectrum is None:
            return None
        secs = self.live_seconds()
        if not secs:
            return None
        total = float(np.sum(self.full_spectrum
                             if self.full_spectrum is not None
                             else self.spectrum))
        frac = _fit.expected_pileup_fraction(
            total, secs, self.options.shaping_time_us)
        if frac is None:
            return None
        return frac * total * float(self.options.pileup_headroom)

    def closure(self, **kw):
        """The absolute sum, as a fraction of 100 wt%.

        This is the diagnostic that normalising throws away. A fit that has
        lost intensity to a mismodelled peak shape, or that is missing an
        element, still sums to exactly 100% once normalised and looks
        healthy. Absolutely, it does not.

        Interpreting it:
            0.95 - 1.05   consistent; the chain is holding together
            below 0.9     something is missing - an unfitted element, dead
                          time not corrected, or the charge read high
            above 1.1     something is double counted, or the charge read
                          low, or the solid angle is too small

        None when the geometry is incomplete, because without charge and
        solid angle there is nothing to compare against.
        """
        if not self.geometry.complete():
            return None
        self.quantify(absolute=True, **kw)
        return self._conc_sum / 100.0

    def _bootstrap_matrix(self, zk, areas):
        """A first guess at the matrix from raw areas, used only to compute
        self-absorption. Crude on purpose - the result is iterated once by
        quantify() being called again with the answer, and the absorption
        correction is not very sensitive to a rough matrix."""
        w = {}
        for Z in zk:
            a = areas.get(self.db.sym[Z], 0.0)
            if a > 0:
                w[Z] = a
        if not w:
            return {14: 1.0}
        t = sum(w.values())
        return {Z: v / t for Z, v in w.items()}

    def concentration_table(self):
        """[(symbol, wt%, relative error %)] sorted by abundance."""
        if not self._conc:
            return []
        er = self.errors()
        ar = self.areas()
        rows = []
        for Z, c in self._conc.items():
            nm = self.db.sym[Z]
            rel = (100 * er.get(nm, 0) / ar[nm]) if ar.get(nm) else float('nan')
            rows.append((nm, c, rel))
        return sorted(rows, key=lambda r: -r[1])

    # -- layered quantification -----------------------------------------

    def quantify_layered(self, layers, assignment, beam_MeV=1.0,
                         theta_deg=135.0, normalise_within_layer=True):
        """Concentrations for a layered sample.

        What is and is not determinable
        -------------------------------
        One spectrum cannot give both the composition and the thickness of
        every layer. There are more unknowns than measurements and no amount
        of fitting fixes that, so something has to be supplied. Three cases
        are tractable:

          1. known stack, unknown concentrations - thicknesses and matrix
             compositions given, solve for how much of each element sits in
             the layer it is assigned to. That is this method.
          2. known compositions, one unknown thickness - solve for that
             instead, with solve_thickness() below.
          3. a film on a known substrate - case 2 with one layer, the common
             one, giving an areal density rather than a concentration.

        The degeneracy that matters
        ---------------------------
        An element present in more than one layer CANNOT be separated from a
        single spectrum. Carbon in a surface film and carbon in a buried
        organic layer make the same peak; only their attenuation differs, and
        that is one equation for two unknowns. So `assignment` is required
        rather than inferred - you are stating where you believe each element
        lives, and the answer is only as good as that belief.

        layers      [dict(zlist=, wfrac=, thick=)] surface first, mg/cm2
        assignment  {Z: layer_index}

        Returns {Z: wt% within its own layer}.
        """
        if self._fit is None:
            raise RuntimeError('fit first')
        if self.efficiency is None:
            raise RuntimeError('load a detector efficiency curve first')

        stack = [Layer(L['zlist'], L['wfrac'], L['thick'],
                       L.get('name', 'layer%d' % i))
                 for i, L in enumerate(layers)]
        ar = self.areas()
        zk = [Z for (Z, sh) in self._meta if sh == 1]
        lym = LayeredYieldModel(self.db)

        out, detail = {}, {}
        for Z in zk:
            nm = self.db.sym[Z]
            a = ar.get(nm, 0.0)
            k = assignment.get(Z)
            if a <= 0 or k is None:
                continue
            yy = lym.yields(stack, [Z], E0=beam_MeV, theta_deg=theta_deg,
                            mac=self.options.mac, fy=self.options.fluor_yield,
                            elam_zmax=self.options.fluor_elam_zmax,
                            n_steps=self._steps_for(layers),
                            in_layer={Z: k})
            y = yy.get(Z, 0.0)
            if not np.isfinite(y) or y <= 0:
                detail[Z] = 'no yield - the line may be wholly absorbed'
                continue
            branch = max(i for _, i in self.db.line_list(Z, 1))
            eff = self.efficiency(self.db.line_energy(Z))
            if eff <= 0:
                continue
            out[Z] = a * branch / (y * eff)
            d = dict(getattr(lym, 'last_detail', {}).get(Z, {}) or {})
            d['layer'] = k
            detail[Z] = d

        if normalise_within_layer:
            by_layer = {}
            for Z in out:
                by_layer.setdefault(assignment[Z], []).append(Z)
            for zs in by_layer.values():
                t = sum(out[Z] for Z in zs)
                if t > 0:
                    for Z in zs:
                        out[Z] = 100.0 * out[Z] / t
        self._conc = out
        self._layer_detail = detail
        self._layer_assignment = dict(assignment)
        return out

    @staticmethod
    def _steps_for(layers, base=900, per_layer=25, cap=30000):
        """Depth steps enough to resolve the THINNEST layer.

        The yield integral walks the stack in equal energy-loss steps. A layer
        thinner than one step gets one step's worth of yield no matter how
        thin it is, so the model silently floors out - a 5 ug/cm2 film and a
        50 ug/cm2 film return the same answer, and a thickness solve then has
        nothing to bisect on. This picks a step count that puts at least
        `per_layer` steps inside the thinnest layer.
        """
        t = [float(L.get('thick', 0.0)) for L in layers
             if float(L.get('thick', 0.0)) > 0]
        if not t:
            return base
        total = sum(t)
        need = int(per_layer * total / min(t))
        return int(min(max(base, need), cap))

    def instrument_constant(self, layers, ref_Z, ref_layer, ref_wt_percent,
                            beam_MeV=1.0, theta_deg=135.0):
        """Counts per unit (yield x efficiency x weight fraction).

        The yield model returns X-rays per unit concentration in arbitrary
        units - it knows the physics but not the solid angle, the charge, or
        the constant hidden in atoms-per-gram. quantify() never needs that
        constant because normalising to 100 wt% cancels it.

        An absolute quantity cannot be normalised away, so anything absolute -
        a film thickness, an areal density - needs the constant measured. The
        practical way is an internal standard: one element whose concentration
        in a known layer you already know. That is what a standard IS, and it
        is why a measurement against one is worth more than an absolute
        calculation.
        """
        if self._fit is None:
            raise RuntimeError('fit first')
        if self.efficiency is None:
            raise RuntimeError('load a detector efficiency curve first')
        nm = self.db.sym[ref_Z]
        area = self.areas().get(nm, 0.0)
        if area <= 0:
            raise ValueError('reference element %s has no fitted area' % nm)
        stack = [Layer(L['zlist'], L['wfrac'], L['thick'], L.get('name', ''))
                 for L in layers]
        lym = LayeredYieldModel(self.db)
        yy = lym.yields(stack, [ref_Z], E0=beam_MeV, theta_deg=theta_deg,
                        mac=self.options.mac, fy=self.options.fluor_yield,
                        elam_zmax=self.options.fluor_elam_zmax,
                        n_steps=self._steps_for(layers),
                        in_layer={ref_Z: ref_layer})
        y = yy.get(ref_Z, 0.0)
        if not np.isfinite(y) or y <= 0:
            raise ValueError('reference %s has no computable yield in layer %d'
                             % (nm, ref_layer))
        branch = max(i for _, i in self.db.line_list(ref_Z, 1))
        eff = self.efficiency(self.db.line_energy(ref_Z))
        return area * branch / (y * eff * (ref_wt_percent / 100.0))

    def solve_thickness(self, layers, unknown_index, element_Z,
                        known_wt_percent, ref_Z, ref_layer, ref_wt_percent,
                        beam_MeV=1.0, theta_deg=135.0,
                        bracket=(1e-6, 1e3), tol=1e-3):
        """Thickness of one layer, calibrated against an internal standard.

        The complement of quantify_layered. Instead of asking how much of an
        element sits in a layer of known thickness, ask how thick a layer of
        known composition must be to produce the observed peak. That is the
        right question for a film - a carbon coat has a composition you
        already know and a thickness you do not.

        ref_Z / ref_layer / ref_wt_percent name the internal standard, which
        supplies the instrument constant (see instrument_constant). Without
        one the answer would be in arbitrary units, so it is required rather
        than optional.

        Bisection on log thickness: yield is monotonic in thickness at fixed
        composition, and the plausible range spans decades.
        """
        k = self.instrument_constant(layers, ref_Z, ref_layer, ref_wt_percent,
                                     beam_MeV, theta_deg)
        nm = self.db.sym[element_Z]
        target = self.areas().get(nm, 0.0)
        if target <= 0:
            raise ValueError('%s has no fitted area' % nm)
        branch = max(i for _, i in self.db.line_list(element_Z, 1))
        eff = self.efficiency(self.db.line_energy(element_Z))
        lym = LayeredYieldModel(self.db)

        def predicted(t):
            ls = [dict(L) for L in layers]
            ls[unknown_index]['thick'] = t
            stack = [Layer(L['zlist'], L['wfrac'], L['thick'],
                           L.get('name', '')) for L in ls]
            yy = lym.yields(stack, [element_Z], E0=beam_MeV,
                            theta_deg=theta_deg, mac=self.options.mac,
                            fy=self.options.fluor_yield,
                            elam_zmax=self.options.fluor_elam_zmax,
                            n_steps=self._steps_for(ls),
                            in_layer={element_Z: unknown_index})
            y = yy.get(element_Z, 0.0)
            if not np.isfinite(y) or y <= 0:
                return 0.0
            return k * y * eff * (known_wt_percent / 100.0) / branch

        lo, hi = bracket
        flo = predicted(lo) - target
        fhi = predicted(hi) - target
        if flo * fhi > 0:
            raise ValueError(
                'no thickness between %.4g and %.4g mg/cm2 reproduces the '
                'observed %s area of %.0f counts (the model spans %.4g to '
                '%.4g). Check the assumed composition or the standard.'
                % (lo, hi, nm, target, predicted(lo), predicted(hi)))
        for _ in range(60):
            mid = float(np.sqrt(lo * hi))
            fm = predicted(mid) - target
            if abs(fm) < tol * target:
                return mid
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return float(np.sqrt(lo * hi))

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
            efficiency=(self.efficiency.source if self.efficiency else None),
            database=str(config.database_path(required=False)),
            options=self.options.to_dict(),
            scrappyfit_version=scrappyfit.__version__)
        (out / ('%s_analysis.json' % pre)).write_text(json.dumps(meta, indent=2))
        written.append('%s_analysis.json' % pre)

        if self._conc:
            rows = ['# element   wt_percent   rel_error_percent']
            for nm, c, rel in self.concentration_table():
                rows.append('%-8s %12.4f %14.2f' % (nm, c, rel))
            (out / ('%s_concentrations.txt' % pre)).write_text(
                chr(10).join(rows) + chr(10))
            written.append('%s_concentrations.txt' % pre)

        if self.mask is not None:
            np.save(out / ('%s_mask.npy' % pre), self.mask)
            written.append('%s_mask.npy' % pre)
        return written
