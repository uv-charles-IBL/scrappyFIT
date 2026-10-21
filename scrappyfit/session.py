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

import math
import numpy as np

from . import config
from . import yieldset as _yieldset
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


#: M-subshell fractions of the total M yield, MEASURED on 2 MeV 4He gold
#: spectra (runs 380001 and 381001, 983k and 1.15M Au M counts) where the
#: subshells are fully resolved by the statistics. The two agreed to the
#: third decimal. Used as a soft prior so that a weak spectrum, where the
#: 81 eV M4/M5 pair is degenerate, defers to them; a strong one overrides
#: them. They are for helium at 0.5 MeV/amu on gold and are the best
#: available for other heavy elements only until measured.
M_SUBSHELL_PRIOR = {'M5': 0.555, 'M4': 0.385, 'Mz': 0.0065,
                    'M3': 0.0375, 'M2': 0.014, 'M1': 0.002}


def _kbeta_free(o, Z, sym):
    """free_kbeta is True (every element from free_kbeta_zmin up), False, or a
    set of symbols / Z. Use a set naming the MAJOR elements: a minor element's
    Kb is not determined by its own counts, and freeing it lets the fit trade
    it against a neighbour (Fe Ka under Mn Kb)."""
    f = o.free_kbeta
    if f is True:
        return Z >= o.free_kbeta_zmin
    if not f:
        return False
    names = {str(x).lower() for x in f}
    return sym.lower() in names or str(Z) in names


def _split_kbeta(lines):
    """Split a K line list into (Ka lines, Kb lines) at the midpoint between the
    strongest line and the strongest line more than 50 eV above it."""
    if len(lines) < 2:
        return lines, []
    ea = max(lines, key=lambda t: t[1])[0]
    up = [t for t in lines if t[0] > ea + 0.05]
    if not up:
        return lines, []
    eb = max(up, key=lambda t: t[1])[0]
    cut = 0.5 * (ea + eb)
    return [t for t in lines if t[0] <= cut], [t for t in lines if t[0] > cut]


class FitOptions:
    """Everything that changes a fit result, in one place so it can be saved
    beside the numbers. A result without its options is not reproducible."""

    def __init__(self, **kw):
        self._e_low = 0.20
        self.e_high = 6.50
        self.noise = 32.74
        self.fano = 28.05
        self.tail_amp = 0.08
        self.tail_len = 1.0
        self.split_m_subshells = True
        """Fit M-shell elements as five subshell components (M1..M5) with
        the full xraylib transition list, instead of the table's five lines.
        See Database.m_subshell_lines for why. Needs xraylib; falls back to
        the table when it is absent."""
        self.snip_passes = 8
        """SNIP passes, GeoPIXE's default. The operator lowers it to keep a
        broad continuum hump as background: on run 354016 GeoPIXE's exported
        SNIP corresponds to about 2 passes at the detector width."""
        self.snip_boost = False
        """strip_clip.pro's 'boost' - divide by the transmission before
        stripping, halve the passes, multiply back. Off by default in
        GeoPIXE and here."""
        self.snip_trim_seb = False
        """strip_clip.pro's 'Trim SEB' option; see background.snip."""
        self.snip_e_trans = None
        """Hybrid split energy (keV): passes below it, double above. None
        follows e_inflection.pro, which returns 6 keV with no filter."""
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
        self.subthreshold_pileup_kev = 0.0
        """Width in keV of the sub-threshold partner distribution for
        SubThresholdPileupComponent; 0 disables it. Set to about 0.35 for a
        spectrum taken with the DPP fast channel not triggering."""
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
        # 'detector'  GeoPIXE's line.pro shape: Gaussian + one exponential tail
        #             from the .detector file, plus this package's optional shelves
        # 'empirical' Hypermet shape with parameters measured on single-line
        #             standards (fitting.hypermet); load with Session.load_response
        self.response_model = 'detector'
        # Fit Kb separately from Ka: True for every Z >= free_kbeta_zmin, or a
        # set of element symbols (recommended: the major elements). The atomic Kb/Ka of
        # the line table is a free-atom value; in compounds it moves (Cl in NaCl
        # +30 %, Mn in oxides +9 %, Si +50 % measured on 2026-09-23), and a fixed
        # ratio makes the fit borrow a neighbouring element to fill the gap.
        self.free_kbeta = False
        self.free_kbeta_zmin = 14
        self.response_scale = False
        # Refine the sum-peak energy deficit (percent) instead of holding
        # sum_deficit. Measured 0.06-0.40 % on the Amptek DPP, against the 0.1 %
        # GeoPIXE assumes.
        self.refine_sum_deficit = True
        self.sum_deficit_spread = 0.0
        """Percent: draw each sum peak as a uniform band of deficits from sum_deficit
        to sum_deficit + this, for broad flat-topped sum humps. 0 = a single line."""
        self.freeze_nonlinear = False
        """Refine nothing non-linear (cal, width, tail, shelf, contact, sum deficit):
        only the areas are solved. With Session.start_pars set to a converged
        whole-field fit, region fits become linear in the data - comparable
        between regions and immune to refinement paths. Two masks over the same
        pixels gave Si 4.5 % and Pb M 14 % apart with free refinement on 287424."""
        # partial pile-up: a flat box from max(E_i, E_j) to E_i + E_j under
        # every sum peak (fitting.fit.PileupPlateauComponent)
        self.use_pileup_plateau = True
        # add a second plateau rising toward the sum. Better on 403001/006,
        # but it destabilised 403004 and 403005 - off until that is understood
        self.pileup_ramp = False
        # DPP fast-channel threshold as an energy, keV. When set: sub-threshold
        # pile-up uses the measured spectrum below it as the partner
        # distribution, and sum peaks / plateau keep only pairs above it.
        # Amptek THFA is in fast-channel units; its energy depends on gain and
        # TPFA, so fit it (scan) rather than convert.
        self.fast_threshold_kev = None
        # width of the Kb/Ka prior as a fraction of the pair's counts: a
        # departure of this much from the table ratio costs one sigma
        self.kbeta_prior_strength = 0.15
        kw = dict(kw)
        if 'e_low' in kw:
            # taken as given: a noise passed alongside is referenced to it
            self._e_low = float(kw.pop('e_low'))
        self.__dict__.update(kw)

    @property
    def e_low(self):
        return self._e_low

    @e_low.setter
    def e_low(self, value):
        """Moving E low keeps the physical width, not the number in noise.

        The width is FWHM_ch^2 = noise^2 + fano^2 (E - e_low), noise and
        fano in channels, E in keV: noise is the FWHM at e_low, GeoPIXE's
        fwhm_origin. GeoPIXE re-derives it from the detector's w0, w1 at the
        current e_low on every fit setup (pixe_setup.pro:44,106-107), so a
        range change does not alter the width at any energy. Re-referencing
        here does the same for every source of noise - a .detector,
        estimate_resolution, a hand value. Session.start_pars, a previous
        fit's parameters, is re-referenced when it is used
        (Session._start_shape).
        Without it the start width depended on whether the range was set
        before or after load_detector (run 404004: start noise 30.906 vs
        30.645 ch, 2026-10-18, tools/gui_fit_dump.py).
        """
        value = float(value)
        w2 = self.noise ** 2 + self.fano ** 2 * (value - self._e_low)
        self.noise = math.sqrt(max(w2, 1e-12))
        self._e_low = value

    def refine_groups(self):
        if self.freeze_nonlinear:
            return ()
        g = list(self.refine)
        if self.use_pileup and self.refine_sum_deficit and 'sumdef' not in g:
            g.append('sumdef')
        if self.response_model == 'empirical':
            # the measured response fixes the tail and shelf shape; cal and width
            # float, and with response_scale the tail and shelf AMOUNTS may be
            # scaled per spectrum (a5, a6), as line.pro's tail is
            keep = ('cal', 'width', 'sumdef') + (('tail',) if self.response_scale else ())
            return tuple(x for x in g if x in keep)
        for flag, name in ((self.use_escape_step, 'shelf'),
                           (self.use_contact_step, 'contact'),
                           (self.use_window_step, 'window')):
            if flag and name not in g:
                g.append(name)
        return tuple(g)

    def to_dict(self):
        return {('e_low' if k == '_e_low' else k):
                (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def find_yield_beside(data_path):
    """The .yield next to a data file that belongs to it, or None.

    GeoPIXE has no naming rule linking a yield to a spectrum: the only link
    it keeps is the yield path saved in the fit setup (.pcm), and a missing
    file there just opens a file requester (fit_setup.pro:2597-2611). So
    this is our own rule, kept narrow so it cannot pick a stranger's file:
      1. the same stem with GeoPIXE's extension: 404004_sic.txt ->
         404004_sic.yield;
      2. else a .yield whose .provenance.json sidecar (written by File >
         Export yield) names this data file as its source spectrum, if
         exactly one does.

    Never raises: a yield that cannot be found must not stop a load.
    """
    try:
        p = pathlib.Path(data_path)
        same = p.with_suffix('.yield')
        if same.is_file():
            return str(same)
        target = os.path.normcase(str(p.resolve()))
        sidecars = list(p.parent.glob('*.provenance.json'))
    except (OSError, ValueError, TypeError):
        return None
    hits = []
    for sc in sidecars:
        try:
            y = sc.with_name(sc.name[:-len('.provenance.json')] + '.yield')
            if not y.is_file():
                continue
            src = json.loads(sc.read_text(encoding='utf-8')).get(
                'source_spectrum')
            if (isinstance(src, str) and src and os.path.normcase(
                    str(pathlib.Path(src).resolve())) == target):
                hits.append(str(y))
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    return hits[0] if len(hits) == 1 else None


class Session:
    """One dataset and its analysis."""

    def __init__(self, database=None, options=None):
        # ShapePars to start (or, with freeze_nonlinear, fix) the fit at. The
        # GUI sets it only while "Hold line shape" is ticked.
        self.start_pars = None
        self.options = options or FitOptions()
        self._db_root = database
        self._db = None
        self.efficiency = None
        self.geometry = Geometry()
        self.detector = None        # a GeoPIXE .detector model, if loaded
        self._tail_fns = None       # its energy-dependent tail functions
        self.response = None        # fitting.hypermet.EmpiricalResponse, if loaded
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
        self.map_info = {}          # how the last element_map was made
        self.dam = None
        self._live_s = None
        self.mca_info = None
        self.yield_path = None      # the sample's GeoPIXE .yield, if known
        self.yield_source = ''      # how it became known; see set_yield()
        self.yield_set = None       # yieldset.YieldSet QUANTIFY reads, if any
        self.yield_region = None    # the region of it in use
        self.yield_search = ''      # why there is no set, when there is none
        self.quantify_basis = ''    # what the last QUANTIFY used
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
        self._conc_yields = None
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
        self.yield_path = None
        self.yield_source = ''
        self.yield_set = None
        self.yield_region = None
        self.yield_search = 'no data file, so nothing to search for'
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
        elif ext == '.mca':
            from .io import amptek as _am
            counts, info = _am.read_mca(p)
            self.spectrum = np.asarray(counts, dtype=float)
            self.events = None
            self.mca_info = info
            self._live_s = info.get('live_s')
        else:
            self._load_columns(p)
        self.full_spectrum = (None if self.spectrum is None
                              else self.spectrum.copy())
        found = find_yield_beside(p)
        ys, why = _yieldset.find_yield_set(p, beside=found)
        if ys is not None:
            self.set_yield_set(ys, getattr(ys, 'remembered_region', None)
                               or None, why)
        else:
            self.yield_search = why
            if found:
                self.set_yield(found, 'found beside the data')
        self.invalidate()
        return self

    # -- the sample's yield file ----------------------------------------

    def set_yield(self, path, source):
        """Record the .yield that describes this sample.

        source says how it became known - 'exported', 'loaded', 'found
        beside the data' - so a dialog can show it. Dialogs that draw a
        stack (layer stack and escape) start from it, and the sample model
        offers to copy it. Nothing in the fit reads it.

        A file of the current yield set just becomes the one shown (and its
        region the one in use); any other file replaces the set with a set
        of one, which quantify_yields() then reads.
        """
        self.yield_path = str(path) if path else None
        self.yield_source = source if path else ''
        if not path:
            return
        ys = self.yield_set
        key = os.path.normcase(os.path.abspath(str(path)))
        hit = None if ys is None else next(
            (e for e in ys.all
             if os.path.normcase(os.path.abspath(e.path)) == key), None)
        if hit is None and ys is not None and source == 'exported':
            # File > Export yield writes the one-layer model's own yields.
            # It is shown, but it does not displace the sample's set.
            return
        if hit is None:
            self.yield_set = _yieldset.YieldSet([str(path)])
            self.yield_set.source = source
            self.yield_region = self.yield_set.default_region()
        elif hit.region:
            self.yield_region = hit.region

    def set_yield_set(self, ys, region=None, source=''):
        """Make ys (a yieldset.YieldSet, or None) the sample's yield files.

        region picks which region's files QUANTIFY reads; None keeps the
        set's only region, or leaves it to be chosen when there are several.
        The first file of that region becomes yield_path, so the dialogs
        that draw a stack open on one of the files QUANTIFY uses.
        """
        self.yield_set = ys if ys is not None and len(ys) else None
        if self.yield_set is None:
            self.yield_region = None
            self.yield_path, self.yield_source = None, ''
            return
        ys.source = source
        if region is None or region not in ys.regions:
            region = ys.default_region()
        self.yield_region = region
        es = ys.entries(region)
        self.yield_path = es[0].path if es else ys.all[0].path
        self.yield_source = source

    def remember_yields(self):
        """Keep the current yield set and region for this data file."""
        if self.path and self.yield_set is not None:
            _yieldset.remember(self.path, self.yield_set.paths,
                               self.yield_region or '')

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
          * the GEOMETRY - active area from the crystal diameter and the
            source distance - so the solid angle does not have to be typed
            in by hand and then disagree with the file the efficiency and
            the tail came from. Only filled in where the caller has not
            already set a value, so a hand-entered geometry is never
            overwritten.
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
        # geometry: the file gives the crystal diameter in mm and the
        # distance from the source, which is exactly what solid_angle_msr
        # wants. 4.652 mm and 77.525 mm on the C1 deflector detector is
        # 17.0 mm2 at 2.83 msr.
        g = self.geometry
        dia = det.get('diameter')
        if dia and not getattr(g, 'area_mm2', None):
            g.area_mm2 = math.pi * (float(dia) / 2.0) ** 2
        dist = det.get('distance')
        if dist and not getattr(g, 'distance_mm', None):
            g.distance_mm = float(dist)

        w0, w1 = det.get('w0'), det.get('w1')
        if w0 and w1 and self.cal and self.cal[0]:
            # GeoPIXE: FWHM_keV^2 = w0 + w1 E.  Here: FWHM_ch^2 = noise^2 +
            # fano^2 (E - e_low), and FWHM_keV = FWHM_ch x cal_a. Equating
            # them gives fano and noise directly. noise is referenced to the
            # current e_low; FitOptions.e_low's setter keeps it so when the
            # range changes later.
            a = float(self.cal[0])
            fano = (w1 ** 0.5) / a
            noise2 = w0 / (a * a) + fano * fano * self.options.e_low
            if noise2 > 0:
                self.options.noise = noise2 ** 0.5
                self.options.fano = fano
        self._escape = None
        self.invalidate()
        return det

    def load_response(self, path_or_response):
        """Adopt a measured detector response (fitting.hypermet) and switch the
        line shape to it. Accepts a JSON path or an EmpiricalResponse."""
        from .fitting.hypermet import EmpiricalResponse
        r = (path_or_response if isinstance(path_or_response, EmpiricalResponse)
             else EmpiricalResponse.load(str(path_or_response)))
        mac = self.options.mac
        r.mu_al = lambda E, db=self.db: db.mu_compound([13], [1.0], float(E), mac) or 0.0
        self.response = r
        self.options.response_model = 'empirical'
        self.invalidate()
        return r

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

    def parse_line(self, spec):
        """'Si' -> (14, 1); 'Ag L', 'AgL', 'Ag:L' -> (47, 2); 'Pb M' -> (82, 3).

        After a space or colon the shell is K, L or M in either case
        ('ag l', 'Si K'). Run together, a whole token that is a symbol stays
        that element, as before ('CL', 'Cl' chlorine, 'PM' promethium); else
        a trailing L or M names the shell, as the --elements option reads it.
        A bare symbol is its K lines.
        """
        raw = str(spec).strip()
        parts = raw.replace(':', ' ').split()
        sh = 1
        if len(parts) == 2 and parts[1].upper() in ('K', 'L', 'M'):
            tok, sh = parts[0], 'KLM'.index(parts[1].upper()) + 1
        else:
            tok = ''.join(parts)
            if (tok.lower() not in self.db.z and len(tok) > 1
                    and tok[-1] in 'LMlm' and tok[:-1].lower() in self.db.z):
                sh = 2 if tok[-1] in 'Ll' else 3
                tok = tok[:-1]
        Z = self.db.z.get(tok.lower())
        if not Z:
            raise ValueError('not an element line: %r' % (spec,))
        return Z, sh

    def map_line_energy(self, Z, shell=1):
        """keV of the line a window map of (Z, shell) centres on: the K major
        line, L-alpha (La_ below the split, La1 above) or M-alpha1, which is
        db.line_energy(Z, shell). Not the strongest single entry: Pb Mb
        (2.442 keV, 0.355) outweighs Ma1 alone (0.301) but not the Ma1 + Ma2
        doublet at 2.34 (db.line_list(82, 3), vendored database, 2026-10-21).
        An element without that line falls back to the strongest it has."""
        db = self.db
        e = db.line_energy(Z, shell)
        if shell == 1 or e > 0.01:
            return e
        lines = db.line_list(Z, shell)
        return max(lines, key=lambda t: t[1])[0] if lines else 0.0

    def map_line_name(self, Z, shell=1):
        """How the Maps tab writes a line: 'Si', 'Ag L', 'Pb M'."""
        return self.db.sym[Z] + {1: '', 2: ' L', 3: ' M'}[shell]

    def element_map(self, which, width=0.075, binning=4, subtract=True):
        """Window map for one line. Needs list-mode data.

        which   a line ('Si', 'Ag L', 'AgL', 'Pb M'; see parse_line) or an
                energy in keV. A line whose window is not inside the spectrum
                raises ValueError rather than returning an empty map: an
                all-zero map looks like data, and a percentile mask on it
                selects every pixel (287424, 2026-10-20).

        Flanking subtraction fails where a weak line sits between two strong
        ones - aluminium between magnesium and silicon is the standard case,
        and it goes negative. Use da_maps() where a DA matrix exists.
        """
        if self.events is None:
            raise RuntimeError('maps need a list-mode (.lmf) file')
        self.map_info = {}           # nothing stale if this map fails
        if isinstance(which, str):
            E0 = self.map_line_energy(*self.parse_line(which))
        else:
            E0 = float(which)
        a, b = self.cal
        lo, hi = _maps.window(E0 - width, E0 + width, a, b)
        nch = len(self.full_spectrum if self.full_spectrum is not None
                  else self.spectrum)
        if E0 <= 0 or lo < 0 or hi >= nch:
            raise ValueError('%s at %.3f keV is outside the spectrum '
                             '(channels 0-%d, %.2f-%.2f keV)'
                             % (which, E0, nch - 1, b, a * (nch - 1) + b))
        bg = None
        if subtract:
            # flanks only from channels that hold events: below the zero of
            # energy and past the last count there is nothing to read. The
            # first count is the detector threshold, and a flank just above
            # it is on the rising edge and still reads somewhat low.
            full = (self.full_spectrum if self.full_spectrum is not None
                    else self.spectrum)
            nz = np.flatnonzero(full)
            first = max(int(np.ceil(-b / a)) if a > 0 else 0,
                        int(nz[0]) if nz.size else 0, 0)
            last = int(nz[-1]) if nz.size else nch - 1
            bg = _maps.flanks(lo, hi, first, last)
            if not bg:
                raise ValueError('%s: neither background flank is on the '
                                 'spectrum; map it without subtraction'
                                 % (which,))
        self.map_info = dict(
            kind='window', line=str(which), energy=E0, window=(lo, hi),
            background=('none' if bg is None else
                        'linear between flanks' if len(bg) == 2 else
                        'flat, %s flank only'
                        % ('low' if bg[0][1] < lo else 'high')),
            flanks=bg or ())
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
        """The SNIP background, built the way strip_clip.pro builds it.

        background.snip() is a port of strip_clip.pro and takes everything
        GeoPIXE gives it. This property used to call it with none of that -
        no transmission boost, no detector width, three passes - and the
        result was a background that over-read the low-energy continuum
        2.2x on run 380001 and could not follow the 2-3 keV bremsstrahlung
        hump at all. Three inputs matter:

          trans   detector efficiency x filter x sample self-absorption. The
                  spectrum is divided by this BEFORE stripping and the strip
                  multiplied back after, so the clip follows the true
                  continuum rather than one bent by the efficiency roll-off.
                  boost_back.pro softens it with an atan cap so that a
                  vanishing efficiency does not blow the boost up.
          w0, w1  the detector's resolution, so the clip width tracks the
                  real peak width at every energy (2 x FWHM). The defaults
                  are for a 160 eV detector, which is the wrong width here.
          passes  strip_clip.pro defaults to 8, halved to 4 when boosting.
        """
        if self._bk is None:
            o = self.options
            a, b = self.cal
            n = len(self.spectrum)
            E = a * np.arange(n) + b

            trans = None
            if self.efficiency is not None and o.snip_boost:
                from .fitting.background import boost_correction
                try:
                    eff = np.array([self.efficiency(max(e, 0.05)) for e in E])
                    trans = boost_correction(E, eff=eff)
                    # boost_back.pro: trans = 1 / (cap * atan((1/trans)/cap))
                    cap = 100.0
                    trans = 1.0 / (cap * np.arctan((1.0 / trans) / cap))
                except Exception:
                    trans = None

            w0 = w1 = None
            det = self.detector
            if det and det.get('w0') and det.get('w1'):
                w0, w1 = float(det['w0']), float(det['w1'])

            passes = int(o.snip_passes)
            if trans is not None:
                passes = max(passes // 2, 1)
            e_trans = o.snip_e_trans
            if e_trans is None:
                # e_inflection.pro: first of 6, 8, ... 16 keV where the
                # filter transmits more than 15 %; 6 keV with no filter
                e_trans = 6.0
                ext = (self.detector or {}).get('external') or []
                if ext:
                    for e in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
                        e_trans = e
                        tr = 1.0
                        for lay in ext:
                            mu = self.db.mu_compound(lay['Z'], lay['F'], e,
                                                     o.mac) or 0.0
                            tr *= np.exp(-mu * lay['thick'] * 1e-3)
                        if tr > 0.15:
                            break

            self._bk = snip(self.spectrum, a, b, o.e_low, o.e_high,
                            passes=passes, w0=w0, w1=w1, trans=trans,
                            e_trans=e_trans, trim_seb=o.snip_trim_seb,
                            use_low_stats=True)
        return self._bk

    def _apply_efficiency(self, lines):
        """Weight an element's lines by the detector efficiency at each.

        A line table gives ATOMIC intensities - what the atom emits. What the
        detector records is that times the efficiency at each line's energy,
        and when the efficiency varies across an element's own lines the two
        are not proportional.

        It is not a small effect. GeoPIXE's donut2x example sits behind 200 um
        of aluminium, where efficiency runs 0.0054 at Fe Ka and 0.0194 at
        Fe Kb - a factor of 3.6 across one element. The atomic Kb/Ka is 0.132,
        so the RECORDED ratio should be 0.478; the spectrum measures 0.503.
        Without this the Fe component is pinned at 0.132 and cannot fit both
        lines, so the fit splits the difference: Ka overshoots by 1.34x, Kb
        undershoots by 2.6x, and the area inflates to cover the gap.

        Normalised to the strongest line, so a component's fitted area keeps
        meaning counts in its major line and quantify()'s own efficiency
        correction is not applied twice.
        """
        if self.efficiency is None or len(lines) < 2:
            return lines
        try:
            e0 = max(lines, key=lambda t: t[1])[0]
            ref = float(self.efficiency(e0))
            if not (ref > 0):
                return lines
            out = []
            for e, inten in lines:
                r = float(self.efficiency(e)) / ref
                if r > 0:
                    out.append((e, inten * r))
            return out or lines
        except Exception:
            return lines

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
            # M shell: one component per subshell, with the full transition
            # list from xraylib, when the table's five lines will not do.
            # The subshell populations are then fitted rather than assumed -
            # the same treatment GeoPIXE gives L1, L2 and L3. This also runs
            # when the table has NO M lines (Z = 45-72: Ag, I, Cs, lanthanides).
            if sh == 3 and self.options.split_m_subshells:
                subs = self.db.m_subshell_lines(Z)
                if subs:
                    for sub in ('M5', 'M4', 'Mz', 'M3', 'M2', 'M1'):
                        ll = subs.get(sub)
                        if not ll:
                            continue
                        ll = self._apply_efficiency(ll)
                        o = self.options
                        if not any(o.e_low <= e <= o.e_high
                                   for e, i in ll if i > 0.01):
                            continue
                        c = _fit.Component(self.db.sym[Z] + sub, ll,
                                           escape=self.escape_model)
                        # soft prior on the subshell fractions - see
                        # M_SUBSHELL_PRIOR for where the numbers come from
                        # (measured on gold, so not a prior for Z < 57,
                        # where M4/M5 radiate only through the Mz lines)
                        c.prior_group = self.db.sym[Z] + 'M'
                        c.prior_fraction = M_SUBSHELL_PRIOR.get(sub) if Z >= 57 else None
                        if self._tail_fns is not None:
                            c.tail_amp_fn, c.tail_len_fn = self._tail_fns
                        else:
                            c.tail_amp_fn = lambda E, a=o.tail_amp: a
                            c.tail_len_fn = lambda E, l=o.tail_len: l
                        out.append(c)
                        meta.append((Z, sh))
                    continue
            if not lines:
                continue
            lines = self._apply_efficiency(lines)
            # A component whose lines all sit outside the fit range has no
            # data to constrain it. Its design column is ~zero inside the
            # window, the solve is ill-conditioned there, and the area comes
            # back as any number at all - oxygen fitted to 1.7e13 counts on a
            # spectrum that started at 0.9 keV. Refuse it instead.
            o = self.options
            if not any(o.e_low <= e <= o.e_high for e, i in lines if i > 0.01):
                continue
            if sh == 1 and _kbeta_free(o, Z, self.db.sym[Z]):
                ka, kb = _split_kbeta(lines)
                if ka and kb and any(o.e_low <= e <= o.e_high for e, i in kb):
                    # intensities are NOT renormalised, so the Ka component's
                    # area keeps its meaning for quantify() and Kb/Ka is simply
                    # area(Kb) / area(Ka) x table ratio
                    cb = _fit.Component(self.db.sym[Z] + 'Kb', kb,
                                        escape=self.escape_model)
                    self._attach_shape(cb)
                    # Soft prior toward the table ratio (equal areas in these
                    # unnormalised units). A free Kb on a WEAK element removes
                    # the only thing separating it from a neighbour's Kb - Fe Ka
                    # sits 86 eV from Mn Kb, and with Fe Kb free the fit put
                    # 17 k counts of Fe under Mn Kb with no Fe Kb at all. With
                    # the prior a strong line's data override it and a weak
                    # line's Kb stays near the table.
                    grp = self.db.sym[Z] + 'K'
                    cb.prior_group, cb.prior_fraction = grp, 0.5
                    _fit.PRIOR_STRENGTH[grp] = o.kbeta_prior_strength
                    out.append(cb)
                    meta.append((Z, 'Kb'))
                    lines = ka
                    kb_group = grp
                else:
                    kb_group = None
            else:
                kb_group = None
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
            if self.response is not None and o.response_model == 'empirical':
                c.response = self.response
            if kb_group:
                c.prior_group, c.prior_fraction = kb_group, 0.5
            out.append(c)
            meta.append((Z, sh))
        if self.response is not None and self.options.response_model == 'empirical':
            for c in out:
                c.response = self.response
        self._meta = meta
        return out

    def _attach_shape(self, c):
        """Tail functions for a component built outside the main loop."""
        if self._tail_fns is not None:
            c.tail_amp_fn, c.tail_len_fn = self._tail_fns
        else:
            c.tail_amp_fn = lambda E, a=self.options.tail_amp: a
            c.tail_len_fn = lambda E, l=self.options.tail_len: l
        if self.response is not None and self.options.response_model == 'empirical':
            c.response = self.response

    def _start_shape(self):
        """Session.start_pars moved to the current range, or None.

        A converged ShapePars is referenced to the range it was fitted
        over: noise is the FWHM at its e_low, a[2] the channel of its eoc
        (0.7 e_low + 0.3 e_high). The same width and calibration at the
        current range: noise^2 += fano^2 (e_low - old e_low), as
        FitOptions.e_low does, and a[2] += a[3] (eoc - old eoc), a[3] in
        channels per keV. GeoPIXE
        re-derives both at every fit setup (pixe_setup.pro:43-44, 71-72,
        106-107), so a range change never moves a line or its width.
        """
        sp = self.start_pars
        if sp is None:
            return None
        import copy as _copy
        o = self.options
        p = _copy.deepcopy(sp)
        w2 = p.a[0] ** 2 + p.a[1] ** 2 * (o.e_low - sp.e_low)
        p.a[0] = math.sqrt(max(w2, 1e-12))
        p.e_low, p.e_high = o.e_low, o.e_high
        p.a[2] = sp.a[2] + sp.a[3] * (p.eoc - sp.eoc)
        return p

    def run_fit(self, elements, should_stop=None, progress=None):
        """Fit `elements`; the result is also kept as self.fit.

        should_stop  optional callable() -> bool, polled between passes and
                     inside every LM residual; True raises
                     fitting.fit.FitCancelled and leaves self.fit unchanged.
        progress     optional callable(text), told as each pass starts.
        """
        o = self.options
        a, b = self.cal
        comps = self.build_components(elements)
        if not comps:
            raise ValueError('no fittable components in %r' % (elements,))
        sump = None
        if o.use_pileup:
            sump = _fit.SumPeakComponent(
                comps, sum_deficit=o.sum_deficit, e_high=o.e_high,
                max_area=self.pileup_cap(), e_min=o.fast_threshold_kev)
            sump.deficit_spread = o.sum_deficit_spread
            comps = comps + [sump]
            if o.use_pileup_plateau:
                comps = comps + [_fit.PileupPlateauComponent(sump, 'flat')]
                if o.pileup_ramp:
                    comps = comps + [_fit.PileupPlateauComponent(sump, 'ramp')]

        subp = None
        if getattr(o, 'subthreshold_pileup_kev', 0.0) > 0 or o.fast_threshold_kev:
            ft = o.fast_threshold_kev
            subp = _fit.SubThresholdPileupComponent(
                [c for c in comps if c is not sump and not isinstance(c, _fit.PileupPlateauComponent)],
                ft if ft else o.subthreshold_pileup_kev,
                spectrum=(self.spectrum if ft else None), cal=(self.cal if ft else None))
            comps = comps + [subp]

        npass = [0]

        def go(start=None, refine_again=False):
            if should_stop is not None and should_stop():
                raise _fit.FitCancelled()
            npass[0] += 1
            if progress is not None:
                progress('fit pass %d%s' % (npass[0], ' (refining again)'
                                            if refine_again else ''))
            return _fit.fit_spectrum(
                self.spectrum, a, b, comps, o.e_low, o.e_high,
                noise=o.noise, fano=o.fano, tail_amp=o.tail_amp,
                tail_len=o.tail_len, background=self.background,
                refine=o.refine_groups(), nonneg=o.nonneg,
                start_pars=start, refine_from_start=refine_again,
                should_stop=should_stop)

        # start_pars without freeze_nonlinear is a starting point, so the
        # first pass refines from it. fit_spectrum alone would take a start
        # as converged and solve only the areas, which held any shape a
        # script carried in, misfit or not.
        start = self._start_shape()
        res = go(start=start,
                 refine_again=start is not None and not o.freeze_nonlinear)
        if subp is not None:
            got = dict(zip(res.names, res.areas))
            subp.update(got, res.pars, len(self.spectrum), a)
            res = go(start=res.pars)
        if sump is not None:
            # Sum-line intensities are products of the parent AREAS, which are
            # not known until the elements have been fitted once. sum_peaks.pro
            # has the same dependency and resolves it the same way: fit,
            # rebuild the sum lines from the result, fit again. Two passes is
            # enough - the sum peaks are a fraction of a percent, so their
            # effect on the parent areas that generated them is negligible.
            for _ in range(2):
                got = dict(zip(res.names, res.areas))
                if not sump.update(got, amplitude=got.get('pileup')):
                    break
                res = go(start=res.pars)
        if sump is not None or subp is not None:
            # The first pass refined calibration and width with the pile-up
            # terms still empty, so the non-linear parameters absorbed the sum
            # peaks: on 403001 the gain moved 3.7 % to cover the Si+Si sum at
            # 3.48 keV with Ti and Au lines. Refine once more now that the
            # pile-up lines exist, then rebuild them from the final areas.
            res = go(start=res.pars, refine_again=True)
            got = dict(zip(res.names, res.areas))
            snap = (list(getattr(sump, '_lines', []) or []), list(getattr(sump, '_pairs', []) or [])) if sump is not None else None
            if subp is not None:
                subp.update(got, res.pars, len(self.spectrum), a)
            if sump is not None:
                sump.update(got, amplitude=got.get('pileup'))
            res2 = go(start=res.pars)
            # Rebuilding the sum lines from the final areas can switch triples
            # on and change the shape the refinement converged against. Keep
            # it only if it does not make the fit worse.
            if res2.chi2 <= res.chi2 or snap is None:
                res = res2
            else:
                sump._lines, sump._pairs = snap
        self._fit = res
        return self._fit

    def preview(self, elements, should_stop=None):
        """The initial estimate GeoPIXE draws while a fit is being set up.

        Follows fit_setup_do_fit /update_initial: SNIP background at scale 1,
        areas as pixe_initial.pro makes them (lines 117-230), then one
        evaluation of the model with no refinement. For each line of an
        element inside the fit range whose intensity is above 0.01 of the
        element's major line (line 0, not necessarily the strongest; escape
        lines included, as correct_lines appends them):

            d = FWHM in channels at E       j = max(int(d/2), 1)
            s = (sum(data) - sum(background)) over channels int(x-j)..int(x+j)
                inclusive, divided by 2j+1
            area = s x 1.0645 x d / intensity

        The element takes the SMALLEST of the per-line areas above 0.1 (all of
        them if none is), floored at 3. pixe_initial.pro notes that a mean
        over-read when a minor line sits under a neighbour. Sum and pile-up
        terms are not drawn.

        The line shape is the one the next fit starts from: Session.start_pars
        (in the GUI, the held shape) moved to the current range, otherwise the
        options' starting width and tail at the session calibration.

        should_stop: optional callable() -> bool, checked per component;
        True raises fitting.fit.FitCancelled.

        Leaves self.fit and the element list quantify() reads untouched. It
        caches the SNIP background like any other caller of self.background,
        and build_components sets the shared Kb prior strength as a fit does,
        so the GUI calls it on a copy, on the one worker thread.

        Returns a FitResult with is_preview = True and zero errors.
        """
        o = self.options
        a, b = self.cal
        meta = self._meta
        try:
            comps = self.build_components(elements)
        finally:
            self._meta = meta
        bk = np.asarray(self.background, float)
        y = np.asarray(self.spectrum, float)
        n = len(y)
        eoc = 0.7 * o.e_low + 0.3 * o.e_high
        pars = self._start_shape()
        if pars is None:
            pars = _fit.ShapePars(o.noise, o.fano, (eoc - b) / a, 1.0 / a,
                                  tail_amp=o.tail_amp, tail_len=o.tail_len,
                                  e_low=o.e_low, e_high=o.e_high)
        A = np.zeros((n, len(comps)))
        areas = np.zeros(len(comps))
        for k, c in enumerate(comps):
            if should_stop is not None and should_stop():
                raise _fit.FitCancelled()
            A[:, k] = c.profile(pars, n)
            lines = list(getattr(c, 'lines', ()))
            if not lines or not lines[0][1] > 0:
                continue
            major = lines[0][1]
            got = []
            for E, bt in lines:
                if not (o.e_low <= E <= o.e_high) or bt / major <= 0.01:
                    continue
                x = float(pars.centroid(E))
                if not (0 <= x < n - 1):
                    continue
                d = float(pars.fwhm_channels(E))
                j = max(int(d / 2.0), 1)
                lo = max(int(x - j), 0)
                hi = min(int(max(x + j, 0)), n - 1)
                sig = (y[lo:hi + 1].sum() - bk[lo:hi + 1].sum()) / (2 * j + 1)
                got.append(sig * 1.0645 * d / bt)
            if not got:
                continue
            pos = [v for v in got if v > 0.1]
            areas[k] = max(min(pos) if pos else min(got), 3.0)
        model = A @ areas + bk
        # the channels of the range at the calibration the lines were drawn
        # at (a held shape keeps its own)
        E = pars.energy_of_channel(np.arange(n))
        ch = (E >= o.e_low) & (E <= o.e_high)
        chi2 = float(np.sum((y[ch] - model[ch]) ** 2
                            / np.maximum(model[ch], 1.0)))
        res = _fit.FitResult(areas, np.zeros_like(areas),
                             [c.name for c in comps], chi2,
                             int(ch.sum()) - len(comps), model, bk, pars, A)
        res.is_preview = True
        return res

    @property
    def fit(self):
        return self._fit

    def unexplained_peaks(self, top=5):
        """Residual peaks of the last fit with candidate lines.

        [(E_keV, excess, sigma, ['As Ka', 'Pb La', ...]), ...] largest first.
        The candidates are the K, L and M major lines within one FWHM of
        the peak, nearest first, so the user can see at a glance which
        element the fit was missing. This is the check that would have
        caught As on 380006 and the width blow-up it caused.
        """
        if self._fit is None:
            return []
        o = self.options
        a, b = self.cal
        fw = 1000.0 * a * float(np.sqrt(o.noise ** 2 + o.fano ** 2 * 5.0))
        pk = self._fit.residual_peaks(self.spectrum, a, b, o.e_low,
                                      o.e_high, fwhm_ev=fw, top=top)
        out = []
        for E, ex, sg in pk:
            cands = []
            for Z in range(11, 93):
                for sh, tag in ((1, 'Ka'), (2, 'La'), (3, 'Ma')):
                    le = self.db.line_energy(Z, sh) if sh == 1 else 0.0
                    if sh == 2:
                        le = self.db.lineE.get(Z, {}).get('La1', 0.0) or                             self.db.lineE.get(Z, {}).get('La_', 0.0)
                    if sh == 3:
                        le = self.db.lineE.get(Z, {}).get('Ma1', 0.0)
                    if le and abs(le - E) * 1000.0 < fw:
                        cands.append((abs(le - E), '%s %s' % (self.db.sym[Z], tag)))
            cands.sort()
            out.append((E, ex, sg, [c for _, c in cands[:4]]))
        return out

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
            self.quantify_basis = (
                'bootstrapped: one layer, its matrix the fitted K-shell '
                'areas, because %s' % (
                    'the sample model is set to bootstrap'
                    if self.yield_set is not None else
                    (self.yield_search or 'no .yield file is known for this '
                     'sample')))
        else:
            self.quantify_basis = 'sample model: one layer, typed matrix'
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
        self._conc_yields = None
        return out

    # -- quantification from GeoPIXE .yield files -----------------------

    M_SUBSHELLS = ('M5', 'M4', 'Mz', 'M3', 'M2', 'M1')

    def _element_photons(self, Z, shell, m_subshells=M_SUBSHELLS):
        """(photons emitted, error) for one element and shell, or None.

        A component's lines carry efficiency relative to its strongest line
        (_apply_efficiency), so area / eff(major line) is the photons the
        element emitted in that shell. When the M subshells were fitted
        apart, only those in m_subshells are counted: a yield file that
        holds just the M5 + M4 share (the lab's 287424 files, whose
        provenance says so) must be divided into M5 + M4 photons only.
        """
        ar, er = self.areas(), self.errors()
        sym = self.db.sym[Z]
        if shell == 3 and any(sym + s in ar for s in self.M_SUBSHELLS):
            subs = self.db.m_subshell_lines(Z)
            parts = [(sym + s, subs.get(s)) for s in m_subshells]
        else:
            parts = [(sym + SHELL_SUFFIX[shell], self.db.line_list(Z, shell))]
        ph, var, found = 0.0, 0.0, False
        for name, ll in parts:
            if name not in ar or not ll:
                continue
            e0 = max(ll, key=lambda t: t[1])[0]
            eff = float(self.efficiency(e0))
            if not eff > 0:
                continue
            found = True
            ph += ar[name] / eff
            var += (er.get(name, 0.0) / eff) ** 2
        return (ph, float(np.sqrt(var))) if found else None

    def quantify_yields(self, region=None):
        """Concentrations from the sample's GeoPIXE .yield files.

        What GeoPIXE does, once per file: every element is quantified from
        the UNKNOWN layer's column (pixe_fit.pro:414), conc = area / (charge
        x solid angle x efficiency x Y) (array_yield.pro:128, pixe_fit.pro:
        767). That is right only for elements that live in the UNKNOWN layer,
        so each file reports just those - Z in that layer's composition - and
        an element that lives in two layers (Sn in SnO2 and ITO) gets a row
        from each.

        So the COLUMN and its normalisation are GeoPIXE's; the photons are
        scrappyFIT's. Y in a GeoPIXE file is the MAJOR line's yield, the
        other lines' share in its relative intensities, so Y x
        sum(intensities) is the shell's total yield, and it is divided into
        the photons the whole shell emitted (_element_photons). GeoPIXE
        divides its major line's area by Y x eff(major) instead. Both handle
        efficiency exactly; they agree when scrappyFIT's major-line fraction
        (database line table) equals 1 / sum(the file's intensities), which
        are absorption-weighted and come from whatever line table made the
        file. Expect differences where those differ most: L lines under an
        overlayer, M shells.

        A file whose .provenance.json says its Pb M yield counts only the
        M5 + M4 share gets M5 + M4 photons; otherwise every fitted M
        subshell is counted.

        An element in several UNKNOWN layers gets a row from each, and each
        row gives the WHOLE signal to that one layer: they are alternatives,
        not parts to add (O in glass, ITO and SnO2 - nearly all O K comes
        from the glass). The note on those rows says so.

        ppm is by weight IN THAT LAYER. Charge is the entered charge times
        the live fraction, times the masked fraction of the raster when a
        region mask is applied - which assumes the dose per pixel is uniform
        over a fully scanned raster (true of 287424: the RBS dose is flat to
        1 %, and the lab's per-region numbers were made the same way).
        Without charge and solid angle the values are ppm x uC x msr: ratios
        within one layer still hold.

        Returns [dict(element, Z, shell, layer, region, ppm, err, file,
        note)], and sets self.quantify_basis.
        """
        if self._fit is None:
            raise RuntimeError('fit first')
        if self.efficiency is None:
            raise RuntimeError('load a detector efficiency curve first; '
                               'without it areas cannot become concentrations')
        ys = self.yield_set
        if ys is None or not len(ys):
            raise RuntimeError('no .yield files for this sample')
        region = self.yield_region if region is None else region
        if region is None and len([r for r in ys.regions if r]) > 1:
            raise RuntimeError(
                'the yield files cover regions %s: choose one in Sample model'
                % ', '.join(r for r in ys.regions if r))
        entries = ys.entries(region)
        if not entries:
            raise RuntimeError('no .yield file for region %r' % region)
        g = self.geometry
        frac = 1.0
        if self.mask is not None:
            m = np.asarray(self.mask, bool)
            frac = float(m.sum()) / float(m.size) if m.size else 1.0
        absolute = g.complete()
        q = (g.charge_uC * g.live_fraction * frac) if absolute else 1.0
        om = g.solid_angle_msr if absolute else 1.0
        self._yield_charge = (q if absolute else None, frac)
        rows, used, homeless = [], [], {}
        for e in entries:
            try:
                d = e.data
            except Exception as ex:
                raise RuntimeError('cannot read %s: %s' % (e.name, ex))
            if _yieldset.is_relative_units(e.path):
                raise RuntimeError(
                    '%s is in RELATIVE units (its .provenance.json says so) '
                    'and cannot give a concentration' % e.name)
            k = e.unknown - 1
            home = set(e.unknown_layer['Z'])
            msubs = (('M5', 'M4') if _yieldset.m54_only(e.path)
                     else self.M_SUBSHELLS)
            nk = d['nk']
            inten = np.asarray(d['intensity'], float).reshape(d['n'], nk)
            n_used = 0
            for j, (Z, sh) in enumerate(zip(d['z2'], d['shell'])):
                Z, sh = int(Z), int(sh)
                if Z not in self.db.sym:
                    continue
                if Z not in home:
                    homeless.setdefault((Z, sh), e)
                    continue
                sym = self.db.sym[Z]
                if (sh == 3 and msubs != self.M_SUBSHELLS
                        and sym + 'M' in self.areas()
                        and not any(sym + x in self.areas()
                                    for x in self.M_SUBSHELLS)):
                    # the file holds the M5 + M4 share and the fit lumped
                    # the whole M shell: no photon count matches the yield
                    rows.append(dict(
                        element=sym, Z=Z, shell=sh, layer=e.layer,
                        region=e.region, file=e.path, ppm=float('nan'),
                        err=float('nan'),
                        note='file holds the M5+M4 share only; fit the M '
                             'subshells apart (split_m_subshells)'))
                    n_used += 1
                    continue
                pe = self._element_photons(Z, sh, msubs)
                if pe is None:
                    continue
                yt = float(d['yield'][k, j]) * float(
                    inten[j, :max(int(d['n_lines'][j]), 1)].sum())
                row = dict(element=self.db.sym[Z], Z=Z, shell=sh,
                           layer=e.layer, region=e.region, file=e.path,
                           ppm=float('nan'), err=float('nan'), note='')
                if not yt > 0:
                    row['note'] = 'zero yield in the UNKNOWN layer'
                else:
                    row['ppm'] = pe[0] / (q * om * yt)
                    row['err'] = pe[1] / (q * om * yt)
                rows.append(row)
                n_used += 1
            used.append((e, n_used))
        # A fitted element that lives in none of the UNKNOWN layers has no
        # file to give its concentration; GeoPIXE would still print one from
        # the wrong column. Say so rather than drop it.
        done = {(r['Z'], r['shell']) for r in rows}
        for (Z, sh), e in sorted(homeless.items()):
            if (Z, sh) in done or self._element_photons(Z, sh) is None:
                continue
            rows.append(dict(element=self.db.sym[Z], Z=Z, shell=sh, layer='-',
                             region=e.region, file='', ppm=float('nan'),
                             err=float('nan'),
                             note='in no UNKNOWN layer of these files: not '
                                  'quantified'))
            done.add((Z, sh))
        layers_of = {}
        for r in rows:
            if r['file']:
                layers_of.setdefault((r['Z'], r['shell']), []).append(
                    r['layer'])
        for r in rows:
            ls = layers_of.get((r['Z'], r['shell']), [])
            if len(ls) > 1 and not r['note']:
                r['note'] = ('also in %s: each row gives it ALL the signal; '
                             'quote one layer' % ', '.join(
                                 x for x in ls if x != r['layer']))
        self._conc_yields = rows
        self._conc = None           # the wt% of a one-layer QUANTIFY is stale
        self.quantify_basis = (
            'yield files (%s%s): %s' % (
                ys.describe(region), '' if absolute else
                '; no charge or solid angle, so ppm x uC x msr',
                ', '.join('%s [%d element%s]' % (e.name, n,
                                                 '' if n == 1 else 's')
                          for e, n in used)))
        return rows

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

    def estimate_resolution(self, apply=True):
        """Initial noise and Fano from the strongest peak in the data.

        The width parameters are in CHANNELS, so a value tuned for one gain
        is wrong for another - 32.7 channels of noise means 55 eV at
        1.7 eV/ch and 150 eV at 4.6 eV/ch. Starting the refinement from the
        wrong regime leaves it stuck: on run 354016 the LEPIXE defaults gave
        chi2 43.6 with the width barely moving, and a start near the right
        value gave 6.8.

        Measures the FWHM of the tallest peak and seeds noise from it, with
        Fano at a fraction that keeps the energy dependence sane. Returns
        (noise, fano, fwhm_eV). Nothing is applied when the spectrum is
        empty or the peak is too weak to measure.
        """
        if self.spectrum is None:
            return None
        y = np.asarray(self.spectrum, float)
        a, b = self.cal
        i = int(np.argmax(y))
        if y[i] < 50:
            return None
        half = y[i] / 2.0
        lo = i
        while lo > 0 and y[lo] > half:
            lo -= 1
        hi = i
        while hi < len(y) - 1 and y[hi] > half:
            hi += 1
        fw = float(hi - lo)
        if fw < 2:
            return None
        E = a * i + b
        # fwhm_ch^2 = noise^2 + fano^2 (E - e_low); split it as GeoPIXE's
        # detector defaults do, with the Fano term about a quarter of the
        # width at this energy
        fano = 0.5 * fw / max(np.sqrt(max(E - self.options.e_low, 0.5)), 1.0)
        noise2 = fw * fw - fano * fano * max(E - self.options.e_low, 0.0)
        noise = float(np.sqrt(max(noise2, 1.0)))
        if apply:
            self.options.noise, self.options.fano = noise, fano
            self.invalidate()
        return noise, fano, 1000.0 * fw * a

    def live_seconds(self):
        """Acquisition duration in seconds, from the LMF clocks. None if the
        file does not carry them - and then the pile-up cap cannot be set."""
        if getattr(self, '_live_s', None):
            return float(self._live_s)
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
