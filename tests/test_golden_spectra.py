"""Golden regression tests: five reference spectra, fitted with the recipe
the analysis that produced them used, and the results pinned.

These are CHARACTERIZATION tests. They lock in what the code does today,
not what is physically true. A failure means the fit changed; whether the
change is a fix or a regression is a judgement for the reviewer. Values we
already know to be wrong are marked inline with where that is recorded.

Re-pinning a value is allowed only with a CHANGELOG line saying which
commit moved it and why. The failure message prints every current value.

All five spectra are loaded from two-column text, so the session has no
run duration and the pile-up cap is off (``Session.pileup_cap`` returns
None). Fits from the original .lmf files will not match these numbers
exactly for that reason.

Pinned 2026-10-18 from master 6c0718b with this file's recipes. Re-pinned
2026-10-21 on fix/blas-threads (thread-independent fits): 287424 in full,
404003 in full, 404004 Fano; quartz and 404001 unchanged (CHANGELOG).
"""
import math
import os
import pathlib

import pytest

from scrappyfit import config
from scrappyfit.session import Session

from labdata import lab_spectrum, missing_lab_data

DATA = pathlib.Path(__file__).parent / 'data'
DET = DATA / 'Amptek_25_C1_deflector.detector'
RESP = (pathlib.Path(__file__).parents[1] / 'scrappyfit' / 'resources'
        / 'response' / 'amptek_c1_25mm2_20260923.json')

# Machine-specific, in part. The 2026-10-18 pins reproduced only on the lab
# desktop: 287424 gave chi2r 2.3259 there and 2.3816 on Linux and a GitHub
# Windows runner (CI run 39296724668). That was the desktop's 16 BLAS threads:
# with one thread it gives 2.3816 too. Since fix/blas-threads (2026-10-21) a
# fit does not depend on the thread count, and 287424, quartz and 404001
# reproduce across numpy 1.26 / 2.5 on that desktop. 404003 and 404004 still do
# not: their widths move about 0.2 % (PARS_RTOL is 0.1 %) with last-bit changes
# such as another numpy or CPU (BUGS.md, 2026-10-21; lab knowledge base,
# findings/scrappyfit-scipy-version-sensitivity.md). On another machine a
# failure of those two says nothing about the code. CI sets CI=true and skips;
# SCRAPPYFIT_RUN_GOLDEN=1 runs them anyway. The five spectra are lab data and
# not in the public repository (tests/labdata.py says where they are looked
# for); without them every case skips.
_ON_CI = bool(os.environ.get('CI')) and not os.environ.get('SCRAPPYFIT_RUN_GOLDEN')

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(config.database_path(required=False) is None,
                       reason='no GeoPIXE database available'),
    pytest.mark.skipif(_ON_CI, reason=(
        'golden spectra are lab data, pinned on the desktop they were made '
        'on; 404003/404004 widths move ~0.2 % on other machines and library '
        'versions (BUGS.md, 2026-10-21). Run them locally before merging; '
        'SCRAPPYFIT_RUN_GOLDEN=1 forces them here.')),
]

AREA_RTOL = 0.005       # 0.5 % of the area ...
AREA_SIGMA = 0.5        # ... or half its fit sigma, whichever is larger
CHI2_RTOL = 0.005
PARS_RTOL = 1e-3        # refined noise (eV), Fano, and the two calibration terms


# --------------------------------------------------------------- recipes

def _spectrum(fname):
    """The lab spectrum, or the tests/data path _needed() reports missing."""
    return lab_spectrum(fname) or DATA / fname


def _deflector_session(spec, cal, e_low, e_high):
    s = Session()
    s.load(str(_spectrum(spec)))
    s.set_calibration(*cal)
    s.load_efficiency([p for p in s.builtin_efficiencies()
                       if 'deflector' in p.lower()][0])
    s.load_detector(str(DET))
    s.options.e_low, s.options.e_high = e_low, e_high
    return s


# 28 Sep 2026 runs, recipe of the lab script le20260928/fit_all.py:
# a GeoPIXE-shape seed fit, then the measured response (E+) started from it.
CAL_20260928 = (0.0019756398864463256, -0.4522001604198639)   # cal.json 'C'

EPLUS = {
    '404001_mnox': (['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'K',
                     'Ca', 'Mn', (25, 2), 'Fe', (29, 2)], 0.17, 7.5, {'Mn'},
                    False),
    # Na, Cu L, Ti L and F dropped: proxies for Ga L / O K misfits
    # (lab knowledge base, findings/poster-maps-sep2026.md)
    '404003_ga2o3': (['C', 'N', 'O', 'Al', 'Si', 'Ti', (31, 2), (79, 3)],
                     0.17, 6.0, {'Al'}, False),
    # one strong line: the calibration is held, free cal drifts (BUGS.md)
    '404004_sic': (['C', 'N', 'O', 'Al', 'Si'], 0.17, 6.0, {'Si'}, True),
}


def _fit_eplus(name):
    els, elo, ehi, kb, calfix = EPLUS[name]

    def base():
        s = _deflector_session(name + '.txt', CAL_20260928, elo, ehi)
        o = s.options
        o.use_escape_step = o.use_contact_step = o.use_window_step = False
        if calfix:
            o.refine = tuple(x for x in o.refine if x != 'cal')
        return s

    g = base()
    g.estimate_resolution()
    P = g.run_fit(els).pars
    s = base()
    o = s.options
    s.load_response(str(RESP))
    o.response_scale = True
    o.free_kbeta = kb
    o.subthreshold_pileup_kev = 0.5
    o.fast_threshold_kev = 0.5
    if not calfix:
        eoc = 0.7 * elo + 0.3 * ehi
        s.set_calibration(1.0 / P.a[3], eoc - P.a[2] / P.a[3])
    o.noise, o.fano = float(P.a[0]), float(P.a[1])
    return s, s.run_fit(els)


def _fit_287424():
    # recipe of the lab script pvsk287424/final_20261006_rbs/final287424.py
    s = _deflector_session('287424_perovskite.txt', (0.0016984, -0.37225),
                           0.2, 6.5)
    spec = ('C,N,O,Na,Mg,Al,Si,Cl,K,Ca,BrL,AgL,InL,SnL,IL,PbM,'
            'AgM,InM,SnM,IM,CsM,CsL')
    els = []
    for tok in spec.split(','):
        sh = 1
        if tok[-1] in 'LM' and len(tok) > 1 and tok[:-1].lower() in s.db.z:
            sh = 2 if tok[-1] == 'L' else 3
            tok = tok[:-1]
        els.append((s.db.z[tok.lower()], sh))
    return s, s.run_fit(els)


def _fit_quartz():
    # the default-options path of test_smoke.test_fit_runs_and_is_reproducible
    s = Session()
    s.load(str(_spectrum('quartz.txt')))
    s.set_calibration(0.0016984, -0.37225)
    return s, s.run_fit(['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si'])


RECIPES = {
    '404001_mnox': lambda: _fit_eplus('404001_mnox'),
    '404003_ga2o3': lambda: _fit_eplus('404003_ga2o3'),
    '404004_sic': lambda: _fit_eplus('404004_sic'),
    '287424_perovskite': _fit_287424,
    'quartz': _fit_quartz,
}


def run_case(name):
    """Fit one reference spectrum; return everything that is pinned."""
    s, r = RECIPES[name]()
    return dict(chi2r=float(r.reduced_chi2),
                pars=[float(x) for x in r.pars.a[:4]],
                areas={k: float(v) for k, v in s.areas().items()},
                errors={k: float(v) for k, v in s.errors().items()})


# ---------------------------------------------------------- pinned values
# areas: name -> (area, fit sigma). Pinned: every component with area > 4
# sigma. Components the fit cannot determine are listed, not pinned - their
# areas move with any change and carry no information.
# pars: FitResult.pars.a[:4] after refinement (noise eV, Fano, cal terms).
# Kb components are in the unnormalised units of session.py build_components:
# Kb/Ka = area(Kb) / area(Ka) x the table ratio.
# Known-wrong values are marked WRONG with where that is recorded.
# Lab-knowledge paths are relative to projects/light-element-pixe in the lab's
# knowledge base (not public).

EXPECTED = {
    '404001_mnox': dict(
        chi2r=1.67484,
        pars=(38.6244, 24.4061, 1425.93, 507.198),
        areas={
            'MnKb':                 (1179209, 3361),
            'Mn':                   (1102443, 1251),
            # Area is what the code finds. Do not read O/Mn as stoichiometry:
            # it needs 0.27-0.61x of the efficiency file
            # (findings/poster-number-audit-sep2026.md,
            # findings/c1-efficiency-model-inconsistency.md).
            'O':                    (283806, 569.4),
            'MnL':                  (86304.44, 379.6),
            'pileup plateau':       (7947.612, 1489),
            'S':                    (3770.875, 79.16),
            'pileup':               (2264.105, 233.2),
            'Si':                   (1618.279, 63.5),
            'N':                    (1233.424, 83.06),
            'Al':                   (822.9609, 58.47),
            # WRONG: spurious Fe; the model's Fe Kb at 7.06 keV has no data
            # (lab notes le20260928/NOTES.md:18, then 2.8 k).
            'Fe':                   (770.8959, 176),
            'Mg':                   (754.5123, 57.53),
            'Na':                   (635.9658, 62.01),
            'C':                    (315.3498, 54.87),
            'Cl':                   (308.6311, 44.92),
        }),  # not pinned (area < 4 sigma): K
    '404003_ga2o3': dict(
        chi2r=1.35018,
        pars=(38.1036, 25.0451, 1201.62, 510.053),
        areas={
            # Al Kb/Ka = 2.524 x 0.013 = 0.033; the table's 0.013 is too low
            # for solids (BUGS.md "Si and Al Kb/Ka"), which is why Kb is free.
            'AlKb':                 (1363270, 4.655e+04),
            'Al':                   (555028.3, 873.8),
            'GaL':                  (125181.3, 394.3),
            'O':                    (65828.28, 272.4),
            'AuM5':                 (39112.67, 239),
            'AuM4':                 (22203.59, 203.8),
            'Ti':                   (7153.517, 85.9),
            'AuM3':                 (3764.421, 129.8),
            'AuMz':                 (2307.723, 124.4),
            'N':                    (1537.792, 71.55),
            'AuM2':                 (999.222, 107.2),
            'Si':                   (720.6023, 83.04),
        }),  # not pinned (area < 4 sigma): AuM1, C
    '404004_sic': dict(
        chi2r=1.9944,
        pars=(40.9898, 21.9466, 1200.22, 506.165),
        areas={
            # Si Kb/Ka = 1.578 x 0.0273 = 0.043, the measured solid value; the
            # table's 0.027 is WRONG for solids (BUGS.md "Si and Al Kb/Ka").
            'SiKb':                 (3414312, 2.037e+04),
            'Si':                   (2163862, 1555),
            # Si+Si pile-up is a flat-topped hump the sum model misfits
            # (methods/pixe-fitting-scrappyfit.md, sum_deficit_spread).
            'pileup':               (4510.373, 71.07),
            # C/Si implies 0.51x the file's C efficiency; corrected value
            # depends on omega_K (findings/c1-efficiency-model-inconsistency.md).
            'C':                    (2002.547, 66.46),
            'pileup plateau':       (1501.021, 87.17),
        }),  # not pinned (area < 4 sigma): Al
    '287424_perovskite': dict(
        # Relative regression only: the absolute scale on this run is biased
        # by aperture vignetting (RETRACTED.md R13, R15).
        chi2r=2.38158,
        pars=(35.9506, 26.911, 1449.75, 588.789),
        areas={
            'Si':                   (217090, 487),
            'IL':                   (34459.95, 210.5),
            'PbM5':                 (31083.23, 193.4),
            'AgL':                  (26946.47, 228.5),
            'Ca':                   (17966.84, 176.5),
            'PbM4':                 (15724.84, 149.3),
            'InL':                  (14742.33, 344.4),
            'BrL':                  (14115.8, 612.5),
            # Degenerate with Ag M at 0.31 keV (RETRACTED.md R3).
            'C':                    (9828.795, 586.6),
            'IM3':                  (8644.389, 701.4),
            'PbMz':                 (6299.316, 177.2),
            'IM2':                  (6092.393, 497.1),
            'Mg':                   (4544.464, 104.6),
            'Na':                   (4896.446, 566),
            'SnL':                  (4310.125, 202.5),
            'K':                    (3307.243, 227.6),
            'CsM1':                 (2691.509, 589.5),
            'PbM3':                 (2379.517, 127),
            'Cl':                   (318.4642, 79.27),
        }),  # not pinned (area < 4 sigma): O, IM5, SnM4, InM4, AgM2, N, AgM5,
             # InM3, AgM3, IM1, SnM1, pileup, and Al (1432 +- 525 since
             # fix/blas-threads; an instrument artefact at the Si Ka level, not
             # sample Al: RETRACTED.md R11 item 2, "claims NOT retracted").
             # O vs I M and N vs In/Sn M are the degeneracies of
             # RETRACTED.md R3 and the project README.
    'quartz': dict(
        chi2r=1.99083,
        pars=(36.2896, 27.4343, 1450.2, 588.789),
        areas={
            'Si':                   (1080876, 1037),
            'O':                    (870415, 926.8),
            'Al':                   (6627.971, 110.1),
            'Na':                   (3733.949, 92.62),
            'F':                    (3637.083, 95.16),
            'C':                    (1228.291, 84),
            'Mg':                   (805.7178, 67.17),
            'pileup':               (634.2358, 61.6),
            'N':                    (444.5433, 104.3),
        }),
}


# ------------------------------------------------------------------ tests

def _needed(name):
    files = [_spectrum('quartz.txt' if name == 'quartz' else name + '.txt')]
    if name != 'quartz':
        files.append(DET)
    if name in EPLUS:
        files.append(RESP)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        missing_lab_data(', '.join(missing))


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_golden_fit(name):
    _needed(name)
    exp = EXPECTED[name]
    got = run_case(name)
    report = 'current values for %s:\n%r' % (name, got)
    bad = []
    if not math.isclose(got['chi2r'], exp['chi2r'], rel_tol=CHI2_RTOL):
        bad.append('chi2r %.5g, pinned %.5g' % (got['chi2r'], exp['chi2r']))
    for i, (g, e) in enumerate(zip(got['pars'], exp['pars'])):
        if not math.isclose(g, e, rel_tol=PARS_RTOL):
            bad.append('pars.a[%d] %.6g, pinned %.6g' % (i, g, e))
    for k, (area, sigma) in exp['areas'].items():
        if k not in got['areas']:
            bad.append('%s missing from the fit' % k)
            continue
        tol = max(AREA_RTOL * abs(area), AREA_SIGMA * sigma)
        if abs(got['areas'][k] - area) > tol:
            bad.append('%s %.6g, pinned %.6g +- %.3g'
                       % (k, got['areas'][k], area, tol))
    assert not bad, '\n'.join(bad) + '\n\n' + report
