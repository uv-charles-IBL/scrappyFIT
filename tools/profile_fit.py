"""Where does a fit's time go? cProfile one golden recipe and break it down.

usage: python -I tools/profile_fit.py <case> [--json out.json]

case is a key of tests/test_golden_spectra.py's RECIPES (quartz,
287424_perovskite, 404001_mnox, 404003_ga2o3, 404004_sic). Needs the lab
spectra (tests/labdata.py). Prints the profiled wall time, the fitted chi2r,
then calls / self / cumulative seconds for the pieces of the fit loop:
least_squares and its residual evaluations, the linear step, the profile
builds (element lines, sum peaks, pile-up plateau, shelves) and the small
solves. "profile builds" is the sum of the three component rows
(element lines, sum peaks, plateau), which do not call each other. The
line_profile, de_shelf, Ep and erfc_saturated rows are pieces of those and
overlap them (the sum peaks call line_profile too): never add them to the
component rows. cProfile slows the fit; use tools/bench_recipes.py for
timings.

Written 2026-10-21 (fix/fit-loop-speed, step 0).
"""
import argparse
import cProfile
import json
import os
import pstats
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# label -> (file suffix, function name, first line or None)
ROWS = [
    ('run_fit', 'session.py', 'run_fit', None),
    ('fit_spectrum', 'fit.py', 'fit_spectrum', None),
    ('least_squares', 'least_squares.py', 'least_squares', None),
    ('resid (one LM evaluation)', 'fit.py', 'resid', None),
    ('linear_step', 'fit.py', 'linear_step', None),
    ('_linear_step_free', 'fit.py', '_linear_step_free', None),
    ('Component._profile_of', 'fit.py', '_profile_of', None),
    ('SumPeakComponent.profile', 'fit.py', 'profile', 'SumPeakComponent'),
    ('PileupPlateau.profile', 'fit.py', 'profile', 'PileupPlateauComponent'),
    ('peakshape.line_profile', 'peakshape.py', 'line_profile', None),
    ('hypermet.line_profile', 'hypermet.py', 'line_profile', None),
    ('de_shelf', 'peakshape.py', 'de_shelf', None),
    ('de_shelf.Ep', 'peakshape.py', 'Ep', None),
    ('erfc_saturated', 'peakshape.py', 'erfc_saturated', None),
    ('_gram', 'fit.py', '_gram', None),
    ('_inv_small', 'fit.py', '_inv_small', None),
    ('_solve_small', 'fit.py', '_solve_small', None),
    ('_matvec', 'fit.py', '_matvec', None),
    ('nnls', '_nnls.py', 'nnls', None),
    ('snip', 'background.py', 'snip', None),
    ('low_stats_filter', 'lowstats.py', 'low_stats_filter', None),
]
BUILDS = ('Component._profile_of', 'SumPeakComponent.profile',
          'PileupPlateau.profile')


def _class_lines():
    """First line of each class in fitting/fit.py, so the several .profile
    methods can be told apart."""
    import inspect
    from scrappyfit.fitting import fit
    out = {}
    for name in ('SumPeakComponent', 'PileupPlateauComponent'):
        cls = getattr(fit, name)
        src, start = inspect.getsourcelines(cls)
        out[name] = (start, start + len(src))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--json')
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    import numpy as np
    import scipy
    import test_golden_spectra as G

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _, res = G.RECIPES[args.case]()
    pr.disable()
    wall = time.perf_counter() - t0

    spans = _class_lines()
    rows = {}
    for (f, line, name), (_, nc, tt, ct, _) in pstats.Stats(pr).stats.items():
        for label, suffix, fn, cls in ROWS:
            if not (f.endswith(os.sep + suffix) and name == fn):
                continue
            if cls is not None and not spans[cls][0] <= line < spans[cls][1]:
                continue
            r = rows.setdefault(label, [0, 0.0, 0.0])
            r[0] += nc
            r[1] += tt
            r[2] += ct
    print('%s  %s  numpy %s scipy %s' % (args.case, time.strftime('%Y-%m-%d %H:%M'),
                                         np.__version__, scipy.__version__))
    print('profiled wall %.1f s, chi2r %.6f' % (wall, res.reduced_chi2))
    print('%-28s %9s %9s %9s' % ('', 'calls', 'self s', 'cum s'))
    for label, *_ in ROWS:
        if label in rows:
            print('%-28s %9d %9.2f %9.2f' % (label, *rows[label]))
    print('profile builds (cum, %s): %.2f s'
          % (' + '.join(BUILDS), sum(rows.get(k, [0, 0, 0])[2] for k in BUILDS)))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'case': args.case, 'date': time.strftime('%Y-%m-%d %H:%M'),
                       'wall_profiled': wall, 'chi2r': res.reduced_chi2,
                       'rows': rows}, fh, indent=1)


if __name__ == '__main__':
    main()
