"""Time the golden recipes, interleaved, and fingerprint their results.

usage: python -I tools/bench_recipes.py [--repeat N] [--cases a,b,...]
                                         [--other DIR] [--json out.json]

--other DIR times a second checkout (say master, from `git worktree add`)
interleaved with this one, fit by fit, so a before/after shares the
machine's state; both read the lab spectra the usual way (tests/labdata.py,
or SCRAPPYFIT_LAB_SPECTRA).

Each fit runs in its own subprocess (so no run inherits another's caches),
and the repeats are interleaved - case 1, 2, ... 5, then again - so a swing
in the machine's background load lands on every case rather than on one.
The time is the whole recipe (tests/test_golden_spectra.py run_case:
session set-up, database parse, file loads, first-use lazy imports and every
fit), not interpreter start-up; each subprocess pays the same fixed part.
For each case it prints min / median of the N times, chi2r, and an md5 of
everything run_case returns (chi2r, pars[:4], every area and error, as
repr, which round-trips floats exactly). The same hash before and after
means bit-identical results; a different hash is NOT a golden failure (it
covers components the pins leave free) - run the golden tests for that.

Needs the lab spectra (tests/labdata.py). Written 2026-10-21
(fix/fit-loop-speed). This desktop's speed drifts: the same code profiled
quartz at 5.3 s on 2026-10-21 22:46 and 11.9 s on 2026-10-22 morning
(tools/profile_fit.py), so a before/after is only fair as one interleaved
run of both.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASES = ['quartz', '287424_perovskite', '404001_mnox', '404003_ga2o3',
         '404004_sic']

CHILD = r'''
import hashlib, json, sys, time
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tests)r)
import test_golden_spectra as G
import scrappyfit
t0 = time.perf_counter()
out = G.run_case(%(case)r)
dt = time.perf_counter() - t0
blob = repr((out['chi2r'], out['pars'], sorted(out['areas'].items()),
             sorted(out['errors'].items())))
print(json.dumps({'t': dt, 'chi2r': out['chi2r'],
                  'md5': hashlib.md5(blob.encode()).hexdigest(),
                  'pkg': scrappyfit.__file__}))
'''


def run_one(case, root=ROOT):
    code = CHILD % {'root': root, 'tests': os.path.join(root, 'tests'),
                    'case': case}
    p = subprocess.run([sys.executable, '-I', '-c', code], cwd=root,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError('%s failed:\n%s' % (case, p.stderr[-2000:]))
    out = json.loads(p.stdout.strip().splitlines()[-1])
    # an installed scrappyfit (an editable install's finder) would win over
    # sys.path, and both sides would time the same code
    if not os.path.normcase(os.path.abspath(out['pkg'])).startswith(
            os.path.normcase(os.path.abspath(root))):
        raise RuntimeError('%s imported %s, not the checkout %s'
                           % (case, out['pkg'], root))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--other')
    ap.add_argument('--json')
    args = ap.parse_args()
    cases = args.cases.split(',')
    roots = [('this', ROOT)] + ([('other', os.path.abspath(args.other))]
                                if args.other else [])
    start = time.strftime('%Y-%m-%d %H:%M')
    # what is being run, read before the runs, not after
    def state(root):
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--',
                                'scrappyfit', 'tests'], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return (head or '?') + (' + uncommitted' if dirty else '')
    heads = {name: state(root) for name, root in roots}
    runs = {(name, c): [] for name, _ in roots for c in cases}
    for rep in range(args.repeat):
        for c in cases:
            # alternate which side goes first, so neither always runs warm
            for name, root in (roots if rep % 2 == 0 else roots[::-1]):
                runs[name, c].append(run_one(c, root))
    print('bench_recipes  %s to %s  repeat %d  OMP_NUM_THREADS=%s'
          % (start, time.strftime('%H:%M'), args.repeat,
             os.environ.get('OMP_NUM_THREADS')))
    for name, root in roots:
        print('  %-5s %s  HEAD %s' % (name, root, heads[name]))
    print('%-20s %-5s %8s %8s %10s  %s' % ('case', '', 'min s', 'median s',
                                            'chi2r', 'md5'))
    summary = {}
    for c in cases:
        for name, _ in roots:
            rs = runs[name, c]
            ts = [r['t'] for r in rs]
            hashes = sorted({r['md5'] for r in rs})
            summary.setdefault(name, {})[c] = {
                'times': ts, 'min': min(ts), 'median': statistics.median(ts),
                'chi2r': rs[0]['chi2r'], 'md5': hashes}
            print('%-20s %-5s %8.2f %8.2f %10.6f  %s' % (
                c, name, min(ts), statistics.median(ts), rs[0]['chi2r'],
                ' '.join(hashes)))
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'start': start, 'heads': heads, 'cases': summary}, fh,
                      indent=1)


if __name__ == '__main__':
    main()
