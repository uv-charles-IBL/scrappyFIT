"""Fit the five golden recipes on one checkout and dump every number bit-exactly.

usage: python -I tools/golden_dump.py <repo root> <out.json> [recipe ...]

Writes repr() of chi2, ndf, every area, error and shape parameter, plus a
sha256 of the full model and profile arrays, and the wall time per fit.
Run it on two checkouts (e.g. a git worktree of master) and compare the
JSON with --compare: any difference outside 'seconds' is a changed fit.

    python -I tools/golden_dump.py --compare a.json b.json

Written 2026-10-20 for fix/fit-speed, which must not move a single bit.
"""
import hashlib
import json
import os
import platform
import sys
import time


def _sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()


def dump(root, out, names):
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, 'tests'))
    import numpy as np
    import scipy
    import test_golden_spectra as g
    res = {'_env': {'python': sys.version.split()[0], 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'machine': platform.node(),
                    'date': time.strftime('%Y-%m-%d %H:%M'),
                    'threads': {k: os.environ.get(k) for k in
                                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                 'MKL_NUM_THREADS')}}}
    for name in names or list(g.RECIPES):
        t0 = time.perf_counter()
        s, r = g.RECIPES[name]()
        dt = time.perf_counter() - t0
        res[name] = {
            'seconds': round(dt, 3),
            'chi2': repr(float(r.chi2)), 'ndf': int(r.ndf),
            'names': list(r.names),
            'areas': [repr(float(a)) for a in r.areas],
            'errors': [repr(float(e)) for e in r.errors],
            'pars': [repr(float(x)) for x in r.pars.a],
            'model_sha': _sha(r.model), 'background_sha': _sha(r.background),
            'profiles_sha': _sha(r.profiles),
        }
        print('%-18s %7.2f s  chi2 %s' % (name, dt, res[name]['chi2']), flush=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    for name in sorted(set(A) & set(B) - {'_env'}):
        for k in A[name]:
            if k == 'seconds':
                continue
            if A[name][k] != B[name].get(k):
                bad += 1
                print('DIFF %s %s' % (name, k))
        print('%-18s %7.2f s -> %7.2f s' % (name, A[name]['seconds'],
                                            B[name]['seconds']))
    print('bit-identical' if not bad else '%d fields differ' % bad)
    return bad


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    dump(sys.argv[1], sys.argv[2], sys.argv[3:])
