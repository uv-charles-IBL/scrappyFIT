"""Fitted numbers must not depend on the BLAS thread count (BUGS.md 2026-10-20)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

CHILD = r'''
import sys, hashlib
sys.path.insert(0, sys.argv[1])
import numpy as np
from scrappyfit.fitting.fit import linear_step

NCOMP, NPROF, NCH = 50, 4096, 3000
rng = np.random.default_rng(0)
x = np.arange(NPROF, dtype=float)


class Comp:
    def __init__(self, i, mu, sig, group=None, frac=None):
        self.name = 'c%d' % i
        self._p = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        self._p /= self._p.sum()
        if group is not None:
            self.prior_group = group
            self.prior_fraction = frac

    def profile(self, pars, n):
        return self._p[:n].copy()


# strongly overlapping Gaussians: centres 55 channels apart, FWHM ~ 100
mus = 100 + 55.0 * np.arange(NCOMP) + rng.uniform(-10, 10, NCOMP)
sigs = rng.uniform(35, 60, NCOMP)
comps = []
for i in range(NCOMP):
    if i in (10, 11):
        comps.append(Comp(i, mus[i], sigs[i], 'g1', 0.4 if i == 10 else 0.6))
    elif i in (20, 21, 22):
        comps.append(Comp(i, mus[i], sigs[i], 'g2', (0.2, 0.3, 0.5)[i - 20]))
    else:
        comps.append(Comp(i, mus[i], sigs[i]))

true = rng.uniform(2e3, 3e4, NCOMP)
bg = np.full(NPROF, 5.0)
mean = sum(t * c.profile(None, NPROF) for t, c in zip(true, comps)) + bg
counts = rng.poisson(mean).astype(float)
channels = np.arange(100, 100 + NCH)

areas, errors, chi2, ndf, A = linear_step(counts, bg, comps, None, channels)
h = hashlib.md5(areas.tobytes() + errors.tobytes()).hexdigest()
print(h, float(chi2).hex(), float(areas[0]).hex(), float(errors[0]).hex())
'''


def _run(threads):
    env = dict(os.environ)
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[k] = str(threads)
    r = subprocess.run([sys.executable, '-I', '-c', CHILD, str(ROOT)],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_linear_step_is_identical_for_1_and_16_blas_threads():
    """A 50-component linear_step gave different bits at 1 and 16 BLAS
    threads on the lab desktop (2026-10-21). On a machine with few cores
    OpenBLAS may run fewer threads than asked, and then this passes without
    exercising much; it is a guard for the lab desktop."""
    one = _run(1)
    many = _run(16)
    assert one == many, 'thread-count dependent result:\n 1 thread : %s\n 16 threads: %s' % (one, many)


def test_inv_small_matches_pinv_with_empty_components():
    """The quartz recipe's Gram matrix has two all-zero rows and columns;
    np.linalg.pinv gave them zero variance and so must _inv_small."""
    from scrappyfit.fitting.fit import _inv_small
    rng = np.random.default_rng(1)
    X = rng.random((200, 9)) * np.logspace(0, 4, 9)
    X = np.insert(X, [3, 7], 0.0, axis=1)
    M = X.T @ X
    got = _inv_small(M)
    ref = np.linalg.pinv(M)
    live = [i for i in range(M.shape[0]) if M[i, i] != 0]
    Mk = M[np.ix_(live, live)]
    # pinv works by SVD on the unscaled matrix, so its error is normwise:
    # about cond x eps x max|inverse|. With columns scaled 1 to 1e4 the
    # inverse's entries span about 1e8, so an elementwise rtol cannot bound
    # it on small entries; the comparison with rtol 1e-7 failed on CI's Linux
    # Python 3.10 leg (master 0ae6fb1, 2026-10-21). Compare normwise instead.
    bound = 100 * np.linalg.cond(Mk) * np.finfo(float).eps * np.abs(ref).max()
    assert np.abs(got - ref).max() <= bound
    # The accuracy check that needs no reference inverse: M x inv = I on the
    # live block. On this test's matrix (lab desktop, 2026-10-21) it is
    # 1.8e-12 for _inv_small and 2.0e-9 for pinv; the bound is for this matrix.
    gk = got[np.ix_(live, live)]
    assert np.abs(Mk @ gk - np.eye(len(live))).max() < 1e-9
    assert not got[:, [3, 8]].any() and not got[[3, 8], :].any()


def test_inv_small_raises_on_a_singular_matrix():
    from scrappyfit.fitting.fit import _inv_small
    M = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        _inv_small(M)


def test_least_squares_runs_on_one_blas_thread(monkeypatch):
    """scipy's trf multiplies the Jacobian through BLAS, which gave the
    quartz chi2r different seventh digits at 1, 4 and 16 threads
    (2026-10-21). fit_spectrum must hold every BLAS to one thread around it."""
    import scipy.optimize
    from threadpoolctl import threadpool_info

    from scrappyfit import config
    from scrappyfit.session import Session
    from labdata import SYNTHETIC
    if config.database_path(required=False) is None:
        pytest.skip('no GeoPIXE database available')

    seen = []

    class Seen(Exception):
        pass

    def spy(*args, **kw):
        seen.append([i['num_threads'] for i in threadpool_info()
                     if i['user_api'] == 'blas'])
        raise Seen()            # fit_spectrum keeps its start; the fit ends fast

    monkeypatch.setattr(scipy.optimize, 'least_squares', spy)
    s = Session()
    s.load(str(SYNTHETIC))
    s.set_calibration(0.0016984, -0.37225)
    s.run_fit(['O', 'Si'])
    assert seen and all(n and set(n) == {1} for n in seen), seen


def test_one_blas_thread_is_held_until_the_last_fit_ends():
    """Two fits at once: the first to finish must not lift the limit under
    the other (reviewer, 2026-10-21)."""
    import scipy.optimize  # noqa: F401  (load scipy's OpenBLAS too)
    from threadpoolctl import threadpool_info
    from scrappyfit.fitting import fit

    def blas():
        return {i['num_threads'] for i in threadpool_info()
                if i['user_api'] == 'blas'}
    before = blas()
    with fit._one_blas_thread():
        with fit._one_blas_thread():
            assert blas() == {1}
        assert blas() == {1}
    assert blas() == before


def test_missing_threadpoolctl_is_an_error(monkeypatch):
    """Without threadpoolctl a fit must fail, not silently skip its
    non-linear refinement (the optimiser's except Exception would hide it)."""
    from scrappyfit.fitting import fit
    monkeypatch.setattr(fit, '_BLAS', None)
    monkeypatch.setitem(sys.modules, 'threadpoolctl', None)
    with pytest.raises(ImportError):
        fit._blas_controller()
