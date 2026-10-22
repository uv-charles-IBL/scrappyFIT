"""Analytic (variable-projection) Jacobian of the shape-parameter fit.

BUGS.md: the shape-parameter Jacobian is by finite differences, one full linear
step per refined parameter. These tests pin the API that replaces it:

  peakshape.line_profile_jac(E, beta, pars, n, idxs, tail_amp, tail_len,
                             beta_tail, do_tail) -> (f, {i: df/da_i})
  <component>.profile_jac(pars, n, idxs)          -> (f, {i: df/da_i})
  fit.residual_jacobian(counts, background, components, pars, channels, idxs)
                                                   -> (r, J)

Each is checked against central finite differences of the existing profile /
residual code. The finite-difference step is 1e-7 relative: at 1e-6 a
cal-offset step (about 1.5e-3 channels) can move an integer window edge
inside the Gaussian core, which the "same nonzero channels" rule cannot see.
"""
import copy
import os

import numpy as np
import pytest

from scrappyfit.fitting import fit as F
from scrappyfit.fitting import hypermet as H
from scrappyfit.fitting import peakshape as PS
from scrappyfit.fitting.peakshape import ShapePars


RESP = os.path.join(os.path.dirname(__file__), '..', 'scrappyfit', 'resources',
                    'response', 'amptek_c1_25mm2_20260923.json')

N = 4096
E_LOW, E_HIGH = 0.2, 6.5
CAL = 0.0017                      # keV per channel
CAL_B = -0.05                     # keV at channel 0
SHAPE_IDXS = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12]


def _pars(seed, tail=True, shelf=True, contact=True, window=True):
    """Realistic ShapePars (about 95 eV noise term, 130 eV at Mn Ka), the
    seeded jitter keeps the sets different."""
    rng = np.random.default_rng(seed)
    eoc = 0.7 * E_LOW + 0.3 * E_HIGH
    p = ShapePars(55.0 * rng.uniform(0.8, 1.2), 22.0 * rng.uniform(0.8, 1.2),
                  (eoc - CAL_B) / CAL * rng.uniform(0.999, 1.001),
                  1.0 / CAL * rng.uniform(0.999, 1.001), 0.0, 0.0,
                  E_LOW, E_HIGH)
    if tail:
        p.a[5], p.a[6] = rng.uniform(0.6, 1.2, 2)
    if shelf:
        p.a[8], p.a[9] = rng.uniform(0.03, 0.08), rng.uniform(0.8, 1.5)
    if contact:
        p.a[11] = rng.uniform(0.02, 0.05)
        if p.a[9] == 0.0:
            p.a[9] = 1.0
    if window:
        p.a[12] = rng.uniform(0.01, 0.04)
        if p.a[9] == 0.0:
            p.a[9] = 1.0
    return p


def _shifted(p, i, delta):
    q = copy.deepcopy(p)
    q.a = np.array(p.a, dtype=float)
    q.a[i] += delta
    return q


def _compare(fn_jac, fn_f, p, idxs, rtol, state):
    """Analytic derivatives of fn_jac against central differences of fn_f.
    Returns the number of (i) actually compared; skips where the integer
    channel window changes inside the step."""
    f, d = fn_jac(p)
    assert np.array_equal(f, fn_f(p)), 'f must equal the plain profile exactly'
    assert set(d) >= set(idxs)
    for i in idxs:
        assert d[i].shape == f.shape
        h = 1e-7 * max(1.0, abs(p.a[i]))
        fp = fn_f(_shifted(p, i, +h))
        if i in (8, 11, 12) and p.a[i] == 0.0:
            # the term is off at exactly 0: kink, so the contract is the
            # one-sided (a_i -> 0+) derivative, compared by forward difference
            fd = (fp - f) / h
            r_i = max(rtol, 1e-4)
        else:
            fm = fn_f(_shifted(p, i, -h))
            if not np.array_equal(fp != 0, fm != 0):
                state['skipped'] += 1
                continue
            fd = (fp - fm) / (2 * h)
            r_i = rtol
        err = np.max(np.abs(d[i] - fd))
        tol = r_i * np.max(np.abs(fd)) + 1e-12
        assert err <= tol, (i, err, tol)
        state['checked'] += 1
    return state


def _need_checks(state, minimum):
    assert state['checked'] >= minimum, state     # not everything skipped


# --- 1. line_profile_jac -----------------------------------------------------

@pytest.mark.parametrize('E', [0.525, 1.74, 5.9])
def test_line_profile_jac_matches_central_differences(E):
    """line_profile_jac must return line_profile exactly plus d f/d a_i."""
    state = {'checked': 0, 'skipped': 0}
    cases = [dict(tail=t, shelf=s, contact=c, window=w)
             for t in (True, False) for s in (True, False)
             for c in (True, False) for w in (True, False)]
    for seed, flags in enumerate(cases):
        p = _pars(seed, **flags)
        for do_tail in (True, False):
            kw = dict(tail_amp=0.08 if flags['tail'] else 0.0,
                      tail_len=0.9 if flags['tail'] else 0.0,
                      beta_tail=1.3, do_tail=do_tail)
            _compare(lambda q: PS.line_profile_jac(E, 0.7, q, N, SHAPE_IDXS,
                                                   **kw),
                     lambda q: PS.line_profile(E, 0.7, q, N, **kw),
                     p, SHAPE_IDXS, 1e-5, state)
    _need_checks(state, 100)


# --- 2. component profile_jac ------------------------------------------------

def _tail_fns():
    return dict(tail_amp_fn=lambda E: 0.04 + 0.01 * E,
                tail_len_fn=lambda E: 0.6 + 0.05 * E)


def _components():
    return [F.Component('O', [(0.525, 1.0)], **_tail_fns()),
            F.Component('Na', [(1.041, 1.0), (1.071, 0.05)], **_tail_fns()),
            F.Component('Si', [(1.740, 1.0), (1.836, 0.03)], **_tail_fns()),
            F.Component('Fe', [(6.404, 1.0), (7.058, 0.13)], **_tail_fns())]


def _check_component(comp, idxs, seeds, rtol, minimum):
    state = {'checked': 0, 'skipped': 0}
    for seed in seeds:
        p = _pars(seed)
        _compare(lambda q: comp.profile_jac(q, N, idxs),
                 lambda q: comp.profile(q, N), p, idxs, rtol, state)
    _need_checks(state, minimum)


def test_component_profile_jac_with_lines_and_tails():
    """Component.profile_jac: several lines, tail functions, shelf terms."""
    comp = _components()[2]
    comp.lines = comp.lines + _components()[1].lines
    comp.n_own = len(comp.lines)
    _check_component(comp, SHAPE_IDXS, range(4), 1e-5, 20)


def test_component_profile_jac_with_empirical_response():
    """Component.profile_jac with .response set, incl. the a5/a6 scale."""
    comp = _components()[2]
    comp.response = H.EmpiricalResponse.load(RESP)
    idxs = [0, 1, 2, 3, 5, 6]
    p = _pars(0)
    p.a[5], p.a[6] = 0.9, 1.0
    ft, _, fs, _ = comp.response.params(1.74)
    assert ft * 0.81 + fs < 0.95, 'would hit the 0.95 cap kink'
    _check_component(comp, idxs, [0, 1, 2], 1e-6, 10)



# --- 2b. hypermet.line_profile_jac (empirical response) ----------------------

def _synth_resp(sloped, al=False):
    """Synthetic EmpiricalResponse; sloped=True gives shelf_slope_keV > 0."""
    def br(E, ft, b, fs, sl):
        d = {'E': E, 'f_tail': ft, 'tail_slope_keV': b, 'f_shelf': fs}
        if sloped:
            d['shelf_slope_keV'] = sl
        return d
    r = H.EmpiricalResponse(
        br([0.3, 1.0, 1.7], [0.10, 0.14, 0.18], [0.03, 0.04, 0.05],
           [0.06, 0.08, 0.10], [0.5, 0.7, 0.9]),
        br([1.8, 3.0, 6.5], [0.20, 0.16, 0.12], [0.05, 0.06, 0.07],
           [0.09, 0.07, 0.05], [0.8, 0.6, 0.5]))
    if al:
        r.al_frac_ref = 0.03
        r.mu_al = lambda E: 1.0 / E ** 2.7
    return r


def _jac_check(E, resp, p, idxs, rtol=1e-6, with_artefact=True):
    state = {'checked': 0, 'skipped': 0}
    _compare(lambda q: H.line_profile_jac(E, 0.7, q, N, resp, idxs, with_artefact),
             lambda q: H.line_profile(E, 0.7, q, N, resp, with_artefact),
             p, idxs, rtol, state)
    return state


@pytest.mark.parametrize('sloped', [True, False])
def test_hypermet_line_profile_jac_matches_central_differences(sloped):
    """hypermet.line_profile_jac: f exact, d f/d a_i to 1e-6, both shelf kinds."""
    resp = _synth_resp(sloped)
    idxs = [0, 1, 2, 3, 5, 6]
    checked = 0
    for E in (0.52, 1.04, 1.74, 2.3, 5.9):
        for seed in range(3):
            p = _pars(seed)
            if seed == 2:
                p.a[5], p.a[6] = 0.0, 0.0
                use = [0, 1, 2, 3]
            else:
                use = idxs
            checked += _jac_check(E, resp, p, use)['checked']
    assert checked >= 40, checked


def test_hypermet_line_profile_jac_at_the_095_clamp():
    """hypermet.line_profile_jac: derivative through the ft + fs <= 0.95 clamp."""
    for sloped in (True, False):
        resp = _synth_resp(sloped)
        for E in (1.04, 2.3):
            p = _pars(0)
            p.a[5], p.a[6] = 3.0, 2.5
            ft, _, fs, _ = resp.params(E)
            assert ft * p.a[5] ** 2 + fs * p.a[6] ** 2 > 0.95, 'clamp not hit'
            st = _jac_check(E, resp, p, [0, 1, 2, 3, 5, 6])
            assert st['checked'] == 6, st


def test_hypermet_line_profile_jac_with_the_al_artefact():
    """hypermet.line_profile_jac: the Al Ka artefact adds to f and to d."""
    for sloped in (True, False):
        resp = _synth_resp(sloped, al=True)
        for E in (1.74, 2.3, 5.9):
            assert resp.al_fraction(E) > 0
            p = _pars(1)
            st = _jac_check(E, resp, p, [0, 1, 2, 3, 5, 6])
            assert st['checked'] >= 5, st
            f0 = H.line_profile(E, 0.7, p, N, resp, False)
            assert not np.array_equal(f0, H.line_profile(E, 0.7, p, N, resp, True))


@pytest.mark.parametrize('zero', [(5,), (6,), (5, 6)])
def test_hypermet_line_profile_jac_with_unscaled_response(zero):
    """hypermet.line_profile_jac: a5/a6 = 0 means unscaled; d/d a_i is 0 there."""
    resp = _synth_resp(True)
    for E in (1.04, 2.3):
        p = _pars(0)
        for i in zero:
            p.a[i] = 0.0
        other = [i for i in (5, 6) if i not in zero]
        idxs = [0, 1, 2, 3] + other
        _jac_check(E, resp, p, idxs)
        f, d = H.line_profile_jac(E, 0.7, p, N, resp, [0, 1, 2, 3, 5, 6])
        assert np.array_equal(f, H.line_profile(E, 0.7, p, N, resp))
        for i in zero:
            assert not d[i].any()


def _sum_setup(width_factor):
    comps = _components()
    sp = F.SumPeakComponent(comps, sum_deficit=0.1, e_high=E_HIGH)
    sp.update({'O': 3e4, 'Na': 2e4, 'Si': 8e4, 'Fe': 1e4})
    sp.width_factor = width_factor
    assert sp._lines and sp._pairs
    return sp


@pytest.mark.parametrize('width_factor', [1.0, 1.25])
def test_sum_peak_profile_jac_includes_sum_deficit(width_factor):
    """SumPeakComponent.profile_jac: d/d a4 (deficit), with width_factor."""
    sp = _sum_setup(width_factor)
    idxs = [0, 1, 2, 3, 4]
    state = {'checked': 0, 'skipped': 0}
    for seed in range(3):
        p = _pars(seed)
        p.a[4] = 0.3
        _compare(lambda q: sp.profile_jac(q, N, idxs),
                 lambda q: sp.profile(q, N), p, idxs, 1e-4, state)
    _need_checks(state, 10)


@pytest.mark.parametrize('kind', ['flat', 'ramp'])
def test_pileup_plateau_profile_jac(kind):
    """PileupPlateauComponent.profile_jac for the flat and ramp boxes."""
    pl = F.PileupPlateauComponent(_sum_setup(1.0), kind=kind)
    idxs = [0, 1, 2, 3]
    state = {'checked': 0, 'skipped': 0}
    for seed in range(3):
        _compare(lambda q: pl.profile_jac(q, N, idxs),
                 lambda q: pl.profile(q, N), _pars(seed), idxs, 1e-4, state)
    _need_checks(state, 8)


def test_background_profile_jac_is_zero():
    """BackgroundComponent.profile_jac: shape-independent, all zeros."""
    bg = F.BackgroundComponent(np.linspace(5.0, 1.0, N))
    p = _pars(0)
    f, d = bg.profile_jac(p, N, SHAPE_IDXS)
    assert np.array_equal(f, bg.profile(p, N))
    for i in SHAPE_IDXS:
        assert d[i].shape == (N,) and not d[i].any()


# --- 3. residual_jacobian ----------------------------------------------------

def _problem(areas=None, bg=(40.0, 5.0), max_area=None, prior=False,
             spread=0.0, seed=5):
    comps = _components()
    p_true = _pars(11)
    p_true.a[4] = 0.3
    sp = F.SumPeakComponent(comps, sum_deficit=0.1, e_high=E_HIGH,
                            max_area=max_area)
    sp.deficit_spread = spread
    areas = dict(areas or {'O': 3e5, 'Na': 2e5, 'Si': 6e5, 'Fe': 1.5e5})
    sp.update(areas)
    if prior:
        for c, f in zip(comps[:2], (0.6, 0.4)):
            c.prior_group, c.prior_fraction = 'M', f
    ch = np.arange(N)
    e = CAL_B + CAL * ch
    shape = bg[0] * np.exp(-e / 2.0) + bg[1]
    bgc = F.BackgroundComponent(shape)
    allc = comps + [sp, bgc]
    amp = [areas['O'], areas['Na'], areas['Si'], areas['Fe'], 3e4, 1.0]
    model = sum(a * c.profile(p_true, N) for a, c in zip(amp, allc))
    counts = np.random.default_rng(seed).poisson(np.maximum(model, 0)).astype(float)
    # evaluate away from the truth, as a mid-fit iterate would be
    p = copy.deepcopy(p_true)
    p.a[0] *= 1.03
    p.a[1] *= 0.97
    p.a[3] *= 1.0002
    p.a[2] += 0.3
    p.a[4] = 0.25
    p.a[5] *= 1.05
    p.a[8] *= 1.1
    channels = np.arange(int((E_LOW - CAL_B) / CAL),
                         int((E_HIGH - CAL_B) / CAL) + 1)
    return counts, np.zeros(N), allc, p, channels


def _reference_resid(counts, background, comps, p, channels, want=None):
    """What fit_spectrum's inner resid() computes (fit.py, 'def resid').
    want, a dict, receives the areas 'ar' and the model 'm'."""
    ar, _, _, _, A = F.linear_step(counts, background, comps, p, channels,
                                   nonneg=True)
    m = F._matvec(A[channels], np.maximum(ar, 0.0)) + background[channels]
    if want is not None:
        want.update(ar=ar, m=m)
    return (counts[channels] - m) / np.sqrt(np.maximum(m, 1.0))


def _check_jacobian(prob, idxs):
    counts, background, comps, p, channels = prob
    want = {}
    ref = _reference_resid(counts, background, comps, p, channels, want)
    r, J = F.residual_jacobian(counts, background, comps, p, channels, idxs)
    # array_equal: the same expressions on identical arrays, and _matvec is
    # einsum, not BLAS
    assert np.array_equal(r, ref)
    assert J.shape == (len(channels), len(idxs))
    for k, i in enumerate(idxs):
        h = 1e-7 * max(1.0, abs(p.a[i]))
        wp, wm = {}, {}
        fd = (_reference_resid(counts, background, comps, _shifted(p, i, h),
                               channels, wp)
              - _reference_resid(counts, background, comps,
                                 _shifted(p, i, -h), channels, wm)) / (2 * h)
        assert np.array_equal(np.sign(wp['ar']), np.sign(wm['ar'])), i
        rel = np.linalg.norm(J[:, k] - fd) / np.linalg.norm(fd)
        # exact, not a fixed-weights approximation: J is differentiated
        # through the reweighting passes (1e-7 or better, 2026-10-22)
        assert rel < 1e-5, (i, rel)
    return want


IDXS = [0, 1, 2, 3, 4, 5, 6, 8, 11]


def test_residual_jacobian_matches_finite_differences_of_residual():
    """residual_jacobian: r equals fit_spectrum's resid; J = dr/da_i to 1e-5."""
    prob = _problem()
    want = _check_jacobian(prob, IDXS)
    assert (want['ar'] > 0).all()


def test_residual_jacobian_with_a_component_pinned_at_its_cap():
    """residual_jacobian: a max_area-pinned component drops out of J."""
    prob = _problem(max_area=1e4)
    info = {}
    F.linear_step(*prob[:2], prob[2], prob[3], prob[4], info=info)
    assert info['over'], 'cap not hit'
    _check_jacobian(prob, IDXS)


def test_residual_jacobian_with_prior_groups():
    """residual_jacobian: the soft-prior rows enter the solve and J."""
    prob = _problem(prior=True)
    info = {}
    F.linear_step(*prob[:2], prob[2], prob[3], prob[4], info=info)
    assert info['free']['groups'], 'no prior group'
    assert info['free']['groups'][0][1] > 1.0 and F.PRIOR_STRENGTH
    _check_jacobian(prob, IDXS)


def test_residual_jacobian_with_a_negative_area_clipped():
    """residual_jacobian: a negative unconstrained area is clipped in the model."""
    prob = _problem(areas={'O': 3e5, 'Na': 2e5, 'Si': 6e5, 'Fe': 0.0},
                    seed=8)    # seed 8: Fe comes out at -22 counts
    want = _check_jacobian(prob, IDXS)
    assert (want['ar'] < 0).any(), want['ar']


def test_residual_jacobian_in_the_low_count_region():
    """residual_jacobian: channels with model <= 1 have dr/dm = -1, no weight term."""
    prob = _problem(bg=(0.0, 0.2))
    want = _check_jacobian(prob, IDXS)
    assert (want['m'] <= 1.0).any()


def test_residual_jacobian_with_deficit_spread():
    """residual_jacobian: d/d a4 through the spread of sum-peak deficits."""
    prob = _problem(spread=3.0)
    sp = prob[2][-2]
    assert sp.deficit_spread > 0 and prob[3].a[4] != 0 and 4 in IDXS
    _check_jacobian(prob, IDXS)


# --- 4. wiring -------------------------------------------------------------

def test_residual_jacobian_reuses_a_solve_bit_for_bit():
    """J from a cached solve (what fit_spectrum's jac passes) equals J from a
    fresh one, to the bit: the profiles are the same arrays."""
    counts, background, comps, p, channels = _problem()
    idxs = [0, 1, 2, 3, 4, 5, 6, 8, 11]
    info = {}
    ar, _, _, _, _ = F.linear_step(counts, background, comps, p, channels,
                                   nonneg=True, info=info)
    r0, J0 = F.residual_jacobian(counts, background, comps, p, channels, idxs)
    r1, J1 = F.residual_jacobian(counts, background, comps, p, channels, idxs,
                                 solved=(ar, info))
    assert np.array_equal(r0, r1)
    assert np.array_equal(J0, J1)


def _small_fit(**kw):
    counts, _, comps, p, _ = _problem()
    comps = [c for c in comps if isinstance(c, F.Component)]
    return F.fit_spectrum(counts, CAL, CAL_B, comps, E_LOW, E_HIGH,
                          p.a[0], p.a[1], refine=('cal', 'width'), **kw)


def test_fit_spectrum_rejects_an_unknown_jacobian():
    with pytest.raises(ValueError):
        _small_fit(jacobian='analytical')


def test_fit_spectrum_runs_with_either_jacobian():
    """Both paths fit the same synthetic spectrum to nearly the same chi2."""
    a = _small_fit(jacobian='analytic')
    b = _small_fit(jacobian='2-point')
    assert np.isfinite(a.chi2) and np.isfinite(b.chi2)
    assert abs(a.chi2 - b.chi2) <= 1e-3 * b.chi2
