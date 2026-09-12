"""
PIXE spectrum fitting.

Structure follows GeoPIXE's pixe_fit.pro: the element amplitudes enter the
model linearly, everything else (calibration, width, tail) does not. So the
fit alternates

    linear step     solve for element areas by weighted least squares
    non-linear step adjust cal / width / tail to reduce chi-squared

which is the standard treatment and is what makes trace elements tractable:
the weak components are solved exactly at every step rather than being pushed
around by a general optimiser.

Weights follow Awaya (1978) as GeoPIXE does - the variance is taken from the
FITTED model, not the observed counts. Using observed counts biases weak peaks
low, because a downward Poisson fluctuation gets a larger weight. That bias is
precisely where trace element analysis lives, so it matters here more than
anywhere else.

    C.G. Ryan et al. (1990), Nucl. Instr. Meth. B49, 271.
    P.R. Bevington (1969), Data Reduction and Error Analysis.
"""

import numpy as np

import copy as _copy

from .peakshape import (P_FANO, P_NOISE, ShapePars,  # noqa: F401
                        line_profile)


class Component:
    """One fitted component - normally one element/shell.

    Escape lines are held in the SAME component as their parent, scaled by the
    escape fraction. They are not separate free parameters: tying them means
    they constrain the parent area rather than adding a degree of freedom.
    """

    def __init__(self, name, lines, tail_amp_fn=None, tail_len_fn=None,
                 beta_tail=1.0, escape=None, tilt_deg=0.0):
        self.name = name
        self.lines = list(lines)          # [(E_keV, relative_intensity), ...]
        if escape is not None:
            self.lines += escape.escape_lines(self.lines, tilt_deg=tilt_deg)
        self.tail_amp_fn = tail_amp_fn
        self.tail_len_fn = tail_len_fn
        self.beta_tail = beta_tail

    def profile(self, pars, n_channels, do_tail=True):
        f = np.zeros(n_channels)
        for E, beta in self.lines:
            ta = self.tail_amp_fn(E) if self.tail_amp_fn else 0.0
            tl = self.tail_len_fn(E) if self.tail_len_fn else 0.0
            f += line_profile(E, beta, pars, n_channels, tail_amp=ta,
                              tail_len=tl, beta_tail=self.beta_tail,
                              do_tail=do_tail)
        return f


def pileup_component(components, pars, n_channels, areas=None, ratio=1.0):
    """Sum-peak pile-up: the self-convolution of the modelled spectrum.

    Two photons arriving within the shaping time are recorded as one event at
    the sum of their energies. The resulting spectrum is the autocorrelation of
    the true one, scaled by a rate-dependent constant. GeoPIXE carries that
    constant as parameter index 4 and fits it.

    Computed from the CURRENT model rather than from the data, so it does not
    import the data's noise into the model, and so it responds to the fit.
    """
    m = np.zeros(n_channels)
    for i, c in enumerate(components):
        w = 1.0 if areas is None else max(areas[i], 0.0)
        if w > 0:
            m += w * c.profile(pars, n_channels)
    if m.sum() <= 0:
        return m
    full = np.convolve(m, m)[:n_channels]
    return ratio * full / max(m.sum(), 1.0)


def expected_pileup_fraction(total_counts, live_seconds,
                             shaping_time_us=1.0):
    """rate x tau: the chance a second photon arrives inside the shaping
    time. First order, good to a factor of a few - ample to reject a fit that
    is out by three orders of magnitude. None when the live time is unknown,
    so the caller decides rather than being handed a guess."""
    if not (total_counts and live_seconds and live_seconds > 0):
        return None
    return (float(total_counts) / float(live_seconds)) *         float(shaping_time_us) * 1e-6


class BackgroundComponent:
    """The continuum, with an amplitude the fit is allowed to choose.

    pixe.pro forms its model as

        result = signal + a[7]*background1 + a[10]*background2 + a[4]*pileup

    so the background enters with a FREE amplitude, exactly like an element.
    This package instead SUBTRACTED a fixed SNIP estimate, which is a
    different thing and a worse one.

    SNIP strips peaks down to the continuum, and next to a very large peak it
    strips a little too deep - the peak's own wings look like peak to it. The
    result is a background that is too low in a band a few hundred eV wide
    around every strong line. Subtract that and the model is short there, and
    the only way the fit can make up the difference is to inflate the element
    that owns the peak. On donut2x the iron area came out 1.34x too large for
    exactly this reason, with 47% of the total chi2 sitting 0.4-1.0 keV out
    from the strong peaks where the model ran +20 sigma short.

    Letting the amplitude float does not let the background eat real peaks:
    its shape is fixed, and a smooth curve cannot take on a peak's shape. It
    only lets the fit say the continuum is 1.1x what SNIP guessed, which is
    the honest degree of freedom that was missing.
    """

    name = 'background'

    def __init__(self, shape):
        self._shape = np.asarray(shape, float)
        self.tail_amp_fn = lambda E: 0.0
        self.tail_len_fn = lambda E: 0.0

    def profile(self, pars, n_channels):
        b = self._shape
        if len(b) < n_channels:
            b = np.pad(b, (0, n_channels - len(b)))
        return b[:n_channels]


class SumPeakComponent:
    """Pile-up as GeoPIXE models it: discrete sum LINES, not a convolution.

    This replaces an earlier version that used the self-convolution of the
    whole model. That was the wrong shape and it mattered enormously. A
    self-convolution of a spectrum-plus-continuum is smooth, positive and
    non-zero everywhere, so the fitter could use it to absorb any modelling
    error anywhere. On run 287424 it took 22% of the spectrum, against a
    physical expectation of 0.017% at 168 counts/s - out by a factor of 1300
    - and it disguised the problem, because switching it off sent chi2 from
    91 to 161 as though pile-up had been doing real work.

    sum_peaks.pro builds something quite different: ONE pseudo-element whose
    lines sit at every pairwise sum E_i + E_j of the strongest real lines,
    with intensity proportional to the product of the parent AREAS. Those are
    sharp features at specific energies. They can fit a sum peak and they
    cannot fit a broad continuum error, which is exactly the property that
    keeps the fit honest.

    Following sum_peaks.pro:
      check_double = 30   strongest lines combined pairwise
      check_triple = 6    strongest also combined in threes
      lines under 100 counts are ignored
      sum_deficit         fraction lost to finite time resolution, 0.1

    The intensities depend on the fitted areas, so the shape has to be
    rebuilt as the fit converges - update() does that, and fit_spectrum
    calls it between passes. The amplitude itself stays a single free
    parameter, as a[4] is in pixe.pro.
    """

    name = 'pileup'
    CHECK_DOUBLE = 30
    CHECK_TRIPLE = 6
    MIN_REL = 0.0003        # sum_peaks.pro: ar > 0.0003 * ar[0]

    def __init__(self, components, sum_deficit=0.1, e_high=None,
                 max_area=None):
        self._src = list(components)
        self.sum_deficit = float(sum_deficit)
        self.e_high = e_high
        self.max_area = max_area
        self._lines = []                 # [(energy, weight)]
        self.tail_amp_fn = lambda E: 0.0
        self.tail_len_fn = lambda E: 0.0

    def update(self, areas, amplitude=None):
        """Rebuild the sum-line list from the current element areas.

        Follows sum_peaks.pro closely, including three things this got wrong
        the first time:

        1. TRIPLES ARE SCALED BY p_factor. GeoPIXE forms the double products
           normalised to the strongest, r2 = d2/d2[0], and the triples as
           r4 = p_factor * d4/d4[0], where p_factor = asum/ar[0] is the
           pile-up probability. Without that factor the triple products - a
           product of THREE areas rather than two - are larger than the
           doubles by roughly the size of an area, about a million. The
           component then peaks at three times the strongest line instead of
           twice: on run 287427 it sat at 5.22 keV, which is 3 x Si Ka, when
           the real sum peak is Si+O at 2.27 keV.

        2. sum_deficit shifts the ENERGY, not the amplitude. sum_peaks.pro
           forms e = (1 - sum_deficit/100) * (e_i + e_j): two pulses that
           merge are slightly clipped, so the sum lands a little BELOW the
           arithmetic sum. It is also a percentage, so 0.1 means 0.1%, not
           10%.

        3. Lines are selected relative to the strongest, ar > 0.0003*ar[0],
           not against an absolute count.

        amplitude: the sum component's own fitted area, for p_factor. On the
        first pass it is unknown and only doubles are built, which is also
        what sum_peaks.pro does - it requires asum > 1000 before it will
        consider triples at all.
        """
        strong = []
        for c in self._src:
            area = areas.get(getattr(c, 'name', None), 0.0)
            if area <= 0.1:
                continue
            for e, inten in getattr(c, 'lines', ()) or ():
                strong.append((area * inten, float(e)))
        if not strong:
            self._lines = []
            return 0
        strong.sort(reverse=True)
        top = strong[0][0]
        strong = [t for t in strong if t[0] > self.MIN_REL * top]

        shift = 1.0 - self.sum_deficit / 100.0
        out = []

        dbl = strong[:self.CHECK_DOUBLE]
        d0 = dbl[0][0] * dbl[0][0]
        for i, (wi, ei) in enumerate(dbl):
            for j, (wj, ej) in enumerate(dbl):
                es = shift * (ei + ej)
                if self.e_high and es > self.e_high:
                    continue
                out.append((es, wi * wj / d0))

        # Triples, only once the amplitude is known and large enough to
        # matter - the same gate sum_peaks.pro applies.
        p_factor = (float(amplitude) / top) if (amplitude and top > 0) else 0.0
        if amplitude and amplitude > 1000.0 and p_factor > 3.0e-4:
            tri = strong[:self.CHECK_TRIPLE]
            t0 = 3.0 * tri[0][0] ** 3
            for wi, ei in tri:
                for wj, ej in tri:
                    for wk, ek in tri:
                        es = shift * (ei + ej + ek)
                        if self.e_high and es > self.e_high:
                            continue
                        out.append((es, p_factor * 3.0 * wi * wj * wk / t0))

        tot = sum(w for _, w in out)
        if tot > 0:
            out = [(e, w / tot) for e, w in out]
        self._lines = out
        return len(out)

    def profile(self, pars, n_channels):
        """Unit-area sum of the sum-peak lines.

        Uses the same line_profile as every other component, so the sum peaks
        get the detector's real width and tail rather than a bare Gaussian.
        The width is then widened by sqrt(2): two independent events add
        their variances, which is also the practical tell that a feature is a
        sum peak rather than a line.
        """
        f = np.zeros(n_channels)
        if not self._lines:
            return f
        # Two independent events add their variances, so a sum peak is
        # sqrt(2) wider than a line at the same energy. Copy the parameters
        # and scale the width terms rather than mutating the fit's own.
        wide = _copy.copy(pars)
        wide.a = np.array(pars.a, dtype=float)
        wide.a[P_NOISE] *= np.sqrt(2.0)
        wide.a[P_FANO] *= np.sqrt(2.0)
        for e, w in self._lines:
            f += line_profile(e, w, wide, n_channels, do_tail=False)
        t = f.sum()
        return f / t if t > 0 else f


class FitResult:
    def __init__(self, areas, errors, names, chi2, ndf, model, background,
                 pars, profiles):
        self.areas = areas
        self.errors = errors
        self.names = names
        self.chi2 = chi2
        self.ndf = ndf
        self.model = model
        self.background = background
        self.pars = pars
        self.profiles = profiles

    @property
    def reduced_chi2(self):
        return self.chi2 / self.ndf if self.ndf > 0 else float('nan')

    def table(self):
        out = ['  component      area        error      rel.err',
               '  ' + '-' * 46]
        for n, a, e in zip(self.names, self.areas, self.errors):
            rel = 100.0 * e / a if a > 0 else float('nan')
            out.append('  %-10s %11.1f %11.1f %9.2f%%' % (n, a, e, rel))
        out.append('  reduced chi2 = %.3f  (%d dof)' % (self.reduced_chi2, self.ndf))
        return '\n'.join(out)


def _weights(model, floor=1.0):
    """Awaya weights: variance from the model, floored so empty channels
    cannot dominate."""
    return 1.0 / np.maximum(model, floor)


def linear_step(counts, background, components, pars, channels,
                n_iter=3, nonneg=True):
    """Solve for areas, honouring any component that declares a max_area.

    A capped component is solved unbounded first. If it comes back over its
    cap it is PINNED at the cap - not dropped, because those counts are real
    and still have to be accounted for - moved into the background, and the
    remaining components re-solved against what is left. The degrees of
    freedom drop accordingly, since a pinned component is no longer fitted.
    """
    caps = {i: c.max_area for i, c in enumerate(components)
            if getattr(c, 'max_area', None) is not None}
    areas, errors, chi2, ndf, A = _linear_step_free(
        counts, background, components, pars, channels, n_iter, nonneg)
    over = {i: lim for i, lim in caps.items() if areas[i] > lim}
    if not over:
        return areas, errors, chi2, ndf, A

    pinned = np.zeros(len(counts))
    for i, lim in over.items():
        pinned += A[:, i] * lim
    keep = [i for i in range(len(components)) if i not in over]
    sub_a, sub_e, chi2, _, _ = _linear_step_free(
        counts, background + pinned, [components[i] for i in keep],
        pars, channels, n_iter, nonneg)

    areas = np.zeros(len(components))
    errors = np.zeros(len(components))
    for j, i in enumerate(keep):
        areas[i] = sub_a[j]
        errors[i] = sub_e[j]
    for i, lim in over.items():
        areas[i] = lim
        errors[i] = 0.0            # pinned, not measured
    return areas, errors, chi2, len(channels) - len(keep), A


def _linear_step_free(counts, background, components, pars, channels,
                      n_iter=3, nonneg=True):
    """Solve for component areas at fixed shape parameters.

    Iterated because the weights depend on the model, which depends on the
    areas. Three passes is what GeoPIXE uses and is enough in practice.

    nonneg
        True     unconstrained solve; negative areas are reported as-is and
                 clipped only for the weighting model. A negative area is
                 informative - it says the component is not present - so this
                 is the right mode when deciding whether to keep an element.
        'strict' true non-negative least squares. Use this when COMPARING
                 models, because an unconstrained chi2 can be lowered by a
                 physically impossible negative peak, which makes two variants
                 incomparable.
    """
    n = len(counts)
    A = np.column_stack([c.profile(pars, n) for c in components])
    Ac = A[channels, :]
    yc = counts[channels] - background[channels]

    model = np.maximum(counts[channels], 1.0)
    areas = None
    for _ in range(n_iter):
        w = _weights(model)
        Aw = Ac * w[:, None]
        M = Ac.T @ Aw
        v = Aw.T @ yc
        try:
            cov = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(M)
        if nonneg == 'strict':
            from scipy.optimize import nnls
            sw = np.sqrt(w)
            areas, _ = nnls(Ac * sw[:, None], yc * sw)
        else:
            areas = cov @ v
        if nonneg:
            # one round of clipping: drop negatives and re-solve on the rest.
            # Negative areas are physically meaningless but ARE informative -
            # they say the component is not detected - so they are reported
            # by fit_spectrum and only clipped for the model used in weighting.
            areas_w = np.maximum(areas, 0.0)
        else:
            areas_w = areas
        model = np.maximum(Ac @ areas_w + background[channels], 1.0)

    resid = yc - Ac @ areas
    w = _weights(model)
    chi2 = float(np.sum(w * resid ** 2))
    ndf = len(channels) - len(components)
    errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return areas, errors, chi2, ndf, A


def fit_spectrum(counts, cal_a, cal_b, components, e_low, e_high,
                 noise, fano, tail_amp=0.0, tail_len=0.0,
                 background=None, refine=('cal', 'width'), max_iter=12,
                 verbose=False, nonneg=True, fit_background=True,
                 start_pars=None):
    """Fit a PIXE spectrum.

    counts            spectrum
    cal_a, cal_b      initial energy calibration, E = cal_a*ch + cal_b
    components        list of Component
    e_low, e_high     fit range, keV
    noise, fano       initial width parameters, in the GeoPIXE parameterisation
                      (channels; w^2 = noise^2 + fano^2*(E - e_low))
    background        pre-computed background, or None to SNIP it here
    fit_background    give the background a free amplitude, as pixe.pro does,
                      instead of subtracting it. See BackgroundComponent -
                      subtracting a SNIP estimate near a strong peak leaves
                      the model short and the fit inflates the element to
                      compensate.
    refine            which non-linear groups to vary: 'cal', 'width', 'tail'
    """
    from .background import snip

    counts = np.asarray(counts, dtype=float)
    n = len(counts)

    if background is None:
        background = snip(counts, cal_a, cal_b, e_low, e_high)
    background = np.asarray(background, float)

    # GeoPIXE fits the background amplitude rather than subtracting a fixed
    # estimate (pixe.pro: result = signal + a[7]*background1 + ...). Do the
    # same: the SNIP curve becomes a component with a free amplitude, and
    # nothing is subtracted from the data.
    fitted_bg = None
    if fit_background:
        fitted_bg = BackgroundComponent(background)
        components = list(components) + [fitted_bg]
        background = np.zeros_like(background)

    # GeoPIXE references the centroid to eoc, so cal_b in ShapePars is the
    # channel of energy eoc, not of zero energy. Convert.
    eoc = 0.7 * e_low + 0.3 * e_high
    ch_at_eoc = (eoc - cal_b) / cal_a
    pars = ShapePars(noise, fano, ch_at_eoc, 1.0 / cal_a,
                     tail_amp=tail_amp, tail_len=tail_len,
                     e_low=e_low, e_high=e_high)
    if start_pars is not None:
        # Continue from a converged set - the sum-peak passes only need the
        # areas re-solved against updated sum lines, not a fresh search.
        pars.a = np.array(start_pars.a, dtype=float)
        refine = ()

    # Seed the tail parameters away from the alpha_zero zero-crossing.
    #
    # alpha = tail_amp*(alpha_zero + a5^2) with alpha_zero = -0.05, so a5 must
    # exceed sqrt(0.05) = 0.224 before the tail is even positive. Starting at
    # a5 = 0 leaves the optimiser sitting in a flat region on the wrong side of
    # the crossing, where small steps produce no improvement and the tail is
    # silently never fitted. Seeding at 1.0 puts it in the responsive range.
    if 'tail' in refine:
        if pars.a[5] == 0.0:
            pars.a[5] = 1.0
        if pars.a[6] == 0.0:
            pars.a[6] = 1.0
    if 'contact' in refine:
        if pars.a[11] <= 0.0:
            pars.a[11] = 0.03
    if 'window' in refine:
        if pars.a[12] <= 0.0:
            pars.a[12] = 0.02
    if 'slope' in refine:
        if pars.a[13] <= 0.0:
            pars.a[13] = 6.0
        if pars.a[14] <= 0.0:
            pars.a[14] = 1.0
    if 'shelf' in refine:
        # seed away from zero, or the search cannot move off the boundary
        if pars.a[8] <= 0.0:
            pars.a[8] = 0.05
        if pars.a[9] <= 0.0:
            pars.a[9] = 1.0

    ch_lo = max(int((e_low - cal_b) / cal_a), 1)
    ch_hi = min(int((e_high - cal_b) / cal_a), n - 2)
    channels = np.arange(ch_lo, ch_hi + 1)

    def chi2_of(p):
        _, _, c2, _, _ = linear_step(counts, background, components, p, channels,
                                     nonneg=nonneg)
        return c2

    best = chi2_of(pars)
    if verbose:
        print('  start chi2 = %.1f' % best)

    # Coordinate descent on the non-linear parameters. Deliberately simple and
    # bounded rather than a general optimiser: these parameters are strongly
    # constrained by the strong peaks, and a wandering optimiser degrades the
    # weak ones. Steps shrink on failure.
    groups = {'cal': [(3, 1e-4), (2, 0.05)],       # gain, offset(channel)
              'width': [(0, 0.2), (1, 0.2)],       # noise, fano
              'tail': [(5, 0.2), (6, 0.2)],
              # incomplete-charge-collection shelf: amplitude and the energy
              # scale over which it decays. Not part of GeoPIXE's model.
              'shelf': [(8, 0.02), (9, 0.2)],
              # contact injection from the C1 window's 250 nm Al layer
              'contact': [(11, 0.02)],
              # Si3N4 window, nitrogen K-edge injection
              'window': [(12, 0.02)],
              # shelf slopes, made energy-dependent rather than fixed
              'slope': [(13, 0.5), (14, 0.2), (15, 0.2)]}
    active = []
    for g in refine:
        active.extend(groups.get(g, []))

    # Levenberg-Marquardt on the non-linear parameters, with the linear
    # areas re-solved inside every residual evaluation (variable projection).
    #
    # This replaces a coordinate descent that stepped one parameter at a time
    # and halved the step on failure. That cannot follow a valley: noise and
    # Fano are strongly correlated - FWHM^2 = noise^2 + fano^2 (E - e_low) -
    # so moving one alone always looks worse and the search stalls wherever
    # it started. On run 380001 it converged to chi2 53.6 from the detector
    # file's widths and 43.3 from a different seed, which is the signature.
    #
    # GeoPIXE's pixe.pro fits the same parameters by Marquardt with analytic
    # derivatives. scipy's trust-region LM with a numerical Jacobian is the
    # same idea and, for a handful of parameters, costs nothing.
    idxs = [i for i, _ in active]
    if idxs:
        from scipy.optimize import least_squares
        lower = [(0.0 if i in (8, 9, 11, 12, 13, 14) else -np.inf)
                 for i in idxs]
        x0 = np.array([pars.a[i] for i in idxs], dtype=float)
        # scale each parameter so a unit step is a sensible move in it
        scale = np.array([max(abs(st), 1e-6) for _, st in active])

        def resid(x):
            trial = ShapePars(pars.a[0], pars.a[1], pars.a[2], pars.a[3],
                              pars.a[5], pars.a[6], e_low, e_high)
            trial.a = pars.a.copy()
            for i, v in zip(idxs, x):
                trial.a[i] = v
            # Unconstrained solve here: the non-linear parameters are set by
            # the strong peaks, which NNLS never clips, and NNLS costs ten
            # times as much per evaluation. The final areas are re-solved
            # with the caller's nonneg setting after the optimiser returns.
            ar, _, _, _, A = linear_step(counts, background, components,
                                         trial, channels, nonneg=True)
            m = A[channels] @ np.maximum(ar, 0.0) + background[channels]
            return (counts[channels] - m) / np.sqrt(np.maximum(m, 1.0))

        try:
            sol = least_squares(resid, x0, bounds=(lower, np.inf),
                                x_scale=scale, method='trf',
                                max_nfev=12 * max(len(idxs), 1),
                                ftol=1e-5, xtol=1e-5)
            c2 = float(np.sum(sol.fun ** 2))
            if c2 < best:
                best = c2
                for i, v in zip(idxs, sol.x):
                    pars.a[i] = float(v)
        except Exception as ex:
            if verbose:
                print('  least_squares failed (%s); keeping the start' % ex)
    if verbose:
        print('  final chi2 = %.1f' % best)

    areas, errors, chi2, ndf, A = linear_step(counts, background, components,
                                              pars, channels, nonneg=nonneg)
    model = A @ np.maximum(areas, 0.0) + background
    names = [c.name for c in components]
    # When the background was fitted as a component the 'background' array
    # here is the zeros it was replaced with. Report the FITTED background -
    # amplitude times shape - so plots and downstream code see the curve the
    # model actually used, not an empty line.
    if fitted_bg is not None:
        k = components.index(fitted_bg)
        background = A[:, k] * max(float(areas[k]), 0.0)
    return FitResult(areas, errors, names, chi2, ndf, model, background,
                     pars, A)
