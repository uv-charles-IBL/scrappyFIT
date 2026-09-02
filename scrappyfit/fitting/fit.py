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
    MIN_LINE_COUNTS = 100.0

    def __init__(self, components, sum_deficit=0.1, e_high=None,
                 max_area=None):
        self._src = list(components)
        self.sum_deficit = float(sum_deficit)
        self.e_high = e_high
        self.max_area = max_area
        self._lines = []                 # [(energy, weight)]
        self.tail_amp_fn = lambda E: 0.0
        self.tail_len_fn = lambda E: 0.0

    def update(self, areas):
        """Rebuild the sum-line list from the current element areas.

        areas: {component name: fitted area}. Lines are (parent area x line
        intensity), the same product sum_peaks.pro forms.
        """
        strong = []
        for c in self._src:
            area = areas.get(getattr(c, 'name', None), 0.0)
            if area <= 0.1:
                continue
            for e, inten in getattr(c, 'lines', ()) or ():
                w = area * inten
                if w > self.MIN_LINE_COUNTS:
                    strong.append((w, float(e)))
        strong.sort(reverse=True)

        out = []
        dbl = strong[:self.CHECK_DOUBLE]
        for i, (wi, ei) in enumerate(dbl):
            for j in range(i, len(dbl)):
                wj, ej = dbl[j]
                es = ei + ej
                if self.e_high and es > self.e_high:
                    continue
                # the cross terms count twice: either photon can arrive first
                out.append((es, wi * wj * (1.0 if i == j else 2.0)))
        tri = strong[:self.CHECK_TRIPLE]
        for i, (wi, ei) in enumerate(tri):
            for j in range(i, len(tri)):
                wj, ej = tri[j]
                for k in range(j, len(tri)):
                    wk, ek = tri[k]
                    es = ei + ej + ek
                    if self.e_high and es > self.e_high:
                        continue
                    out.append((es, wi * wj * wk))
        total = sum(w for _, w in out)
        if total > 0:
            scale = (1.0 - self.sum_deficit) / total
            out = [(e, w * scale) for e, w in out]
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
                 verbose=False, nonneg=True):
    """Fit a PIXE spectrum.

    counts            spectrum
    cal_a, cal_b      initial energy calibration, E = cal_a*ch + cal_b
    components        list of Component
    e_low, e_high     fit range, keV
    noise, fano       initial width parameters, in the GeoPIXE parameterisation
                      (channels; w^2 = noise^2 + fano^2*(E - e_low))
    background        pre-computed background, or None to SNIP it here
    refine            which non-linear groups to vary: 'cal', 'width', 'tail'
    """
    from .background import snip

    counts = np.asarray(counts, dtype=float)
    n = len(counts)

    if background is None:
        background = snip(counts, cal_a, cal_b, e_low, e_high)

    # GeoPIXE references the centroid to eoc, so cal_b in ShapePars is the
    # channel of energy eoc, not of zero energy. Convert.
    eoc = 0.7 * e_low + 0.3 * e_high
    ch_at_eoc = (eoc - cal_b) / cal_a
    pars = ShapePars(noise, fano, ch_at_eoc, 1.0 / cal_a,
                     tail_amp=tail_amp, tail_len=tail_len,
                     e_low=e_low, e_high=e_high)

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

    steps = {i: s for i, s in active}
    for _ in range(max_iter):
        improved = False
        for idx, _ in active:
            for sign in (+1, -1):
                trial = ShapePars(pars.a[0], pars.a[1], pars.a[2], pars.a[3],
                                  pars.a[5], pars.a[6], e_low, e_high)
                trial.a = pars.a.copy()
                trial.a[idx] += sign * steps[idx]
                if idx in (8, 9, 11, 12, 13, 14) and trial.a[idx] < 0.0:
                    continue
                c2 = chi2_of(trial)
                if c2 < best - 1e-6:
                    best = c2
                    pars = trial
                    improved = True
                    break
        if not improved:
            for i in steps:
                steps[i] *= 0.5
            if max(steps.values()) < 1e-7:
                break
    if verbose:
        print('  final chi2 = %.1f' % best)

    areas, errors, chi2, ndf, A = linear_step(counts, background, components,
                                              pars, channels, nonneg=nonneg)
    model = A @ np.maximum(areas, 0.0) + background
    names = [c.name for c in components]
    return FitResult(areas, errors, names, chi2, ndf, model, background,
                     pars, A)
