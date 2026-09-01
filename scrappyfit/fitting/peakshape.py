"""
PIXE peak shape, ported from GeoPIXE's line.pro.

The model is a Gaussian with a low-energy exponential tail:

    FWHM (channels)   w^2 = a0^2 + a1^2 * (E - eow)
    centroid          c   = a2 + a3 * (E - eoc)
    Gaussian          G   = kf*sqrt(kw)*beta/w * exp(-kw*(x-c)^2 / (2*w^2))
    tail  (x < c)     T   = alpha*beta_tail*beta*exp((x-c)/(gamma*w))/w * (1 - g)

    kw = 5.5451 (8 ln 2), kf = 0.39894 (1/sqrt(2 pi))
    eow = e_low
    eoc = 0.7*e_low + 0.3*e_high
    alpha = tail_amp * (alpha_zero + a5^2),  alpha_zero = -0.05
    gamma = tail_len * (gamma_zero + a6^2),  gamma_zero =  0.1

Two details that are easy to get wrong and matter for trace work:

  * The Gaussian is AREA-NORMALISED. kf*sqrt(kw)/w is exactly 1/(sigma*sqrt(2pi))
    with sigma = w/sqrt(8 ln 2). So the fitted amplitude a[k] is a peak AREA,
    not a height. Getting this wrong rescales every concentration.

  * The energy origins eow and eoc are not cosmetic. Referencing width and
    centroid to a mid-range energy decorrelates offset from gain and noise
    from Fano during the fit. Setting them to zero makes the fit far less
    stable, which shows up first in the weakest peaks - the trace elements.

  * The tail is evaluated only for x < c, and is suppressed under the peak by
    the (1 - g) factor so it does not double-count the Gaussian core.

Parameter vector follows GeoPIXE's indexing:
    0 noise      2 cal B     4 pileup    5 tail amp       7  backgnd 1
    1 Fano       3 cal A                 6 tail length   10  backgnd 2
"""

import numpy as np
from scipy.special import erfc

KW = 5.5451
KW05 = np.sqrt(KW)
KF = 0.39894
ALPHA_ZERO = -0.05
GAMMA_ZERO = 0.1

# parameter indices
P_NOISE, P_FANO, P_CALB, P_CALA = 0, 1, 2, 3
P_PILEUP, P_TAILAMP, P_TAILLEN = 4, 5, 6
# Extension beyond GeoPIXE's vector: the flat incomplete-charge-collection
# shelf. GeoPIXE has no equivalent term. Indices 8 and 9 were unused.
P_SHELF0, P_SHELFE = 8, 9
# Contact-injection shelf from the C1 window's 250 nm Al metallisation.
P_CONTACT = 11
# Si3N4 window: nitrogen K-edge photoelectron injection.
P_WINDOW = 12
# Shelf slope parameters. Heirwegh fits the reciprocal slopes separately at
# each incident energy rather than holding them fixed, so they are made
# energy-dependent here: S1(E) = a13 + a15 * E, S2(E) = a14.
P_DES1, P_DES2, P_DES1E = 13, 14, 15
SHELF_REF = 1.0          # keV, reference energy for the step amplitude scaling
STEP_FRAC = 0.92         # diffusion step: down to this fraction of the peak energy
SI_LVV = 0.090           # keV, Si LVV Auger escape - a FIXED offset below the peak
# Reciprocal slopes of the difference-of-exponentials shelf, in units of the
# peak sigma. S1 sets how fast the shelf falls away toward higher energy; S2
# sets how rounded the low-energy cut-off is. Heirwegh (2014) eq. 2.1-2.2.
# Measured outcome: fitting S1 and S2 freely per spectrum improves chi2 by only
# ~0.02 and returns values that do NOT transfer between samples (3.0/0.2 from
# the chondrite, 12.0/3.4 from the quartz, each worse than the default on the
# other sample). A genuine detector property must transfer, so these are held
# fixed. Heirwegh could fit them because his spectra were MONOCHROMATIC with
# 40-100 million counts and one peak; in a polychromatic spectrum every peak's
# shelf overlaps its neighbours and the slopes are simply not determined.
DE_S1 = 6.0
DE_S2 = 1.0
AL_L_EDGE = 0.0727       # keV, Al L3 binding energy - sets the contact-shelf edge
N_K_EDGE = 0.4099        # keV, nitrogen K edge in the 90 nm Si3N4 window
SI_GRID_OPEN = 0.78      # C1 support grid is 78% open; 15 um Si blocks the rest
SI_K_EDGE = 1.839        # keV; the step intensity is discontinuous here


class ShapePars:
    """The non-linear parameters shared by every line in a spectrum."""

    def __init__(self, noise, fano, cal_b, cal_a,
                 tail_amp=0.0, tail_len=0.0, e_low=1.0, e_high=10.0):
        self.a = np.zeros(16)
        self.a[P_NOISE] = noise
        self.a[P_FANO] = fano
        self.a[P_CALB] = cal_b
        self.a[P_CALA] = cal_a
        self.a[P_TAILAMP] = tail_amp
        self.a[P_TAILLEN] = tail_len
        self.e_low = e_low
        self.e_high = e_high

    @property
    def eow(self):
        return self.e_low

    @property
    def eoc(self):
        return 0.7 * self.e_low + 0.3 * self.e_high

    def fwhm_channels(self, E):
        """FWHM in channels at energy E (keV)."""
        w2 = self.a[P_NOISE] ** 2 + self.a[P_FANO] ** 2 * (E - self.eow)
        return np.sqrt(np.maximum(w2, 1e-12))

    def fwhm_kev(self, E):
        return self.fwhm_channels(E) * self.a[P_CALA]

    def centroid(self, E):
        """Centroid channel for a line at energy E (keV)."""
        return self.a[P_CALB] + self.a[P_CALA] * (E - self.eoc)

    def energy_of_channel(self, ch):
        return self.eoc + (np.asarray(ch, float) - self.a[P_CALB]) / self.a[P_CALA]



def de_slopes(pars, E):
    """Reciprocal slopes at this photon energy.

    Heirwegh fits S1 and S2 independently for each incident energy rather than
    adopting one pair for the whole spectrum, and the fitted values do move
    with energy - the shelves are produced by escaping electrons whose range,
    and therefore whose energy distribution, depends on how deep the photon
    was absorbed. A single fixed pair is a convenience, not physics.

    S1(E) = a13 + a15 * E     falls away toward higher energy
    S2(E) = a14               roundness of the low-energy cut-off
    """
    s1 = pars.a[P_DES1] if pars.a[P_DES1] > 1e-6 else DE_S1
    s2 = pars.a[P_DES2] if pars.a[P_DES2] > 1e-6 else DE_S2
    s1 = s1 + pars.a[P_DES1E] * E
    return max(s1, 0.05), max(s2, 0.05)


def de_shelf(x, c_p, s_p, c_c, s_c, S1=DE_S1, S2=DE_S2):
    """Sloped shelf: the difference-of-exponentials form of Heirwegh (2014).

    A flat step is the wrong shape. Papp's electron-spectroscopy measurements
    and the Guelph Monte Carlo both show the electron-escape shelves slope,
    falling away toward higher energy, with a rounded low-energy cut-off set
    by the maximum kinetic energy the escaping electron can carry. On a log
    scale the difference between a flat and a sloped shelf is obvious, and it
    is precisely the region where a spurious element would otherwise be fitted.

        DE(i) = 1/2 [ E'(i, S1) - E'(i, S2) ]

        E'(i,S) = exp( (i-i_p)/(sigma_p S) + 1/(2 S^2) )
                  x { erfc[ (1/sqrt2)( (i-i_p)/sigma_p + 1/S ) ]
                    - erfc[ (1/sqrt2)( (i-i_c)/sigma_c + 1/S ) ] }

    i_p, sigma_p   peak channel and Gaussian width
    i_c, sigma_c   the cut-off channel and its width
    S              reciprocal slope, in units of sigma_p

    Returned unnormalised; the caller scales it to the wanted area fraction.
    """
    def Ep(S):
        S = max(float(S), 1e-3)
        z = (x - c_p) / (s_p * S) + 1.0 / (2.0 * S * S)
        z = np.clip(z, -700.0, 700.0)
        a = erfc((1.0 / np.sqrt(2.0)) * ((x - c_p) / s_p + 1.0 / S))
        b = erfc((1.0 / np.sqrt(2.0)) * ((x - c_c) / max(s_c, 1e-6) + 1.0 / S))
        return np.exp(z) * (a - b)
    return 0.5 * (Ep(S1) - Ep(S2))


def line_profile(E, beta, pars, n_channels, tail_amp=0.0, tail_len=0.0,
                 beta_tail=1.0, do_tail=True):
    """Unit-area profile for one line, over the whole spectrum.

    E          line energy, keV
    beta       relative intensity of this line within its element
    pars       ShapePars
    tail_amp   detector tail amplitude at this energy (tail_amplitude.pro)
    tail_len   detector tail length at this energy (tail_length.pro)
    beta_tail  extra tail scaling, larger for beta lines

    Returns an array of length n_channels. Multiply by the element amplitude
    to get counts. Because the Gaussian is area-normalised, the sum of the
    returned array is beta (plus the tail contribution).
    """
    f = np.zeros(n_channels, dtype=float)

    w = float(pars.fwhm_channels(E))
    w2 = w * w
    c = float(pars.centroid(E))
    if c < 0 or c > n_channels - 1:
        return f

    use_tail = do_tail and tail_amp > 0.0
    if use_tail:
        m1 = min(max(int(25.0 * w), 50), 1000)
        m2 = min(max(int(4.0 * w), 5), 1000)
    else:
        m1 = m2 = min(max(int(4.0 * w), 5), 1000)

    # --- Gaussian core ---
    lo = max(int(np.floor(c - m2)), 0)
    hi = min(int(np.ceil(c + m2)), n_channels - 1)
    if hi <= lo:
        return f
    x = np.arange(lo, hi + 1, dtype=float)
    kxc2 = KW * (x - c) ** 2 / w2
    g_core = np.exp(-kxc2 / 2.0)
    f[lo:hi + 1] += KF * KW05 * beta / w * g_core

    # --- low-energy tail ---
    if use_tail:
        lo_t = max(int(np.floor(c - m1)), 0)
        hi_t = int(np.floor(c))
        if hi_t > lo_t:
            xt = np.arange(lo_t, hi_t, dtype=float)
            kxc2t = KW * (xt - c) ** 2 / w2
            gt = np.exp(-kxc2t / 2.0)
            alpha = tail_amp * (ALPHA_ZERO + pars.a[P_TAILAMP] ** 2)
            gamma = tail_len * (GAMMA_ZERO + pars.a[P_TAILLEN] ** 2)
            if gamma > 0:
                xcw = (xt - c) / (gamma * w)
                # exp(xcw) with xcw <= 0, so no overflow
                aket = alpha * beta_tail * beta * np.exp(xcw) / w
                f[lo_t:hi_t] += aket * (1.0 - gt)

    # --- short-step diffusion feature ---
    # Heirwegh (2014, Guelph, Campbell group) characterises this as a step
    # extending from the primary peak down to about 90-95% of the peak ENERGY
    # - a fractional offset, not a fixed one - produced by out-diffusion and
    # recombination of secondary electrons at the contact boundary. Its area
    # relative to the parent peak is a few percent, rising sharply at the Si K
    # binding energy (1.839 keV) where photons begin to ionise the silicon K
    # shell. Carbon, nitrogen, oxygen and fluorine all sit BELOW that edge, so
    # light-element work lives entirely in the low side of that discontinuity.
    #
    #   fs(E) = a8 * exp(-E / a9)        area fraction, fitted per detector
    #   extent  E down to STEP_FRAC * E
    #
    # Magnitude is contact-specific - the work function at the contact-Si
    # boundary governs how many secondary electrons escape (Ni/Si 0.45 eV,
    # Al/Si 0.20 eV) - so a8 and a9 must be fitted for the detector in use
    # rather than taken from the literature.
    s0 = pars.a[P_SHELF0]
    if s0 > 0.0:
        es = pars.a[P_SHELFE] if pars.a[P_SHELFE] > 1e-6 else SHELF_REF
        fs = s0 * np.exp(-E / es)
        # The shelf extends below the peak by whichever of the two escape
        # mechanisms reaches further:
        #   Si LVV Auger escape   a FIXED ~90 eV, set by the Auger energy
        #   diffusion short step  a FRACTIONAL ~8% of the peak energy
        # They cross near 1.1 keV. Every light element of interest - C, N, O,
        # F - lies below that crossing, so for light-element work the fixed
        # Auger term dominates and a purely fractional model gets the width
        # badly wrong: at C Ka, 8% is 22 eV against a 62 eV resolution, so a
        # fractional-only shelf is entirely buried under the Gaussian.
        drop = max(SI_LVV, (1.0 - STEP_FRAC) * E)
        c_lo = float(pars.centroid(E - drop))
        lo_s = max(int(np.floor(c_lo)), 0)
        hi_s = int(np.floor(c))
        if fs > 0.0 and hi_s > lo_s:
            xs = np.arange(lo_s, hi_s, dtype=float)
            sig = w / np.sqrt(KW)
            S1, S2 = de_slopes(pars, E)
            d = de_shelf(xs, c, sig, c_lo, sig, S1, S2)
            tot = d.sum()
            if tot > 0:
                f[lo_s:hi_s] += fs * beta * d / tot

    # --- contact-injection shelf, C1 window ---
    # The Amptek C1 window is 90 nm Si3N4 over 250 nm Al. A photon absorbed in
    # that aluminium ejects an electron which is injected into the Si bulk,
    # having first lost an unknown amount of energy in the metal, so it shows
    # up as a continuum whose high-energy edge sits at E minus the Al binding
    # energy. For light elements the relevant shell is Al L3 at 72.7 eV: Al K
    # (1.56 keV) is out of reach for C, N, O and F photons.
    #
    # The edge position is fixed by atomic physics and only the amplitude is
    # fitted. That restraint matters, because this feature is nearly
    # degenerate with the Si LVV Auger escape step above - 73 eV against 90
    # eV, both well inside one resolution width - so fitting both extents as
    # well as both amplitudes would not be determined by the data.
    #
    # C1 carries 250 nm of Al where C2 carries 30 nm, so this term should be
    # substantially larger on a C1 detector than a C2 one.
    ct = pars.a[P_CONTACT]
    if ct > 0.0:
        es2 = pars.a[P_SHELFE] if pars.a[P_SHELFE] > 1e-6 else SHELF_REF
        fc = ct * np.exp(-E / es2)
        c_lo2 = float(pars.centroid(max(E - AL_L_EDGE, 1e-4)))
        lo_c = max(int(np.floor(c_lo2)), 0)
        hi_c = int(np.floor(c))
        if fc > 0.0 and hi_c > lo_c:
            xc = np.arange(lo_c, hi_c, dtype=float)
            sig = w / np.sqrt(KW)
            S1, S2 = de_slopes(pars, E)
            d = de_shelf(xc, c, sig, c_lo2, sig, S1, S2)
            tot = d.sum()
            if tot > 0:
                f[lo_c:hi_c] += fc * beta * d / tot

    # --- Si3N4 window contribution ---
    # The C1 window is 90 nm of Si3N4 sitting in front of the 250 nm Al. A
    # photon above the nitrogen K edge (409.9 eV) can ionise N in that window
    # and inject the photoelectron into the crystal, giving a further shelf
    # whose edge lies at E minus the N K binding energy.
    #
    # This term switches OFF below 409.9 eV, which is a real discontinuity in
    # the response: carbon (277 eV) and nitrogen (392 eV) sit below it, oxygen
    # (525 eV) and everything above sit on the other side. So the light
    # elements do not share a single smooth response curve.
    #
    # The 15 um Si support grid is deliberately NOT modelled as a spectral
    # feature. At 78% open it removes 22% of the flux, but 15 um of silicon is
    # optically thick to every energy in this window, so blocked photons are
    # simply lost rather than redistributed - a flat normalisation, not a
    # lineshape term. Its own Si fluorescence would appear as a fixed 1.740
    # keV peak; measured on the gold-masked carbon spectrum it is 15 counts
    # against 34,000 of carbon, i.e. negligible.
    wn = pars.a[P_WINDOW]
    if wn > 0.0 and E > N_K_EDGE:
        es3 = pars.a[P_SHELFE] if pars.a[P_SHELFE] > 1e-6 else SHELF_REF
        fw = wn * np.exp(-E / es3)
        c_lo3 = float(pars.centroid(max(E - N_K_EDGE, 1e-4)))
        lo_w = max(int(np.floor(c_lo3)), 0)
        hi_w = int(np.floor(c))
        if fw > 0.0 and hi_w > lo_w:
            xw = np.arange(lo_w, hi_w, dtype=float)
            sig = w / np.sqrt(KW)
            S1, S2 = de_slopes(pars, E)
            d = de_shelf(xw, c, sig, c_lo3, sig, S1, S2)
            tot = d.sum()
            if tot > 0:
                f[lo_w:hi_w] += fw * beta * d / tot

    return f


def element_profile(lines, pars, n_channels, tail_amp_fn=None,
                    tail_len_fn=None, do_tail=True):
    """Sum of line profiles for one element.

    lines  sequence of (energy_keV, relative_intensity)
    tail_amp_fn, tail_len_fn  callables E -> tail amplitude / length,
                              normally from the detector model
    """
    f = np.zeros(n_channels, dtype=float)
    for E, beta in lines:
        ta = tail_amp_fn(E) if tail_amp_fn else 0.0
        tl = tail_len_fn(E) if tail_len_fn else 0.0
        f += line_profile(E, beta, pars, n_channels, tail_amp=ta,
                          tail_len=tl, do_tail=do_tail)
    return f
