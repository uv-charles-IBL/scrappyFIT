"""Detector geometry and the absolute yield normalisation.

Why absolute quantification is worth the trouble
-----------------------------------------------
Normalising a result to 100 wt% is convenient and it cancels the solid angle,
the charge and the atoms-per-gram constant in one stroke. It is also a way to
be confidently wrong: a fit that has lost a third of its intensity to a
mismodelled peak shape, or that is missing an element entirely, still sums to
100% afterwards and looks fine. The normalisation destroys precisely the
evidence that something is missing.

An absolute calculation cannot hide that. If the concentrations sum to 68% or
to 140%, the analysis is telling you so. That number is the single most useful
diagnostic in the whole chain, and it only exists if the solid angle and the
charge are known.

The GeoPIXE convention
----------------------
From calc_yield.pro, yields are computed per microcoulomb per millisteradian:

    solid = mgm * omega * N_A * ions_per_uC / (4 pi * cos(beam) * state)

with mgm = 1e-3 (g to mg), omega = 1e-3 sr (one msr), N_A = 6.02252e23,
ions_per_uC = 6.2418e12. So a stored yield is

    counts per ppm per uC per msr

and a measurement scales it by the actual charge and the actual solid angle:

    counts = concentration x yield x Q x Omega

which inverts to give a concentration with no normalisation anywhere.

Where this goes wrong in practice
---------------------------------
Three inputs, each with a characteristic failure:

  charge      the weakest link. Current integrated on the sample rather than
              in a Faraday cup reads high unless secondary electrons are
              suppressed, sometimes by tens of percent. It is also the one
              people most often take on trust.
  solid angle geometric and reliable IF the distance is really what the
              drawing says. A few mm of error on a short working distance is
              a large fractional error, since it goes as 1/r^2.
  dead time   counts that were never recorded. Correcting for it scales
              everything up; ignoring it makes every concentration low by the
              same factor, which is invisible in a normalised result and
              obvious in an absolute one.
"""

import math

# calc_yield.pro
AVOGADRO = 6.02252e23
IONS_PER_MICROCOULOMB = 6.2418e12
FOUR_PI = 12.5663706
G_TO_MG = 1.0e-3
MSR = 1.0e-3                      # one millisteradian, in steradian


def solid_angle(distance_mm, diameter_mm, tilt_deg=0.0, square=False):
    """Solid angle in MILLISTERADIAN, following solid_angle.pro.

    distance_mm  detector face to sample
    diameter_mm  active diameter (or side, for a square detector)
    tilt_deg     detector tilt away from the sample normal
    square       True for a square active area

    Omega = A cos(tilt) / r^2, with A the active area. The small-angle form is
    used, as GeoPIXE does; at the working distances of a PIXE chamber the
    exact spherical-cap result differs by well under a percent, and the
    distance is never known that well anyway.
    """
    if distance_mm <= 1e-3 or diameter_mm <= 0:
        return 0.0
    if square:
        area = diameter_mm ** 2
    else:
        area = math.pi * (diameter_mm / 2.0) ** 2
    omega_sr = area * math.cos(math.radians(tilt_deg)) / (distance_mm ** 2)
    return omega_sr * 1000.0          # sr -> msr


def solid_angle_from_area(area_mm2, distance_mm, tilt_deg=0.0):
    """Same, when the datasheet gives an active AREA rather than a diameter -
    which is how Amptek quotes it (25 mm2, 70 mm2)."""
    if distance_mm <= 1e-3 or area_mm2 <= 0:
        return 0.0
    return (area_mm2 * math.cos(math.radians(tilt_deg))
            / (distance_mm ** 2)) * 1000.0


# ----------------------------------------------------------------- the fix
#
# yield_normalisation() below is a faithful port of calc_yield.pro, and it is
# the WRONG constant for this package's yield model. GeoPIXE's yield integral
# does not carry atoms per gram, so calc_yield.pro supplies Avogadro itself.
# physics/yields.py DOES carry it, as N_A_over_A(). Using GeoPIXE's constant
# on this model therefore counts Avogadro twice, and gets the depth and
# concentration units wrong as well. The net error was a factor of 61.6, in
# the direction that makes every absolute concentration too small.
#
# The right constant for THIS model, derived term by term:
#
#   1e21    yields.py multiplies by N_A_over_A, which is Avogadro x 1e-24,
#           and integrates over a depth in mg/cm2 rather than g/cm2 (x 1e3).
#           So its output is the true 4pi yield per unit mass fraction x 1e-21
#   1e-6    GeoPIXE quotes yields per ppm, not per unit mass fraction
#   6.2418e12   protons per microcoulomb
#   1e-3    one millisteradian expressed in steradian
#   / 4pi   the yield above is into the whole sphere
#
# Checked, not just derived: running LayeredYieldModel on the layer model
# inside a GeoPIXE .yield file and dividing GeoPIXE's stored yields by ours
# gives 4.859e23 with a 2.0% spread over Z = 15-30, against the 4.967e23
# below - agreement to 2.2%, the remainder being cross-section interpolation.
# The two were computed independently, so this is a real check on both.

YIELD_TO_GEOPIXE = 1.0e21 * 1.0e-6 * IONS_PER_MICROCOULOMB * MSR / FOUR_PI
"""Converts a yields.py yield into GeoPIXE's counts per ppm per uC per msr."""

PPM_PER_WT_PERCENT = 1.0e4


def yield_normalisation(cos_beam=1.0, charge_state=1.0):
    """The constant that turns a bare cross-section integral into
    counts per ppm per uC per msr, exactly as calc_yield.pro forms it.

    cos_beam      cosine of the beam's angle of incidence on the surface. A
                  tilted target presents a longer path per unit depth, which
                  raises the yield.
    charge_state  the ion's charge state; a doubly charged beam delivers half
                  the ions per microcoulomb.
    """
    cb = max(abs(cos_beam), 1e-6)
    return (G_TO_MG * MSR * AVOGADRO * IONS_PER_MICROCOULOMB
            / (FOUR_PI * cb * float(charge_state)))


def live_fraction(real_time_s=None, live_time_s=None, dead_percent=None):
    """Fraction of the acquisition the detector was actually counting.

    Given either the two clocks or a dead-time percentage. Counts are divided
    by this, so ignoring it makes every concentration low by the same factor -
    invisible once normalised to 100%, obvious in an absolute result.
    """
    if dead_percent is not None:
        return max(1.0 - float(dead_percent) / 100.0, 1e-6)
    if real_time_s and live_time_s and real_time_s > 0:
        return max(float(live_time_s) / float(real_time_s), 1e-6)
    return 1.0


class Geometry:
    """Everything needed to turn counts into an absolute concentration."""

    def __init__(self, charge_uC=None, solid_angle_msr=None,
                 distance_mm=None, area_mm2=None, diameter_mm=None,
                 tilt_deg=0.0, cos_beam=1.0, charge_state=1.0,
                 live_fraction=1.0):
        self.charge_uC = charge_uC
        self._omega = solid_angle_msr
        self.distance_mm = distance_mm
        self.area_mm2 = area_mm2
        self.diameter_mm = diameter_mm
        self.tilt_deg = tilt_deg
        self.cos_beam = cos_beam
        self.charge_state = charge_state
        self.live_fraction = live_fraction

    @property
    def solid_angle_msr(self):
        if self._omega:
            return self._omega
        if self.distance_mm and self.area_mm2:
            return solid_angle_from_area(self.area_mm2, self.distance_mm,
                                         self.tilt_deg)
        if self.distance_mm and self.diameter_mm:
            return solid_angle(self.distance_mm, self.diameter_mm,
                               self.tilt_deg)
        return None

    def complete(self):
        return bool(self.charge_uC and self.solid_angle_msr)

    def missing(self):
        m = []
        if not self.charge_uC:
            m.append('charge (uC)')
        if not self.solid_angle_msr:
            m.append('solid angle (msr), or detector distance and area')
        return m

    def scale(self):
        """counts = concentration_in_wt_percent x yield x THIS.

        Uses YIELD_TO_GEOPIXE, not yield_normalisation - see the note above
        that constant. A concentration is divided by this, so every factor
        here that is too large makes the answer too small.
        """
        if not self.complete():
            raise ValueError('absolute quantification needs: '
                             + ', '.join(self.missing()))
        cb = max(abs(self.cos_beam), 1e-6)
        return (self.charge_uC * self.solid_angle_msr
                * YIELD_TO_GEOPIXE * PPM_PER_WT_PERCENT
                * self.live_fraction / (cb * float(self.charge_state)))

    def describe(self):
        o = self.solid_angle_msr
        return ('charge %s uC, solid angle %s msr%s, live fraction %.4g'
                % ('%.4g' % self.charge_uC if self.charge_uC else '?',
                   '%.4g' % o if o else '?',
                   (' (%.4g mm2 at %.4g mm)'
                    % (self.area_mm2, self.distance_mm))
                   if self.area_mm2 and self.distance_mm else '',
                   self.live_fraction))
