"""Solid angle, the absolute normalisation, and the silicon escape prefactor."""

import math

import pytest

from scrappyfit.physics.geometry import (Geometry, live_fraction,
                                         solid_angle, solid_angle_from_area,
                                         yield_normalisation)


def test_solid_angle_area_and_diameter_agree():
    """25 mm2 quoted as an area, or as the equivalent diameter, must match."""
    d = 2.0 * math.sqrt(25.0 / math.pi)
    a = solid_angle_from_area(25.0, 30.0)
    b = solid_angle(30.0, d)
    assert a == pytest.approx(b, rel=1e-9)
    assert a == pytest.approx(25.0 / 900.0 * 1000.0, rel=1e-9)


def test_solid_angle_falls_as_inverse_square():
    near = solid_angle_from_area(25.0, 30.0)
    far = solid_angle_from_area(25.0, 60.0)
    assert near / far == pytest.approx(4.0, rel=1e-9)


def test_tilt_reduces_the_projected_area():
    straight = solid_angle_from_area(25.0, 30.0, 0.0)
    tilted = solid_angle_from_area(25.0, 30.0, 60.0)
    assert tilted / straight == pytest.approx(0.5, rel=1e-6)


def test_yield_normalisation_matches_calc_yield():
    """The constant from calc_yield.pro, to the precision it is written."""
    k = yield_normalisation(1.0, 1.0)
    expect = 1e-3 * 1e-3 * 6.02252e23 * 6.2418e12 / 12.5663706
    assert k == pytest.approx(expect, rel=1e-12)
    # a doubly charged beam delivers half the ions per microcoulomb
    assert yield_normalisation(1.0, 2.0) == pytest.approx(k / 2, rel=1e-12)


def test_live_fraction_from_either_form():
    assert live_fraction(dead_percent=20.0) == pytest.approx(0.8)
    assert live_fraction(real_time_s=100.0, live_time_s=75.0) == pytest.approx(0.75)
    assert live_fraction() == 1.0


def test_incomplete_geometry_refuses_rather_than_guessing():
    """A missing charge must raise, not silently behave as 1 uC - that would
    put a plausible-looking wrong number in front of someone."""
    g = Geometry(area_mm2=25.0, distance_mm=30.0)
    assert not g.complete()
    assert any('charge' in m for m in g.missing())
    with pytest.raises(ValueError):
        g.scale()


def test_scale_is_linear_in_each_input():
    base = Geometry(charge_uC=10.0, area_mm2=25.0, distance_mm=30.0).scale()
    twice_q = Geometry(charge_uC=20.0, area_mm2=25.0, distance_mm=30.0).scale()
    assert twice_q / base == pytest.approx(2.0, rel=1e-12)
    half_live = Geometry(charge_uC=10.0, area_mm2=25.0, distance_mm=30.0,
                         live_fraction=0.5).scale()
    assert half_live / base == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------- escape

def test_silicon_escape_prefactor_and_magnitude():
    """Published silicon escape fractions are about 1% just above the K edge,
    falling with energy. Taking GeoPIXE's detector.GAMMA as 1 - which omits
    omega_K (1 - 1/r) / 2 entirely - gave 42% for calcium.
    """
    from scrappyfit.physics.gpdb import Database
    from scrappyfit.physics.escape import EscapeModel

    db = Database()
    em = EscapeModel(db, 14, 116.5)          # 500 um of silicon
    assert 0.015 < em.prefactor < 0.030

    # nothing can escape below the Si K edge
    assert em.fraction(0.525) == 0.0         # O Ka
    assert em.fraction(1.740) == 0.0         # Si Ka, below 1.8389

    ca = em.fraction(3.692)
    assert 0.004 < ca < 0.020, 'Ca escape fraction %.4f is not physical' % ca

    # monotonically falling once above the edge
    for lo, hi in ((2.014, 2.622), (2.622, 3.692), (3.692, 6.404)):
        assert em.fraction(lo) > em.fraction(hi)


# ------------------------------------------------- the yield unit constant

def test_yield_to_geopixe_decomposes_into_its_factors():
    """The constant that was wrong by 60x. Assert the factors, not the
    number, so a future edit to one of them cannot quietly move it."""
    from scrappyfit.physics.geometry import (FOUR_PI, IONS_PER_MICROCOULOMB,
                                             MSR, PPM_PER_WT_PERCENT,
                                             YIELD_TO_GEOPIXE)
    expect = 1.0e21 * 1.0e-6 * IONS_PER_MICROCOULOMB * MSR / FOUR_PI
    assert YIELD_TO_GEOPIXE == pytest.approx(expect, rel=1e-12)
    assert YIELD_TO_GEOPIXE == pytest.approx(4.9671e23, rel=1e-3)
    assert PPM_PER_WT_PERCENT == 1.0e4


def test_scale_does_not_use_geopixes_own_constant():
    """calc_yield.pro's constant carries Avogadro because GeoPIXE's yield
    integral does not. Ours does, via N_A_over_A. Using GeoPIXE's constant
    here counts Avogadro twice; this pins the two apart."""
    from scrappyfit.physics.geometry import (yield_normalisation,
                                             PPM_PER_WT_PERCENT,
                                             YIELD_TO_GEOPIXE)
    g = Geometry(charge_uC=1.0, area_mm2=25.0, distance_mm=30.0)
    wrong = 1.0 * g.solid_angle_msr * yield_normalisation()
    assert g.scale() == pytest.approx(
        g.solid_angle_msr * YIELD_TO_GEOPIXE * PPM_PER_WT_PERCENT, rel=1e-12)
    # 60.2 against the derived constant; 61.6 against the one measured
    # from a GeoPIXE .yield file. The 2% between them is cross-section
    # interpolation, and is the honest size of the agreement.
    assert wrong / g.scale() == pytest.approx(60.22, rel=0.01)
