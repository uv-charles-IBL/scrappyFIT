"""Profiles along lines, polylines, annuli and regions."""

import numpy as np
import pytest

from scrappyfit.analysis.profiles import (line_profile, polyline_profile,
                                          profile_table, radial_profile,
                                          region_stats)


def ramp(n=32):
    """Value equals the column index, so a horizontal cut is its own answer."""
    return np.tile(np.arange(n, dtype=float), (n, 1))


def test_horizontal_cut_recovers_the_ramp():
    img = ramp()
    d, v, s = line_profile(img, (0, 10), (31, 10))
    assert len(d) == 32
    assert d[0] == 0.0 and d[-1] == pytest.approx(31.0)
    assert v == pytest.approx(np.arange(32.0))
    assert s == pytest.approx(np.zeros(32))


def test_distance_uses_pixel_size():
    d, _, _ = line_profile(ramp(), (0, 10), (31, 10), pixel_size=2.5)
    assert d[-1] == pytest.approx(31.0 * 2.5)


def test_diagonal_length_is_euclidean():
    d, _, _ = line_profile(ramp(), (0, 0), (30, 30))
    assert d[-1] == pytest.approx(30.0 * np.sqrt(2.0))


def test_width_averages_across_and_reports_spread():
    """On a ramp that varies only along the path, a wide cut must not change
    the value and must report zero spread - the width is perpendicular."""
    img = ramp()
    d1, v1, s1 = line_profile(img, (2, 16), (29, 16), width=1)
    d5, v5, s5 = line_profile(img, (2, 16), (29, 16), width=5)
    assert v5 == pytest.approx(v1, abs=1e-9)
    assert s5 == pytest.approx(np.zeros_like(s5), abs=1e-9)
    assert s1 == pytest.approx(np.zeros_like(s1))


def test_width_spread_is_nonzero_across_a_gradient():
    """Cut ALONG the ramp's constant direction: now the width spans the
    gradient and the spread must show it."""
    img = ramp()
    _, v, s = line_profile(img, (16, 2), (16, 29), width=5)
    assert v == pytest.approx(np.full(len(v), 16.0))
    assert (s > 0.5).all()


def test_median_resists_a_hot_pixel():
    img = ramp()
    img[16, 10] = 1e6
    _, vm, _ = line_profile(img, (10, 14), (10, 18), width=5, reduce='median')
    _, va, _ = line_profile(img, (10, 14), (10, 18), width=5, reduce='mean')
    # Across the width the samples are {8, 9, x, 11, 12}. Normally x = 10 and
    # the median is 10; where the hot pixel sits, x = 1e6 and the median
    # becomes 11 - the rank shifts by one place. That one-count shift is the
    # whole cost of a corrupted pixel, against a mean that goes to 2e5.
    assert vm.min() == pytest.approx(10.0)
    assert vm.max() == pytest.approx(11.0)
    assert va.max() > 1e4


def test_nearest_versus_bilinear():
    """Off-grid samples: bilinear interpolates, nearest snaps."""
    img = ramp()
    _, vb, _ = line_profile(img, (0.5, 5), (10.5, 5), samples=11, order=1)
    _, vn, _ = line_profile(img, (0.5, 5), (10.5, 5), samples=11, order=0)
    assert vb[0] == pytest.approx(0.5)
    assert vn[0] in (0.0, 1.0)


def test_sampling_off_the_edge_clamps_rather_than_raising():
    d, v, _ = line_profile(ramp(), (-5, 5), (40, 5))
    assert np.isfinite(v).all()
    assert v[0] == pytest.approx(0.0)
    assert v[-1] == pytest.approx(31.0)


def test_polyline_accumulates_distance_without_double_counting():
    img = ramp()
    pts = [(0, 5), (10, 5), (10, 15)]
    d, v, _ = polyline_profile(img, pts)
    assert d[0] == 0.0
    assert d[-1] == pytest.approx(20.0)
    assert np.all(np.diff(d) > 0), 'a repeated vertex would give a zero step'


def test_polyline_matches_a_single_line_when_collinear():
    img = ramp()
    d1, v1, _ = line_profile(img, (0, 5), (30, 5))
    d2, v2, _ = polyline_profile(img, [(0, 5), (15, 5), (30, 5)])
    assert d2[-1] == pytest.approx(d1[-1])
    assert v2[-1] == pytest.approx(v1[-1])


def test_radial_profile_on_a_cone():
    """Value = distance from centre, so the radial mean IS the radius."""
    n = 65
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.hypot(xx - c, yy - c)
    r, mean, std, count = radial_profile(img, centre=(c, c), nbins=20, rmax=20)
    good = count > 0
    assert np.allclose(mean[good], r[good], atol=0.6)
    assert count.sum() > 0


def test_radial_profile_finds_a_ring():
    """The donut case: a shell at a known radius must peak there."""
    n = 81
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    img = np.exp(-0.5 * ((rr - 20.0) / 2.0) ** 2)
    r, mean, _, _ = radial_profile(img, centre=(c, c), nbins=40, rmax=40)
    assert abs(r[int(np.nanargmax(mean))] - 20.0) < 1.5


def test_region_stats_and_area_scaling():
    img = ramp()
    mask = np.zeros_like(img, dtype=bool)
    mask[4:8, 10:14] = True                 # 16 pixels, columns 10..13
    st = region_stats(img, mask)
    assert st['n'] == 16
    assert st['mean'] == pytest.approx(11.5)
    assert st['min'] == 10 and st['max'] == 13
    st2 = region_stats(img, mask, pixel_size=0.5)
    assert st2['area'] == pytest.approx(16 * 0.25)


def test_region_stats_ignores_nan_and_rejects_a_wrong_shape():
    img = ramp()
    img[5, 11] = np.nan
    mask = np.zeros_like(img, dtype=bool)
    mask[4:8, 10:14] = True
    assert region_stats(img, mask)['n'] == 15
    with pytest.raises(ValueError):
        region_stats(img, np.zeros((4, 4), dtype=bool))


def test_profile_table_puts_every_element_on_one_path():
    maps = {'Si': ramp(), 'Fe': ramp() * 2.0}
    d, series = profile_table(maps, (0, 5), (31, 5))
    assert set(series) == {'Si', 'Fe'}
    assert len(d) == len(series['Si']) == len(series['Fe'])
    assert series['Fe'] == pytest.approx(series['Si'] * 2.0)


def test_degenerate_line_is_an_error_not_a_nan():
    with pytest.raises(ValueError):
        line_profile(ramp(), (5, 5), (5, 5))
