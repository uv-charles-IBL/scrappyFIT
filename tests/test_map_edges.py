"""Map smoothing must not darken the field edge (BUGS.md: window maps read low along the field edge)."""
import numpy as np
import pytest

from scrappyfit.analysis import maps


def test_smooth_keeps_a_flat_map_flat_to_the_edge():
    """A flat map stays flat to the edge (zero-padding read ~0.70 on the edge, ~0.49 in the corner)."""
    out = maps.smooth(np.ones((64, 64)), 1.0)
    assert out[0, 32] == pytest.approx(1.0, abs=1e-12)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(out, 1.0, atol=1e-12)


def test_flat_event_map_has_edge_pixels_equal_to_interior():
    """A uniform event field, mapped then smoothed, must not dim the outer pixels."""
    rng = np.random.default_rng(1)
    n = 800_000
    x = rng.integers(0, 256, n)
    y = rng.integers(0, 256, n)
    e = rng.integers(0, 1000, n)
    m = maps.smooth(maps.element_map(x, y, e, 0, 999, nbin=256, binning=4), 1.0)
    interior = m[8:-8, 8:-8].mean()
    edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    assert edge == pytest.approx(interior, rel=0.03)
