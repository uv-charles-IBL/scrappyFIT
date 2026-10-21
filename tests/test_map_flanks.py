"""Net window maps must not take background from channels off the spectrum."""
import numpy as np
import pytest

from scrappyfit import config
from scrappyfit.session import Session

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

CAL = (0.0016984, -0.37225)
NCH = 4096


def _flat_session(ch_min, n=400000, seed=0):
    """Flat continuum, no peaks, only in channels >= ch_min."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, n)
    y = rng.integers(0, 256, n)
    e = rng.integers(ch_min, NCH, n)
    s = Session()
    s.cal = CAL
    s.events = (x, y, e)
    s.full_spectrum = np.bincount(e, minlength=NCH).astype(float)
    s.spectrum = s.full_spectrum.copy()
    return s


def test_net_map_of_flat_continuum_is_zero_at_low_end():
    """C K net map of a flat, peak-free spectrum reads ~0, though its low
    flank lo-2d..lo-d reaches below channel 220 where no events exist."""
    s = _flat_session(220)
    m = s.element_map('C', subtract=True)
    peak = s.element_map('C', subtract=False)
    assert abs(m.mean()) < 0.05 * peak.mean()   # ~1.5 % counting noise; the bug read 32-51 %


def test_net_map_of_flat_continuum_is_zero_at_high_end():
    """A net map at 6.5 keV, whose high flank runs past channel 4095, reads
    ~0 on a flat spectrum instead of using the empty off-spectrum flank."""
    s = _flat_session(0)
    m = s.element_map(6.5, subtract=True)
    peak = s.element_map(6.5, subtract=False)
    assert abs(m.mean()) < 0.05 * peak.mean()   # ~1.5 % counting noise; the bug read 32-51 %


def test_flanks_clip_drop_and_keep():
    from scrappyfit.analysis.maps import flanks
    # both on the spectrum: unchanged, one peak-width away each side
    assert flanks(1000, 1009, 0, 4095) == ((980, 990), (1019, 1029))
    # low flank clipped to under half a width: dropped, high one kept
    assert flanks(1000, 1009, 987, 4095) == ((1019, 1029),)
    # clipped but still at least half a width: kept, clipped
    assert flanks(1000, 1009, 985, 4095) == ((985, 990), (1019, 1029))
    assert flanks(1000, 1009, 995, 1022) == ()


def test_map_info_says_which_background_was_used():
    s = _flat_session(220)
    s.element_map('C', subtract=True)
    assert s.map_info['background'] == 'flat, high flank only'
    s.element_map('Si', subtract=True)
    assert s.map_info['background'] == 'linear between flanks'
    s.element_map('Si', subtract=False)
    assert s.map_info['background'] == 'none'


def test_one_window_background_is_flat_from_that_window():
    from scrappyfit.analysis.maps import element_map
    # 2 events per channel 0..99 at pixel (0, 0); peak window 40-49 (10 ch)
    e = np.repeat(np.arange(100), 2)
    x = y = np.zeros_like(e)
    m = element_map(x, y, e, 40, 49, bg=((60, 79),), nbin=4)
    assert m[0, 0] == pytest.approx(0.0)
    assert m.sum() == pytest.approx(0.0)
