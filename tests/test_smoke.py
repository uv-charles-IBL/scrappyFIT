"""Smoke tests: the things that must not silently break.

Each of these guards a specific failure that actually happened during
development, so they are regression tests rather than coverage.
"""
import numpy as np
import pytest

from scrappyfit import config
from scrappyfit.session import Session

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')


def test_database_loads():
    s = Session()
    assert len(s.db.lineE) > 80


def test_k_lines_unchanged_by_rebuild():
    """The rebuild must touch L and M only. A changed K line would silently
    move every concentration."""
    from scrappyfit.physics.gpdb import Database
    a = Database()
    b = Database(lines_file='xray_lines_rebuilt.txt')
    for Z in a.lineI:
        for k in ('Ka1', 'Ka_', 'Kb_'):
            assert abs(a.lineI[Z].get(k, 0) - b.lineI[Z].get(k, 0)) < 1e-9


def test_mixed_mac_switches_at_the_edge():
    """Below the K edge FFAST, above it XCOM. Getting this backwards is
    silent and wrong."""
    s = Session()
    db = s.db
    below = db.mu(14, 1.740, 'mixed')          # Si K edge is 1.839
    assert abs(below - db.mu(14, 1.740, 'ffast')) < 1e-6
    above = db.mu(14, 2.500, 'mixed')
    assert abs(above - db.mu(14, 2.500, 'xcom')) < 1e-6


def test_light_elements_need_a_sub_kev_dataset():
    """XCOM and Sabbatucci start at 1 keV; selecting them for oxygen must
    return None rather than an extrapolated number."""
    db = Session().db
    assert db.mu(8, 0.525, 'xcom') is None
    assert db.mu(8, 0.525, 'ffast') is not None


def test_calcium_has_no_l_alpha_in_the_rebuild():
    """Ca cannot emit La - its 3d shell is empty."""
    from scrappyfit.physics.gpdb import Database
    db = Database(lines_file='xray_lines_rebuilt.txt')
    assert db.lineI[20].get('La_', 0) == 0.0
    assert db.lineI[20].get('Ll', 0) > 0.1


def test_fit_runs_and_is_reproducible():
    import pathlib
    p = pathlib.Path(__file__).parent / 'data' / 'quartz.txt'
    if not p.exists():
        pytest.skip('no bundled test spectrum')
    s = Session()
    s.load(str(p))
    s.set_calibration(0.0016984, -0.37225)
    r = s.run_fit(['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si'])
    assert np.isfinite(r.reduced_chi2)
    assert s.areas()['Si'] > s.areas()['Al'] * 10
