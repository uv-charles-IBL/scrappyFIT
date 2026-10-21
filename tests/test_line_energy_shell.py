"""PixeDB.line_energy(Z, shell) must follow the shell, not always return K-alpha."""
import numpy as np
import pytest

from scrappyfit import config
from scrappyfit.session import Session

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')



def _ka_name(db, Z):
    return 'Ka_' if Z <= db.KA_S else 'Ka1'


def test_line_energy_shell1_is_k_alpha_for_every_element():
    """Shell 1 must stay bit-identical to the Ka_/Ka1 energy (guards the fix
    against changing K)."""
    db = Session().db
    for Z in range(11, 93):
        assert db.line_energy(Z, 1) == db.lineE.get(Z, {}).get(_ka_name(db, Z), 0.0)


def test_line_energy_shell2_is_l_alpha():
    """line_energy(Z, 2) returned K-alpha (Ag 22.16 keV) instead of L-alpha."""
    db = Session().db
    for Z in (47, 82):
        nm = 'La_' if Z <= db.LA_S else 'La1'
        assert db.line_energy(Z, 2) == pytest.approx(db.lineE[Z][nm], abs=1e-3)
    assert db.line_energy(47, 2) == pytest.approx(2.984, abs=0.01)
    assert db.line_energy(82, 2) == pytest.approx(10.55, abs=0.01)
    assert db.major_line(47, 2) == 'La_'
    assert db.major_line(82, 2) == 'La1'


def test_line_energy_shell2_switches_group_at_la_s():
    """L-alpha is the La_ group up to db.LA_S and La1 above, like line_list."""
    db = Session().db
    assert db.major_line(db.LA_S, 2) == 'La_'
    assert db.major_line(db.LA_S + 1, 2) == 'La1'
    assert db.line_energy(db.LA_S + 1, 2) == pytest.approx(
        db.lineE[db.LA_S + 1]['La1'], abs=1e-3)


def test_line_energy_shell3_is_m_alpha():
    """line_energy(82, 3) returned K-alpha (74.96 keV) instead of Pb Ma1."""
    db = Session().db
    assert db.line_energy(82, 3) == pytest.approx(db.lineE[82]['Ma1'], abs=1e-3)
    assert db.line_energy(82, 3) == pytest.approx(2.345, abs=0.01)


def test_line_energy_missing_shell_line_is_zero():
    """An element with no L (or M) line gives 0.0, not its K-alpha."""
    db = Session().db
    Z = 3                                  # Li has no L-alpha
    assert db.lineE.get(Z, {}).get('La_', 0.0) == 0.0
    assert db.line_energy(Z, 2) == 0.0
    assert db.lineE.get(Z, {}).get('Ma1', 0.0) == 0.0
    assert db.line_energy(Z, 3) == 0.0


def test_predict_escape_for_ag_l_parent_uses_l_alpha():
    """artefacts.predict for an Ag L parent must put the Si escape at
    Ag La - 1.74 keV, not drop it because line_energy gave 22 keV."""
    from scrappyfit.physics import artefacts
    db = Session().db
    cal = (0.0016984, -0.37225)
    y = np.full(4096, 10.0)
    esc, _, _ = artefacts.predict(
        db, y, cal, [(47, 2)], e_low=0.2, e_high=6.5, areas={'AgL': 1e6},
        escape_fraction=lambda E: 0.01)
    assert len(esc) == 1
    assert esc[0].energy == pytest.approx(db.lineE[47]['La_'] - db.lineE[14]['Ka_'], abs=1e-9)
