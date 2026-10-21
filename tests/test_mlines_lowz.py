"""M lines for Z = 45-72 (no table data): I, Cs from xraylib; Ag with borrowed rates."""
import pytest
from scrappyfit.physics.gpdb import Database

xl = pytest.importorskip('xraylib')


def _energies(d):
    return sorted(round(e, 3) for v in d.values() for e, _ in v)


def test_iodine_and_caesium_m_lines():
    db = Database()
    i = db.m_subshell_lines(53)
    assert {'M1', 'M2', 'M3', 'M4', 'M5'} <= set(i) and 'Mz' not in i      # no 4f: M4/M5 are the Mz lines
    for e in (0.497, 0.509, 0.688, 0.744, 0.825, 0.881, 0.949):
        assert e in _energies(i)
    cs = db.m_subshell_lines(55)
    for e in (0.564, 0.567, 0.767, 0.834, 0.921, 0.986, 1.045):
        assert any(abs(e - x) < 0.003 for x in _energies(cs))


def test_silver_borrows_rates():
    db = Database()
    ag = db.m_subshell_lines(47)
    assert abs(ag['M5'][0][0] - 0.311) < 0.003 and abs(ag['M4'][0][0] - 0.310) < 0.003
    assert not any(e > 0.75 for v in ag.values() for e, _ in v)           # no 5p final states in Ag


def test_gold_groups_unchanged():
    db = Database()
    au = db.m_subshell_lines(79)
    assert [round(e, 3) for e, _ in au['Mz']][:2] == [1.647, 1.66]
    assert round(au['M5'][0][0], 3) == 2.122 and round(au['M4'][0][0], 3) == 2.203


def test_components_built_without_table_lines():
    from scrappyfit.session import Session
    s = Session(); s.cal = (0.0017, -0.37); s.options.e_low, s.options.e_high = 0.2, 6.5
    names = [c.name for c in s.build_components([(53, 3), (47, 3)])]
    assert 'IM5' in names and 'AgM5' in names
