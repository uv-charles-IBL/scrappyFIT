"""The attenuation datasets, and that the mixed rule really switches."""

import pytest

from scrappyfit.physics.gpdb import Database


@pytest.fixture(scope='module')
def db():
    return Database()


def test_every_advertised_dataset_actually_works(db):
    """mac_datasets() is what the GUI offers. Every name in it must return a
    number for a mid-range case, or the dropdown lies."""
    for name in db.mac_datasets():
        v = db.mu(26, 6.404, name)          # Fe Ka, comfortably in range
        assert v is not None, '%s returned None for Fe Ka' % name
        assert 10.0 < v < 500.0, '%s gave mu=%.4g for Fe Ka' % (name, v)


def test_sabbatucci_aliases_all_resolve(db):
    """The file is MAC_SabbatucciSalvat2016.txt and nobody types that. Every
    alias must reach the same table - a bare KeyError used to look like a
    missing data file rather than a misspelt name."""
    ref = db.mu(14, 6.404, 'sabbatuccisalvat2016')
    for alias in ('sabbatucci', 'sabbatuccisalvat', 'salvat'):
        assert db.mu(14, 6.404, alias) == pytest.approx(ref)
    for alias in ('henke', 'henke1993'):
        assert db.mu(14, 6.404, alias) == pytest.approx(
            db.mu(14, 6.404, 'henke1993'))
    for alias in ('ffast', 'chantler'):
        assert db.mu(14, 6.404, alias) == pytest.approx(
            db.mu(14, 6.404, 'ffast'))


def test_unknown_dataset_names_the_valid_ones(db):
    with pytest.raises(KeyError) as e:
        db.mu(14, 6.404, 'nonesuch')
    assert 'ffast' in str(e.value)


def test_datasets_actually_differ(db):
    """If two datasets returned identical numbers, one is not being loaded."""
    h = db.mu(26, 6.404, 'henke1993')
    f = db.mu(26, 6.404, 'ffast')
    s = db.mu(26, 6.404, 'sabbatucci')
    x = db.mu(26, 6.404, 'xcom')
    assert len({round(v, 3) for v in (h, f, s, x)}) == 4


def test_xcom_has_nothing_below_one_kev(db):
    """The reason the mixed rule exists at all."""
    assert db.mu(8, 0.525, 'xcom') is None
    assert db.mu(6, 0.277, 'xcom') is None
    assert db.mu(8, 0.525, 'ffast') is not None


def test_mixed_rule_switches_where_heirwegh_says(db):
    """FFAST below 1 keV and below the absorber's own K edge; XCOM above.

    Absorber is silicon, K edge 1.839 keV.
    """
    Z = 14
    for E in (0.277, 0.525, 1.487):          # sub-keV, and Al Ka under the edge
        assert db.mu(Z, E, 'mixed') == pytest.approx(db.mu(Z, E, 'ffast')), E
    for E in (2.013, 6.404):                 # P Ka and Fe Ka, above the edge
        assert db.mu(Z, E, 'mixed') == pytest.approx(db.mu(Z, E, 'xcom')), E
        assert db.mu(Z, E, 'mixed') != pytest.approx(db.mu(Z, E, 'ffast')), E


def test_a_characteristic_line_is_always_below_its_own_edge(db):
    """So self-absorption of an element's own Ka always takes the FFAST arm.
    Worth pinning: it looks like the mixed rule is not switching until you
    notice that Ka < K edge is a law, not a coincidence."""
    for Z in (14, 20, 26):
        ka = db.line_energy(Z)
        assert ka < db.edge[(Z, 'K')]
        assert db.mu(Z, ka, 'mixed') == pytest.approx(db.mu(Z, ka, 'ffast'))
