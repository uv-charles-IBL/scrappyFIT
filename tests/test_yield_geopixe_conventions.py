"""What GeoPIXE reads out of a .yield file, and how write_yield must lay it out.

pixe_fit.pro quantifies every element from ONE column of the yield table,
yield[*, unknown-1], with `unknown` 1-based (geo_yield2.pro). It reads line 0
as the major line with intensity 1 and takes Y as that line's yield
(array_yield.pro: counts_per_ppm_uc = omega * eff(E_major) * T * Y).
"""

import numpy as np
import pytest

from scrappyfit.io.gpyield import read_yield
from scrappyfit.io.gpyield_write import from_session, write_yield
from scrappyfit.physics.geometry import YIELD_TO_GEOPIXE
from scrappyfit.physics.gpdb import Database
from scrappyfit.physics.layers import Layer, LayeredYieldModel
from scrappyfit.session import Session

# K, L and M elements; I L is the case where the database's sum-normalised
# list has I0 ~ 0.48 and GeoPIXE over-reported 2.1x
ELS = [(8, 1), (14, 1), (26, 1), (53, 2), (82, 3)]


@pytest.fixture(scope='module')
def db():
    return Database()


def _layers():
    return [dict(zlist=[14, 8], afrac=[1 / 3, 2 / 3], thick=0.1, name='top'),
            dict(zlist=[26], afrac=[1.0], thick=1.0, name='mid'),
            dict(zlist=[14], afrac=[1.0], thick=500.0, name='sub')]


def _totals():
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [1.5, 2.5, 3.5, 4.5, 5.5],
                     [0.5, 0.7, 0.9, 1.1, 1.3]])


@pytest.mark.parametrize('unknown', [1, 2, 3])
def test_unknown_round_trips(tmp_path, db, unknown):
    p = write_yield(tmp_path / 'u.yield', db, ELS, _layers(),
                    np.ones((3, len(ELS))), unknown=unknown)
    assert read_yield(p)[0]['unknown'] == unknown


def test_unknown_defaults_to_first_layer(tmp_path, db):
    p = write_yield(tmp_path / 'd.yield', db, ELS[:1], _layers()[:1],
                    np.ones((1, 1)))
    assert read_yield(p)[0]['unknown'] == 1      # not 0, which is column -1


@pytest.mark.parametrize('bad', [0, 4, -1])
def test_unknown_out_of_range_refused(tmp_path, db, bad):
    with pytest.raises(ValueError):
        write_yield(tmp_path / 'b.yield', db, ELS, _layers(),
                    np.ones((3, len(ELS))), unknown=bad)


def _read_lines(d):
    nk = d['nk']
    I = np.asarray(d['intensity']).reshape(d['n'], nk)
    E = np.asarray(d['e']).reshape(d['n'], nk)
    return I, E


def test_line_zero_is_major_with_intensity_one(tmp_path, db):
    p = write_yield(tmp_path / 'l.yield', db, ELS, _layers(),
                    np.ones((3, len(ELS))))
    d = read_yield(p)[0]
    I, E = _read_lines(d)
    for k, (Z, sh) in enumerate(ELS):
        n = d['n_lines'][k]
        assert n >= 1, (Z, sh)
        assert I[k, 0] == pytest.approx(1.0)
        assert I[k, :n].max() == pytest.approx(1.0)       # strongest line
        assert np.all(np.diff(I[k, :n]) <= 1e-7)          # descending
        # line 0 is the database's strongest line
        e_major = max(db.line_list(Z, sh), key=lambda t: t[1])[0]
        assert E[k, 0] == pytest.approx(e_major, abs=1e-4)


def test_major_yield_times_intensities_is_total_shell_yield(tmp_path, db):
    T = _totals()
    imaj = np.array([max(i for _, i in db.line_list(Z, sh))
                     for Z, sh in ELS])
    p = write_yield(tmp_path / 't.yield', db, ELS, _layers(), T * imaj,
                    unknown=2)
    d = read_yield(p)[0]
    I, _ = _read_lines(d)
    for k in range(len(ELS)):
        n = d['n_lines'][k]
        got = np.asarray(d['yield'])[:, k] * I[k, :n].sum()
        assert np.allclose(got, T[:, k], rtol=1e-5)


def test_caller_line_tables_kept_in_order_and_scaled_to_line0(tmp_path, db):
    # absorption-weighted: the second line transmits better than its
    # emitted share; line 0 stays the major line the yield refers to
    tab = {(26, 1): [(6.404, 0.5), (6.391, 0.25), (7.058, 0.6)]}
    p = write_yield(tmp_path / 'c.yield', db, [(26, 1), (14, 1)],
                    _layers()[:1], np.ones((1, 2)), lines=tab)
    d = read_yield(p)[0]
    I, E = _read_lines(d)
    assert d['n_lines'][0] == 3
    assert np.allclose(I[0, :3], [1.0, 0.5, 1.2], rtol=1e-6)
    assert np.allclose(E[0, :3], [6.404, 6.391, 7.058], atol=1e-4)
    assert I[1, 0] == pytest.approx(1.0)          # Si from the database


def test_from_session_writes_geopixe_units_without_efficiency(tmp_path):
    s = Session()
    s._meta = [(14, 1), (26, 1)]
    s.efficiency = lambda e: 0.5                 # must NOT reach the file
    s.efficiency.source = 'test'
    matrix = {8: 0.53, 14: 0.47}
    p = from_session(s, tmp_path / 's.yield', matrix=dict(matrix),
                     beam_MeV=3.0)
    d = read_yield(p)[0]
    assert d['unknown'] == 1

    lym = LayeredYieldModel(s.db)
    lay = Layer(list(matrix), list(matrix.values()), 1e4, 'matrix')
    yy = lym.yields([lay], [14, 26], E0=3.0, theta_deg=135.0,
                    mac=s.options.mac, fy=s.options.fluor_yield,
                    elam_zmax=s.options.fluor_elam_zmax, n_steps=900)
    want = [yy[14] * YIELD_TO_GEOPIXE, yy[26] * YIELD_TO_GEOPIXE]
    assert np.allclose(d['yield'][0], want, rtol=1e-5)

    # absorption-weighted lines: Fe Kb (7.06 keV) escapes SiO2 better than
    # Fe Ka, so its written share exceeds the emitted one; still line 0 = 1
    I, _ = _read_lines(d)
    fe = sorted(s.db.line_list(26, 1), key=lambda t: -t[1])
    assert I[1, 0] == pytest.approx(1.0)
    kb = [j for j, (e, _) in enumerate(fe) if e > 7.0][0]
    assert I[1, kb] > fe[kb][1] / fe[0][1]
