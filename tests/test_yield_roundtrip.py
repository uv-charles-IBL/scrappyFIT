"""A .yield file's layer F array is an ATOM fraction (decode_formula.pro).
Writing weight fractions there gives a file that loads cleanly and reads back
with the wrong matrix; H and Pb differ in mass 200-fold, so they show it."""

import numpy as np

from scrappyfit.io.gpyield import read_yield
from scrappyfit.io.gpyield_write import from_session, write_yield
from scrappyfit.physics.gpdb import Database
from scrappyfit.session import Session

W = {1: 0.1, 82: 0.9}              # weight fractions: ~96 atom-% hydrogen


def _atom(db, w):
    n = {z: v / db.A[z] for z, v in w.items()}
    t = sum(n.values())
    return {z: v / t for z, v in n.items()}


def _check(layer, db):
    a = _atom(db, W)
    assert layer['Z'] == [1, 82]
    assert np.allclose(layer['F'], [a[1], a[82]], rtol=1e-5)
    assert np.allclose(layer['W'], [W[1], W[82]], rtol=1e-5)
    assert layer['F'][0] > 0.9                  # not the 0.1 written before


def test_from_session_writes_atom_fractions(tmp_path):
    s = Session()
    s._meta = [(14, 1), (26, 1)]                # as if Si and Fe were fitted
    path = from_session(s, tmp_path / 'hpb.yield', matrix=dict(W))
    y = read_yield(path)[0]
    _check(y['layers'][0], s.db)
    # the formula is read by decode_formula with weight=0: atom proportions
    assert y['weight'][0] == 0
    assert y['formula'][0].startswith('(H)0.9')


def test_write_yield_takes_atom_fractions_under_either_key(tmp_path):
    # 'wfrac' is the legacy name and has always meant atom fractions here;
    # yield_v2_outputs.py (287424) depends on it not being converted
    db = Database()
    a = _atom(db, W)
    common = dict(zlist=[1, 82], thick=1e4, formula='HPb', name='m')
    for i, L in enumerate([dict(common, afrac=[a[1], a[82]]),
                           dict(common, wfrac=[a[1], a[82]])]):
        p = write_yield(tmp_path / ('m%d.yield' % i), db, [(14, 1)], [L],
                        np.array([[1.0]]))
        _check(read_yield(p)[0]['layers'][0], db)
