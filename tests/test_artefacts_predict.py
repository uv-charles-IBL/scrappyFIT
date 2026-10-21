"""artefacts.predict as the GUI calls it."""
import numpy as np
import pytest

from scrappyfit import config
from scrappyfit.session import Session

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')


def test_predict_accepts_and_uses_escape_fraction():
    """GUI on_artefacts passes escape_fraction=; predict() must accept it and
    scale each escape peak by it, not crash with TypeError."""
    from scrappyfit.physics import artefacts
    db = Session().db
    cal = (0.0016984, -0.37225)
    y = np.full(4096, 10.0)
    areas = {'Ca': 1e5, 'Ti': 2e5, 'Si': 1e6}
    els = [(14, 1), (20, 1), (22, 1)]
    frac = lambda E: 0.02 * E            # distinct per parent energy

    esc, sums, p = artefacts.predict(
        db, y, cal, els, e_low=0.2, e_high=6.6, areas=areas,
        resolution=lambda E: (32.74 ** 2 + 28.05 ** 2 * E) ** 0.5,
        escape_fraction=frac)

    by = {a.parents[0]: a for a in esc}
    assert {'Ca', 'Ti'} <= set(by)
    for name, Z in (('Ca', 20), ('Ti', 22)):
        E = db.line_energy(Z, 1)
        assert by[name].intensity == pytest.approx(areas[name] * frac(E))


def test_predict_escape_markers_follow_ge_crystal():
    """predict() hard-codes the Si Ka/edge; on a Ge detector escapes must sit
    at E - Ge Ka1 for parents above the Ge K edge, and none for those below."""
    from scrappyfit.physics import artefacts
    from scrappyfit.physics.escape import EscapeModel
    db = Session().db
    frac = EscapeModel(db, 32, 5000.0).fraction
    y = np.full(4096, 10.0)
    cal = (0.007, 0.0)
    areas = {'Zr': 1e6, 'Mo': 1e6, 'Fe': 1e6}
    els = [(40, 1), (42, 1), (26, 1)]

    esc, _, _ = artefacts.predict(
        db, y, cal, els, e_low=0.2, e_high=25.0, areas=areas,
        escape_fraction=frac, crystal_Z=32)

    by = {a.parents[0]: a for a in esc}
    assert {'Zr', 'Mo'} <= set(by)
    assert 'Fe' not in by
    ge_ka = db.lineE[32]['Ka1']
    for name, Z in (('Zr', 40), ('Mo', 42)):
        assert by[name].energy == pytest.approx(
            db.line_energy(Z, 1) - ge_ka, abs=1e-6)
        assert by[name].energy != pytest.approx(
            db.line_energy(Z, 1) - artefacts.SI_KA, abs=1e-3)


def test_predict_si_unchanged_and_other_crystals_have_no_escapes():
    """crystal_Z=14 must give exactly the markers the Si-only code gave; a
    crystal escape_energy.pro does not know (here Cd) gets none, not Si's."""
    from scrappyfit.physics import artefacts
    db = Session().db
    cal = (0.0016984, -0.37225)
    y = np.full(4096, 10.0)
    areas = {'Ca': 1e5, 'Ti': 2e5, 'Si': 1e6}
    els = [(14, 1), (20, 1), (22, 1)]

    old, _, _ = artefacts.predict(db, y, cal, els, areas=areas)
    si, _, _ = artefacts.predict(db, y, cal, els, areas=areas, crystal_Z=14)
    assert [(a.label, a.energy, a.intensity) for a in si] == \
        [(a.label, a.energy, a.intensity) for a in old]
    for a in si:
        E = db.line_energy({'Ca': 20, 'Ti': 22}[a.parents[0]], 1)
        assert a.energy == E - 1.7397
    # pinned from master eaea2c1 (Si-only predict), 2026-10-19
    assert [(a.label, round(a.energy, 4), round(a.intensity, 6)) for a in si]         == [('Ca esc', 1.9503, 175.319154), ('Ti esc', 2.7683, 212.552153)]

    cd, _, _ = artefacts.predict(db, y, cal, els, areas=areas, crystal_Z=48)
    assert cd == []
