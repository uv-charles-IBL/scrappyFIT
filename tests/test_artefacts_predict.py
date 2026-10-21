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


def test_predict_si_default_and_other_crystals_have_no_escapes():
    """crystal_Z=14 must give the default (Si) markers, at the db Si Ka_; a
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
        assert a.energy == E - db.lineE[14]['Ka_']
    # pinned from master eaea2c1 (Si-only predict), 2026-10-19; energies
    # re-pinned 2026-10-21 (fix/spectrum-display): db Si Ka_ 1.740, not
    # 1.7397, moves each Si escape marker down 0.3 eV; counts unchanged
    assert [(a.label, round(a.energy, 4), round(a.intensity, 6)) for a in si]         == [('Ca esc', 1.95, 175.319154), ('Ti esc', 2.768, 212.552153)]

    cd, _, _ = artefacts.predict(db, y, cal, els, areas=areas, crystal_Z=48)
    assert cd == []


def test_si_escape_markers_use_the_fit_escape_energy():
    """artefacts SI_KA/SI_K_EDGE must equal EscapeModel's e_alpha/k_edge (db Si Ka_, edge)."""
    from scrappyfit.physics import artefacts
    from scrappyfit.physics.escape import EscapeModel
    db = Session().db
    em = EscapeModel(db, 14, 5000.0)
    e_alpha, k_edge = artefacts.crystal_escape(db, 14)
    assert e_alpha == pytest.approx(em.e_alpha, abs=1e-9)
    assert k_edge == pytest.approx(em.k_edge, abs=1e-9)
    assert k_edge == pytest.approx(db.edge[(14, 'K')], abs=1e-9)

    y = np.full(4096, 10.0)
    cal = (0.0016984, -0.37225)
    areas = {'Ca': 1e5, 'Ti': 2e5, 'Si': 1e6}
    els = [(14, 1), (20, 1), (22, 1)]
    esc, _, _ = artefacts.predict(db, y, cal, els, areas=areas, crystal_Z=14)
    by = {a.parents[0]: a for a in esc}
    assert {'Ca', 'Ti'} <= set(by)
    for name, Z in (('Ca', 20), ('Ti', 22)):
        assert by[name].energy == pytest.approx(
            db.line_energy(Z, 1) - em.e_alpha, abs=1e-9)


@pytest.mark.xfail(strict=True, reason='BUGS.md: No Ge K-beta escape markers')
def test_ge_crystal_gets_kbeta_escape_markers():
    """predict() on Ge must also mark the K-beta escape (E - Ge Kb1) for parents above the edge."""
    from scrappyfit.physics import artefacts
    from scrappyfit.physics.escape import EscapeModel
    db = Session().db
    frac = EscapeModel(db, 32, 5000.0).fraction
    y = np.full(4096, 10.0)
    cal = (0.007, 0.0)
    areas = {'Zr': 1e6, 'Mo': 1e6}
    els = [(40, 1), (42, 1)]

    esc, _, _ = artefacts.predict(
        db, y, cal, els, e_low=0.2, e_high=25.0, areas=areas,
        escape_fraction=frac, crystal_Z=32)

    ge_ka = db.lineE[32]['Ka1']
    ge_kb = db.lineE[32]['Kb1']
    for name, Z in (('Zr', 40), ('Mo', 42)):
        E = db.line_energy(Z, 1)
        mine = [a for a in esc if a.parents == (name,)]
        beta = [a for a in mine if abs(a.energy - (E - ge_kb)) < 1e-6]
        alpha = [a for a in mine if abs(a.energy - (E - ge_ka)) < 1e-6]
        assert len(alpha) == 1
        assert len(beta) == 1, f'no Kb escape marker for {name}'
        b = beta[0]
        assert b.kind == 'escape' and name in b.label
        assert b.intensity < alpha[0].intensity
