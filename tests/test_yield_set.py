"""A data file finds its layered yield set (one .yield per UNKNOWN layer).

BUGS.md: "QUANTIFY bootstraps a one-layer matrix for 287424 though its
layered .yield files exist". The default-yield lookup only searches beside
the data file; the new scrappyfit.yieldset adds a remembered choice and a
search path. Synthetic spectrum and synthetic .yield files, no lab data.
"""
import os

import numpy as np
import pytest

from scrappyfit import config

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

GAIN, OFFSET = 0.01, 0.0
FILES = ['pvsk287424_bare_UNKNOWN-film.yield',
         'pvsk287424_bare_UNKNOWN-glass.yield',
         'pvsk287424_plateau_UNKNOWN-glass.yield']
# major-line yields (photons/ppm/uC/msr) per layer, elements [O, Si]
YIELDS = {FILES[0]: [[0.5, 1.5], [0.4, 1.2]],
          FILES[1]: [[0.5, 1.5], [0.4, 1.2]],
          FILES[2]: [[0.9, 2.5], [0.8, 2.2]]}


def _write_spectrum(path):
    x = np.arange(1024)
    E = GAIN * x + OFFSET
    y = np.full(x.size, 20.0)
    for e, area in ((0.525, 20000.0), (1.740, 30000.0)):
        s = 0.06 / GAIN
        c = (e - OFFSET) / GAIN
        y += area * np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    y = np.random.default_rng(1).poisson(y)
    path.write_text('\n'.join('%d %d' % (i, v) for i, v in enumerate(y)))


def _write_yields(folder):
    from scrappyfit.io.gpyield_write import write_yield
    from scrappyfit.physics.gpdb import Database
    db = Database()
    film = dict(zlist=[14, 8], afrac=[1 / 3, 2 / 3], thick=0.5,
                formula='SiO2', name='film')
    glass = dict(zlist=[14, 8, 11], afrac=[0.25, 0.6, 0.15], thick=1e4,
                 formula='Na2O.SiO2', name='glass')
    folder.mkdir()
    for name in FILES:
        unknown = 1 if name.endswith('film.yield') else 2
        write_yield(folder / name, db, [(8, 1), (14, 1)], [film, glass],
                    np.array(YIELDS[name]), unknown=unknown, title=name)
    return folder


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('SCRAPPYFIT_YIELD_STORE', str(tmp_path / 'yields.json'))
    data = tmp_path / '287424.txt'
    _write_spectrum(data)
    ydir = _write_yields(tmp_path / 'yields')
    return data, ydir


def test_data_file_finds_its_yield_set_on_the_search_path(setup):
    """Loading 287424 must find its per-layer .yield files off the data folder."""
    from scrappyfit.session import Session
    from scrappyfit import yieldset as ys
    data, ydir = setup
    ys.set_search_path([str(ydir)])
    s = Session()
    s.load(str(data))
    assert s.yield_set is not None
    assert s.yield_set.regions == ['bare', 'plateau']
    s.set_yield_set(s.yield_set, 'bare', 'test')
    got = {e.layer for e in s.yield_set.entries('bare')}
    assert got == {'film', 'glass'}


def test_quantify_yields_uses_each_layers_file(setup):
    """QUANTIFY must read each UNKNOWN layer's file, not bootstrap one layer."""
    from scrappyfit.io.gpyield import read_yield
    from scrappyfit.physics.geometry import Geometry
    from scrappyfit.session import Session
    from scrappyfit import yieldset as ys
    data, ydir = setup
    ys.set_search_path([str(ydir)])
    s = Session()
    s.load(str(data))
    s.set_calibration(GAIN, OFFSET)
    s.load_efficiency(s.builtin_efficiencies()[0])
    s.run_fit(['O', 'Si'])
    s.geometry = Geometry(charge_uC=1.0, solid_angle_msr=2.0)
    s.set_yield_set(s.yield_set, 'bare', 'test')

    rows = s.quantify_yields()
    names = {os.path.basename(r['file']) for r in rows if r['file']}
    assert names <= set(FILES[:2])
    assert {(r['element'], r['layer']) for r in rows if r['element'] == 'Si'} \
        == {('Si', 'glass'), ('Si', 'film')}
    assert 'Na' not in {r['element'] for r in rows}
    assert 'bootstrapped' not in s.quantify_basis
    assert any(n in s.quantify_basis for n in names)

    # one value, recomputed from the file
    d = read_yield(str(ydir / FILES[1]))[0]
    k = 1                                           # Si (O is 0)
    nk = d['intensity'].size // d['n']
    inten = d['intensity'].reshape(d['n'], nk)[k, :d['n_lines'][k]]
    # the database's strongest Si K line (the file stores it as float32)
    e_major = max(s.db.line_list(14, 1), key=lambda t: t[1])[0]
    assert e_major == pytest.approx(float(d['e'].reshape(d['n'], nk)[k, 0]),
                                    abs=1e-5)
    Y = float(d['yield'][d['unknown'] - 1, k])
    photons = s.areas()['Si'] / float(s.efficiency(e_major))
    want = photons / (1.0 * 2.0 * Y * float(inten.sum()))
    row = next(r for r in rows if r['element'] == 'Si'
               and r['layer'] == 'glass')
    assert row['ppm'] == pytest.approx(want, rel=1e-9)


def test_remembered_choice_is_reported_as_remembered(setup):
    """A yield choice remembered for a data file must come back as such."""
    from scrappyfit import yieldset as ys
    data, ydir = setup
    f = str(ydir / FILES[0])
    ys.remember(str(data), [f], 'bare')
    found, source = ys.find_yield_set(str(data))
    assert found is not None
    assert 'remembered' in source


# -- the GUI: Sample model, Layer stack and QUANTIFY follow the yield set --

def test_gui_quantify_reads_the_yield_set_and_says_so(
        qtbot, setup, monkeypatch):
    QtCore = pytest.importorskip('PyQt5.QtCore')
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit import yieldset as ys
    from scrappyfit.gui.app import MainWindow
    data, ydir = setup
    ys.set_search_path([str(ydir)])
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    try:
        assert w.open_path(str(data))
        s = w.session
        assert s.yield_set is not None and s.yield_region is None
        assert w.sample.rb_yields.isChecked()
        assert 'choose a region' in w.sample.lbl_yset.text()

        # choosing a region is remembered for this data file
        w.sample.regionPicked.emit('bare')
        assert s.yield_region == 'bare'
        assert ys.remembered(str(data))[1] == 'bare'
        txt = w.sample.lbl_yset.text()
        assert FILES[0] in txt and FILES[1] in txt and FILES[2] not in txt

        # the Layer stack opens on one of the files QUANTIFY reads, and lists
        # the whole set
        w.on_layers()
        assert os.path.basename(s.yield_path) in FILES[:2]
        assert 'QUANTIFY reads its UNKNOWN layer' in \
            w.layerview.lbl_yield.text()
        assert w.layerview.cmb_files.count() == 3

        w.ed_gain.setText('%.10g' % GAIN)
        w.ed_off.setText('%.10g' % OFFSET)
        w.ed_gain.editingFinished.emit()
        w.cmb_eff.setCurrentIndex(1)
        for sym in ('O', 'Si'):
            qtbot.mouseClick(w.elements.table.buttons[sym],
                             QtCore.Qt.LeftButton)
        qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                        timeout=60000)
        qtbot.mouseClick(w.btn_fit, QtCore.Qt.LeftButton)
        qtbot.waitUntil(lambda: not w.jobs.busy and not w.preview_pending,
                        timeout=60000)
        assert s.fit is not None
        w.ed_charge.setText('1.0')
        w.ed_dist.setText('20')
        w.ed_area.setText('1')
        w.on_quantify()
        log = w.log.toPlainText()
        assert 'QUANTIFY FAILED' not in log
        assert 'IN THE LAYER' in log and FILES[1] in log
        assert 'bootstrapped' not in s.quantify_basis
        assert s._conc_yields

        # bootstrapping instead is a choice, and is said to be one
        w.sample.rb_boot.setChecked(True)
        w.on_quantify()
        assert s.quantify_basis.startswith('bootstrapped')
        assert 'Yields: bootstrapped' in w.log.toPlainText()
    finally:
        w.close()


def test_gui_says_why_it_bootstraps_when_no_yield_is_found(qtbot, setup):
    pytest.importorskip('PyQt5.QtWidgets')
    from scrappyfit.gui.app import MainWindow
    data, _ = setup                     # search path left empty
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        assert w.open_path(str(data))
        assert w.session.yield_set is None
        assert not w.sample.rb_yields.isEnabled()
        t = w.sample.lbl_yset.text()
        assert 'bootstraps' in t and 'search' in t
        assert 'QUANTIFY will bootstrap' in w.log.toPlainText()
    finally:
        w.close()


# -- quantify_yields on a fit given directly (no fitting) -----------------

def _fake_fit(s, areas):
    import types
    names = list(areas)
    s._fit = types.SimpleNamespace(names=names,
                                   areas=[areas[n] for n in names],
                                   errors=[0.0] * len(names))


def _pb_file(folder, name, pb_m=None):
    """One-layer perovskite-like file with Pb M, I L and K, UNKNOWN = 1."""
    import json
    from scrappyfit.io.gpyield_write import write_yield
    from scrappyfit.physics.gpdb import Database
    lay = dict(zlist=[82, 53], afrac=[0.25, 0.75], thick=0.5, name='pvsk')
    p = folder / name
    write_yield(p, Database(), [(82, 3), (53, 2), (19, 1)], [lay],
                np.array([[0.1, 0.2, 0.3]]), unknown=1)
    if pb_m:
        sc = folder / (name[:-len('.yield')] + '.provenance.json')
        d = json.loads(sc.read_text()) if sc.exists() else {}
        d['pb_m'] = pb_m
        sc.write_text(json.dumps(d))
    return str(p)


def _session_with(tmp_path, paths, areas):
    from scrappyfit.physics.geometry import Geometry
    from scrappyfit.session import Session
    from scrappyfit.yieldset import YieldSet
    s = Session()
    s.load_efficiency(s.builtin_efficiencies()[0])
    _fake_fit(s, areas)
    s.geometry = Geometry(charge_uC=1.0, solid_angle_msr=1.0)
    s.set_yield_set(YieldSet(paths), None, 'test')
    return s


PB_AREAS = {'PbM5': 1000.0, 'PbM4': 500.0, 'PbM3': 300.0, 'IL': 2000.0,
            'K': 100.0}


def test_pb_m_counts_m5_m4_only_when_the_file_says_so(tmp_path):
    whole = _pb_file(tmp_path, 'a_whole.yield')
    share = _pb_file(tmp_path, 'b_m54.yield',
                     pb_m='Pb M yield counts the M5+M4 share only')
    rows = {}
    for p in (whole, share):
        s = _session_with(tmp_path, [p], PB_AREAS)
        rows[p] = next(r for r in s.quantify_yields() if r['element'] == 'Pb')
    # same Y in both files, so the ppm ratio is the photon ratio
    s = _session_with(tmp_path, [whole], PB_AREAS)
    every = s._element_photons(82, 3)[0]
    m54 = s._element_photons(82, 3, ('M5', 'M4'))[0]
    assert every > m54 > 0
    assert rows[share]['ppm'] / rows[whole]['ppm'] == pytest.approx(
        m54 / every, rel=1e-12)


def test_element_in_no_unknown_layer_is_listed_not_quantified(tmp_path):
    s = _session_with(tmp_path, [_pb_file(tmp_path, 'x.yield')], PB_AREAS)
    k = [r for r in s.quantify_yields() if r['element'] == 'K']
    assert len(k) == 1 and k[0]['layer'] == '-'
    assert 'not quantified' in k[0]['note'] and k[0]['ppm'] != k[0]['ppm']


def test_element_in_two_layers_says_the_rows_are_alternatives(setup):
    from scrappyfit.yieldset import YieldSet
    data, ydir = setup
    s = _session_with(ydir, [str(ydir / f) for f in FILES[:2]],
                      {'Si': 1000.0, 'O': 500.0})
    si = [r for r in s.quantify_yields() if r['element'] == 'Si']
    assert len(si) == 2
    assert all('ALL the signal' in r['note'] for r in si)
    assert all(r['ppm'] > 0 for r in si)
    assert isinstance(s.yield_set, YieldSet)


def test_region_mask_takes_its_pixel_share_of_the_charge(tmp_path):
    p = _pb_file(tmp_path, 'x.yield')
    s = _session_with(tmp_path, [p], PB_AREAS)
    full = next(r for r in s.quantify_yields() if r['element'] == 'I')
    m = np.zeros((256, 256), bool)
    m[:64, :] = True                                  # a quarter
    s.mask = m
    quarter = next(r for r in s.quantify_yields() if r['element'] == 'I')
    assert quarter['ppm'] == pytest.approx(4.0 * full['ppm'], rel=1e-12)
    assert s._yield_charge == (pytest.approx(0.25), pytest.approx(0.25))


def test_relative_unit_files_are_not_found_and_not_quantified(setup):
    import json
    from scrappyfit import yieldset as ys
    from scrappyfit.session import Session
    data, ydir = setup
    for f in FILES:
        (ydir / (f[:-len('.yield')] + '.provenance.json')).write_text(
            json.dumps({'yield_units': 'RELATIVE units - ratios only'}))
    ys.set_search_path([str(ydir)])
    found, why = ys.find_yield_set(str(data))
    assert found is None and 'search path' in why
    s = _session_with(ydir, [str(ydir / FILES[1])], {'Si': 1000.0})
    with pytest.raises(RuntimeError, match='RELATIVE'):
        s.quantify_yields()
    assert Session().yield_set is None


def test_an_exported_yield_does_not_displace_a_found_set(setup, tmp_path):
    from scrappyfit.session import Session
    from scrappyfit import yieldset as ys
    data, ydir = setup
    ys.set_search_path([str(ydir)])
    s = Session()
    s.load(str(data))
    found = s.yield_set
    assert found is not None
    exported = str(tmp_path / 'model.yield')
    s.set_yield(exported, 'exported')
    assert s.yield_set is found                 # QUANTIFY still reads it
    assert s.yield_path == exported             # the dialogs show the file
    s.set_yield(exported, 'loaded')             # an explicit pick replaces it
    assert s.yield_set.paths == [exported]


def test_m54_file_refuses_a_lumped_m_component(tmp_path):
    share = _pb_file(tmp_path, 'b_m54.yield',
                     pb_m='Pb M yield counts the M5+M4 share only')
    s = _session_with(tmp_path, [share], {'PbM': 1800.0, 'IL': 2000.0})
    pb = next(r for r in s.quantify_yields() if r['element'] == 'Pb')
    assert pb['ppm'] != pb['ppm'] and 'M5+M4' in pb['note']
    whole = _pb_file(tmp_path, 'a_whole.yield')
    s = _session_with(tmp_path, [whole], {'PbM': 1800.0, 'IL': 2000.0})
    pb = next(r for r in s.quantify_yields() if r['element'] == 'Pb')
    assert pb['ppm'] > 0 and not pb['note']
