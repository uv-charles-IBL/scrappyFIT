"""The Sabbatucci-Salvat table is not shipped (EGSnrc source is AGPL-3.0); it is
built on request into a per-user folder. Converter tests run offline on a
tiny synthetic photo_shellwise.data; behaviour tests guard the new policy."""

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

import scrappyfit
from scrappyfit.io import sabbatucci
from scrappyfit.physics.gpdb import Database

F32 = np.float32
ALIASES = ('sabbatucci', 'sabbatuccisalvat2016', 'sabbatuccisalvat', 'salvat')

# Z=1: no shells, 3 rows.  Z=2: 2 shells, 4 rows, E repeated (edge pair).
# Rows are (E_MeV, SigTot_barns, *shell columns).
Z1_ROWS = [(F32(0.001), F32(10.0)), (F32(0.002), F32(4.0)),
           (F32(0.004), F32(1.0))]
Z2_ROWS = [(F32(0.001), F32(20.0), F32(111.0), F32(222.0)),
           (F32(0.002), F32(8.0), F32(333.0), F32(444.0)),
           (F32(0.002), F32(16.0), F32(555.0), F32(666.0)),
           (F32(0.004), F32(2.0), F32(777.0), F32(888.0))]
WEIGHTS = {1: ('H', 1.0079), 2: ('He', 4.0026)}

# The bundled database, not whatever config resolves: an external GeoPIXE
# tree may hold its own copy of the table in dat/.
BUNDLED = str(Path(scrappyfit.__file__).parent / 'resources' / 'database')


def make_photo_shellwise():
    """Bytes in photo_shellwise.data layout for the two synthetic elements."""
    blocks = []
    for rows, ns in ((Z1_ROWS, 0), (Z2_ROWS, 2)):
        b = struct.pack('<hh', len(rows), ns)
        for r in rows:
            b += struct.pack('<%df' % (ns + 2), *r)
        blocks.append(b)
    head = 2 + 4 * 2
    pos = [head, head + len(blocks[0])]
    return struct.pack('<h2i', 2, *pos) + b''.join(blocks)


def synthetic_text():
    return sabbatucci.to_geopixe_text(
        sabbatucci.read_photo_shellwise(make_photo_shellwise()), WEIGHTS,
        zmax=2)


def no_network(monkeypatch):
    def boom(url):
        raise AssertionError('network used: %s' % url)
    monkeypatch.setattr(sabbatucci, '_download', boom)


def install(folder):
    (folder / sabbatucci.FILENAME).write_bytes(synthetic_text().encode('ascii'))


# -- A. converter, offline ---------------------------------------------------

def test_reader_returns_both_elements_and_skips_shell_columns():
    """A wrong Nshell stride would read shell columns as SigTot or shift rows."""
    t = sabbatucci.read_photo_shellwise(make_photo_shellwise())
    assert sorted(t) == [1, 2]
    for z, rows in ((1, Z1_ROWS), (2, Z2_ROWS)):
        e, s = t[z]
        assert e.dtype == np.float32 and len(e) == len(rows)
        assert list(e) == [r[0] for r in rows]
        assert list(s) == [r[1] for r in rows]


def test_to_geopixe_text_format():
    """Header, Element lines, Range/Npoints, repeated edge abscissa, LF ends."""
    txt = synthetic_text()
    assert '\r' not in txt
    assert '\nElements   2\n' in txt
    assert '\nElement    H     1     1.007900\n' in txt
    assert '\nElement    He    2     4.002600\n' in txt
    assert txt.endswith('EndElement\nEnd\n')
    he = txt[txt.index('\nElement    He'):]
    e0, e1 = 1000 * float(F32(0.001)), 1000 * float(F32(0.004))
    assert '\nRange      %.6E %.6E\n' % (e0, e1) in he
    assert '\nNpoints    4\n' in he
    rows = he.split('Data\n')[1].split('EndElement')[0].splitlines()
    assert len(rows) == 4
    edge = '  %.6E  ' % (1000 * float(F32(0.002)))
    assert sum(r.startswith(edge) for r in rows) == 2
    for r, line in zip(Z2_ROWS, rows):
        assert line == '  %.6E  %.6E' % (
            1000 * float(r[0]), float(r[1]) * 0.602214076 / 4.0026)


def test_fetch_installs_the_table_from_a_local_source(tmp_path, monkeypatch):
    """fetch() checks the pinned sha, converts and writes FILENAME, offline."""
    no_network(monkeypatch)
    data = make_photo_shellwise()
    src = tmp_path / 'photo_shellwise.data'
    src.write_bytes(data)
    monkeypatch.setattr(sabbatucci, 'SOURCE_SHA256',
                        hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(sabbatucci, 'OUTPUT_SHA256', None)   # real-file pin
    dat = tmp_path / 'dat'
    dat.mkdir()
    (dat / 'MAC_Henke1993.txt').write_text(
        'Element    H     1     1.007900\nElement    He    2     4.002600\n')
    dest = tmp_path / 'out'
    got = sabbatucci.fetch(dest_dir=dest, source=src, dat_dir=str(dat))
    assert got == dest / sabbatucci.FILENAME and got.exists()
    assert got.read_text() == synthetic_text()


def test_fetch_rejects_a_source_that_is_not_the_pinned_file(tmp_path,
                                                           monkeypatch):
    """A wrong file must raise ValueError naming the EGSnrc commit."""
    no_network(monkeypatch)
    src = tmp_path / 'photo_shellwise.data'
    src.write_bytes(make_photo_shellwise())      # the real pin cannot match
    with pytest.raises(ValueError, match=sabbatucci.EGSNRC_COMMIT):
        sabbatucci.fetch(dest_dir=tmp_path / 'out', source=src)
    assert not (tmp_path / 'out' / sabbatucci.FILENAME).exists()


# -- B. behaviour of the not-shipped policy ----------------------------------

def test_the_sabbatucci_table_is_not_shipped_in_the_package():
    """The AGPL-derived table must not sit in the MIT package's dat folder."""
    dat = Path(scrappyfit.__file__).parent / 'resources/database/dat'
    assert not (dat / sabbatucci.FILENAME).exists()


def test_selecting_sabbatucci_without_the_table_explains_how_to_get_it(
        tmp_path, monkeypatch):
    """Every alias raises SabbatucciMissing naming AGPL, EGSnrc, the command."""
    monkeypatch.setenv('SCRAPPYFIT_DATA_DIR', str(tmp_path))
    db = Database(BUNDLED)
    for alias in ALIASES:
        with pytest.raises(sabbatucci.SabbatucciMissing) as e:
            db.mu(26, 6.404, alias)
        assert isinstance(e.value, FileNotFoundError)
        for word in ('AGPL', 'EGSnrc', 'python -m scrappyfit.io.sabbatucci'):
            assert word in str(e.value), alias


def test_installed_table_is_read_from_the_user_folder_under_every_alias(
        tmp_path, monkeypatch):
    """All aliases return the same in-bracket value from the per-user table."""
    install(tmp_path)
    monkeypatch.setenv('SCRAPPYFIT_DATA_DIR', str(tmp_path))
    db = Database(BUNDLED)
    vals = [db.mu(2, 3.0, a) for a in ALIASES]
    assert vals[0] is not None
    assert all(v == pytest.approx(vals[0]) for v in vals)
    k = 0.602214076 / 4.0026            # barns/atom -> cm2/g for He
    # log-log between the post-edge 2 keV row (16 b) and the 4 keV row (2 b);
    # a package-shipped table would give a different number
    f = np.log(3.0 / 2.0) / np.log(2.0)
    assert vals[0] == pytest.approx(k * np.exp(np.log(16.0) + f * np.log(2.0 / 16.0)),
                                    rel=1e-5)
    assert db.mu(2, 0.5, 'sabbatucci') is None          # below first energy


def test_mac_installed_reports_whether_a_dataset_is_available(tmp_path,
                                                              monkeypatch):
    """mac_installed(name) is True for shipped sets and tracks the Sabbatucci file."""
    monkeypatch.setenv('SCRAPPYFIT_DATA_DIR', str(tmp_path))
    db = Database(BUNDLED)
    for name in ('ffast', 'henke1993', 'xcom', 'mixed'):
        assert db.mac_installed(name) is True
    assert db.mac_installed('sabbatucci') is False
    install(tmp_path)
    assert db.mac_installed('sabbatucci') is True
    with pytest.raises(KeyError):
        db.mac_installed('nonesuch')


@pytest.mark.skipif(not __import__('os').environ.get('SCRAPPYFIT_PHOTO_SHELLWISE'),
                    reason='set SCRAPPYFIT_PHOTO_SHELLWISE to a local copy of '
                           "EGSnrc's photo_shellwise.data")
def test_real_photo_shellwise_reproduces_the_v0_25_0_table(tmp_path,
                                                           monkeypatch):
    """End to end on EGSnrc's real file, offline: fetch() checks the source
    hash, converts, and checks the output against v0.25.0's table."""
    import os
    no_network(monkeypatch)
    p = sabbatucci.fetch(tmp_path, os.environ['SCRAPPYFIT_PHOTO_SHELLWISE'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() ==         sabbatucci.OUTPUT_SHA256
