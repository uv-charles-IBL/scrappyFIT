"""Fetch and convert the Sabbatucci & Salvat (2016) photoelectric table.

scrappyFIT is MIT; EGSnrc's data directory is AGPL-3.0-or-later. So the
MAC_SabbatucciSalvat2016.txt table is not shipped (v0.25.0 was the last
release that did). It is built on request from EGSnrc's own file,

    HEN_HOUSE/data/photo_shellwise.data   at EGSNRC_COMMIT

and written to a per-user data folder (user_data_dir()), where
Database.load_mac() finds it. Nothing downloads on its own: selecting the
dataset without the table raises SabbatucciMissing, naming this module.

    python -m scrappyfit.io.sabbatucci              download, convert, install
    python -m scrappyfit.io.sabbatucci --source F   convert a local copy of F

Layout of photo_shellwise.data (little-endian, no record markers), from the
EGSnrc commit that added it (733200391f, 2017-08-03):

    NZ                         int16
    fpos(1..NZ)                int32, byte offset of each element's block
    NE, Nshell                 int16, int16          (at fpos(Z))
    NE rows of  E  SigTot  Sig(1..Nshell)            float32, MeV and barns/atom

Conversion: E in keV = 1000 E(MeV); mu/rho = SigTot N_A 1e-24 / A in cm^2/g,
with A taken from the Element lines of the bundled MAC_Henke1993.txt (the
same 92 atomic weights the 0.25.0 table carried; mass.txt's differ). EGSnrc
already repeats the abscissa at each edge, which is the GeoPIXE MAC format's
edge encoding, so the rows are written as they come.

The output reproduces v0.25.0's bundled file byte for byte (OUTPUT_SHA256).
"""
import argparse
import hashlib
import os
import pathlib
import struct
import sys
import tempfile

import numpy as np

FILENAME = 'MAC_SabbatucciSalvat2016.txt'
EGSNRC_COMMIT = 'bd49a47aae3938f49f4100b8e018091861408b19'   # master, 2026-09-28
SOURCE_URL = ('https://raw.githubusercontent.com/nrc-cnrc/EGSnrc/%s/'
              'HEN_HOUSE/data/photo_shellwise.data' % EGSNRC_COMMIT)
# Both hashes: sha256sum, 2026-10-21. SOURCE is the file at SOURCE_URL;
# OUTPUT is v0.25.0's committed table (git cat-file -p f46b045:scrappyfit/
# resources/database/dat/MAC_SabbatucciSalvat2016.txt), which convert()
# reproduces byte for byte.
SOURCE_SHA256 = 'bc025da416049d4712d06ae7f4556011b41a45a9ab6671bea56379caba545ce9'
OUTPUT_SHA256 = '2f22c2c9c923fc9cabf94d11d597118ecd55281ffe8430a516e7158c434458d2'
LICENCE = ('EGSnrc, Copyright National Research Council of Canada, '
           'GNU AGPL-3.0-or-later '
           '(https://github.com/nrc-cnrc/EGSnrc/blob/master/LICENCE.md)')
ZMAX = 92
N_A = 0.602214076             # Avogadro / 1e24

_ENV = 'SCRAPPYFIT_DATA_DIR'

HEADER = """\
; GeoPIXE tabulated mass attenuation coefficients.
; Total photoelectric cross sections from the EGSnrc distribution
; of the Sabbatucci & Salvat renormalized MCDF calculation
; (HEN_HOUSE/data/photo_shellwise.data), converted from
; barns/atom to cm^2/g via mu/rho = sigma N_A 1e-24 / A.
; Photoabsorption only; excludes coherent/incoherent scattering.
; As distributed by EGSnrc the grid starts at 1 keV, so this set
; does NOT cover the sub-keV region.
;
Dataset    SabbatucciSalvat2016
Reference  L. Sabbatucci, F. Salvat, Radiat. Phys. Chem. 121 (2016) 122-140
Quantity   photoelectric mass attenuation coefficient
Units      keV cm2/g
Elements   %d
;
; Edge discontinuities are encoded as repeated abscissae:
; two rows with identical energy bracket a jump. Never
; interpolate across such a pair.
;
"""


class SabbatucciMissing(FileNotFoundError):
    """The Sabbatucci-Salvat table was selected but is not installed."""


def user_data_dir():
    """Per-user folder for data scrappyFIT may not ship: ~/.scrappyfit/data,
    or $SCRAPPYFIT_DATA_DIR."""
    env = os.environ.get(_ENV)
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / '.scrappyfit' / 'data'


def installed_path():
    """Where the converted table lives once fetched (it may not exist)."""
    return user_data_dir() / FILENAME


def missing_message():
    return (
        'The Sabbatucci-Salvat 2016 MAC table is not installed. scrappyFIT '
        'does not ship it: it is converted from EGSnrc\'s '
        'HEN_HOUSE/data/photo_shellwise.data, distributed under %s. To '
        'download EGSnrc\'s file and convert it into %s, run\n'
        '    python -m scrappyfit.io.sabbatucci\n'
        'or call scrappyfit.io.sabbatucci.fetch().'
        % (LICENCE, user_data_dir()))


def read_photo_shellwise(data):
    """{Z: (E_MeV, SigTot_barns)} as float32 arrays, from the file's bytes."""
    nz = struct.unpack_from('<h', data, 0)[0]
    fpos = struct.unpack_from('<%di' % nz, data, 2)
    out = {}
    for z, p in enumerate(fpos, start=1):
        ne, ns = struct.unpack_from('<hh', data, p)
        rows = np.frombuffer(data, '<f4', ne * (ns + 2), p + 4)
        rows = rows.reshape(ne, ns + 2)
        out[z] = (rows[:, 0].copy(), rows[:, 1].copy())
    return out


def henke_atomic_weights(dat_dir):
    """{Z: (symbol, A)} from the Element lines of MAC_Henke1993.txt."""
    out = {}
    with open(os.path.join(dat_dir, 'MAC_Henke1993.txt'),
              errors='ignore') as fh:
        for line in fh:
            w = line.split()
            if len(w) == 4 and w[0] == 'Element':
                out[int(w[2])] = (w[1], float(w[3]))
    return out


def to_geopixe_text(tables, weights, zmax=ZMAX):
    """The GeoPIXE MAC text table (LF line ends) for Z = 1..zmax."""
    zs = [z for z in range(1, zmax + 1) if z in tables]
    parts = [HEADER % len(zs)]
    for z in zs:
        sym, A = weights[z]
        e_mev, sig = tables[z]
        e = [1000.0 * float(x) for x in e_mev]
        mu = [float(s) * N_A / A for s in sig]
        parts.append('Element    %-2s %4d %12.6f\n' % (sym, z, A))
        parts.append('Range      %.6E %.6E\n' % (e[0], e[-1]))
        parts.append('Npoints    %d\n' % len(e))
        parts.append('Data\n')
        parts.extend('  %.6E  %.6E\n' % r for r in zip(e, mu))
        parts.append('EndElement\n')
    parts.append('End\n')
    return ''.join(parts)


def convert(data, dat_dir=None):
    """photo_shellwise.data bytes -> the table's text. The atomic weights
    come from the bundled MAC_Henke1993.txt, whatever database is configured."""
    if dat_dir is None:
        dat_dir = str(pathlib.Path(__file__).resolve().parent.parent
                      / 'resources' / 'database' / 'dat')
    return to_geopixe_text(read_photo_shellwise(data),
                           henke_atomic_weights(dat_dir))


def _download(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch(dest_dir=None, source=None, dat_dir=None):
    """Download EGSnrc's photo_shellwise.data (or read `source`, a local
    copy), check it is the pinned file, convert it and install the table in
    dest_dir (default user_data_dir()). Returns the table's path.

    The downloaded file is EGSnrc's, under AGPL-3.0-or-later; calling this
    is the user's choice to obtain it. It is used only to build the table
    and is not kept."""
    data = (pathlib.Path(source).read_bytes() if source is not None
            else _download(SOURCE_URL))
    got = hashlib.sha256(data).hexdigest()
    if got != SOURCE_SHA256:
        raise ValueError(
            'photo_shellwise.data does not match EGSnrc commit %s '
            '(sha256 %s, expected %s)' % (EGSNRC_COMMIT, got, SOURCE_SHA256))
    text = convert(data, dat_dir).encode('ascii')
    if OUTPUT_SHA256 and hashlib.sha256(text).hexdigest() != OUTPUT_SHA256:
        raise ValueError(
            'converted table differs from the one v0.25.0 shipped; check '
            'MAC_Henke1993.txt, whose atomic weights it uses')
    dest = pathlib.Path(dest_dir) if dest_dir is not None else user_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest), suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(text)
        os.replace(tmp, str(dest / FILENAME))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest / FILENAME


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m scrappyfit.io.sabbatucci',
        description='Download EGSnrc\'s photo_shellwise.data (AGPL-3.0) and '
                    'convert it into the %s MAC table.' % FILENAME)
    ap.add_argument('--source', help='convert this local photo_shellwise.data '
                                     'instead of downloading it')
    ap.add_argument('--dest', help='install folder (default %s)'
                                   % user_data_dir())
    ns = ap.parse_args(argv)
    if ns.source is None:
        print('Downloading %s\n(%s).' % (SOURCE_URL, LICENCE))
    path = fetch(ns.dest, ns.source)
    print('Wrote %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
