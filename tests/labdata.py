"""Where the tests find data that is not in the repository.

Two kinds of data live outside git, and tests that need them skip cleanly
when they are absent:

SCRAPPYFIT_LAB_SPECTRA
    A directory holding the lab's reference spectra (287424_perovskite.txt,
    404001_mnox.txt, 404003_ga2o3.txt, 404004_sic.txt, quartz.txt). They are
    the UNT Ion Beam Lab's measurements and are not published with the code.
    Looked for, in order: this variable, this checkout's tests/data, and the
    main checkout's tests/data when this is a git worktree (a lab checkout
    keeps them there untracked, .gitignore). Without them the golden spectra
    skip, and the GUI journeys and smoke fit run on
    tests/data/synthetic_quartz.txt (tools/make_synthetic_spectrum.py).

SCRAPPYFIT_REQUIRE_LAB_DATA=1
    Turns those skips into failures, so a lab machine cannot pass the golden
    check by not having the data. Set it where the golden spectra must run.

SCRAPPYFIT_GEOPIXE_DATA
    One or more directories, separated by os.pathsep (';' on Windows, ':'
    elsewhere), searched in order for GeoPIXE's own example files:
    test_data/yields, data/Topaz/287427.dai, data/lmf/287424.lmf and an
    OMDAQ session folder 20231215. The usual roots are a GeoPIXE source
    tree and a GeoPIXE install.
"""
import functools
import os
import pathlib
import subprocess

DATA = pathlib.Path(__file__).parent / 'data'
SYNTHETIC = DATA / 'synthetic_quartz.txt'
LAB_SPECTRA_ENV = 'SCRAPPYFIT_LAB_SPECTRA'
REQUIRE_ENV = 'SCRAPPYFIT_REQUIRE_LAB_DATA'
GEOPIXE_DATA_ENV = 'SCRAPPYFIT_GEOPIXE_DATA'


@functools.lru_cache(maxsize=None)
def _main_checkout_data():
    """tests/data of the main checkout when this is a worktree, else None."""
    try:
        common = subprocess.run(
            ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
            cwd=str(DATA.parent), capture_output=True, text=True,
            timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not common:
        return None
    d = pathlib.Path(common).parent / 'tests' / 'data'
    return None if d.resolve() == DATA.resolve() else d


def lab_spectrum(name):
    """Path to one lab reference spectrum, or None if it is not here."""
    roots = []
    env = os.environ.get(LAB_SPECTRA_ENV)
    if env:
        roots.append(pathlib.Path(env))
    roots.append(DATA)
    main = _main_checkout_data()
    if main is not None:
        roots.append(main)
    for r in roots:
        p = r / name
        if p.is_file():
            return p
    return None


def lab_data_required():
    return os.environ.get(REQUIRE_ENV, '') not in ('', '0')


def missing_lab_data(what):
    """Skip, or fail under SCRAPPYFIT_REQUIRE_LAB_DATA=1."""
    import pytest
    msg = 'missing lab data: %s (tests/labdata.py)' % what
    if lab_data_required():
        pytest.fail(msg + '; %s is set' % REQUIRE_ENV)
    pytest.skip(msg)


def geopixe_file(rel):
    """Path to rel under the first SCRAPPYFIT_GEOPIXE_DATA root that has
    it, or None."""
    for root in os.environ.get(GEOPIXE_DATA_ENV, '').split(os.pathsep):
        if root:
            p = pathlib.Path(root) / rel
            if p.exists():
                return p
    return None
