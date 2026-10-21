"""Where scrappyFIT finds its atomic database, and how that is configured.

The physics modules need GeoPIXE's data directory: ElamDB12.txt, the MAC_*.txt
tables, xsect_K/L/M.txt, hubbell.dat and the x-ray line tables. Resolution
order, first hit wins:

    1. an explicit path passed to set_database_path()
    2. the SCRAPPYFIT_DB environment variable
    3. a 'database' directory vendored inside this package
    4. the path recorded in ~/.scrappyfit/config.json
    5. a GeoPIXE install discovered next to this checkout

Keeping this in one place means an installation on a new machine - or off a
flash drive - is a single setting rather than an edit scattered through the
modules.
"""
import json
import os
import pathlib

_ENV = 'SCRAPPYFIT_DB'
_CFG = pathlib.Path.home() / '.scrappyfit' / 'config.json'
_override = None

# Files that must be present for a directory to count as a usable database.
REQUIRED = ('dat/ElamDB12.txt', 'dat/xray_lines.txt', 'dat/xsect_K.txt')


def _is_db(p):
    p = pathlib.Path(p)
    return all((p / r).exists() for r in REQUIRED)


def set_database_path(path):
    """Point scrappyFIT at a database directory for the rest of the session."""
    global _override
    p = pathlib.Path(path).expanduser().resolve()
    if not _is_db(p):
        raise ValueError('not a GeoPIXE database directory: %s' % p)
    _override = p
    return p


def save_database_path(path):
    """Persist the choice to ~/.scrappyfit/config.json."""
    p = set_database_path(path)
    _CFG.parent.mkdir(parents=True, exist_ok=True)
    _CFG.write_text(json.dumps({'database': str(p)}, indent=2))
    return p


def _candidates():
    if _override:
        yield _override
    env = os.environ.get(_ENV)
    if env:
        yield pathlib.Path(env)
    here = pathlib.Path(__file__).resolve().parent
    yield here / 'resources' / 'database'
    if _CFG.exists():
        try:
            yield pathlib.Path(json.loads(_CFG.read_text()).get('database', ''))
        except Exception:
            pass
    # a GeoPIXE tree sitting beside this checkout
    for up in (here.parent.parent, here.parent.parent.parent):
        for name in ('GeoPIXE-main', 'GeoPIXE'):
            yield up / name / 'Workspace' / 'main' / 'database'


def database_path(required=True):
    """Resolved database directory, or None when required=False and absent."""
    for c in _candidates():
        try:
            if c and _is_db(c):
                return pathlib.Path(c).resolve()
        except OSError:
            continue
    if required:
        raise FileNotFoundError(
            'No GeoPIXE database found. Set the %s environment variable, call '
            'config.save_database_path(...), or vendor one into '
            'scrappyfit/resources/database.' % _ENV)
    return None
