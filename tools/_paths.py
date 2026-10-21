"""Where the tools find data that is not in the repository.

Scripts take their input paths on the command line. When one is left out,
the default is looked up under the directories in SCRAPPYFIT_GEOPIXE_DATA,
separated by os.pathsep (';' on Windows): typically a GeoPIXE source tree
(test_data/, data/), a GeoPIXE install (detector files, run folders) and
wherever the .dam files are kept. The same variable serves the tests
(tests/labdata.py).

    import _paths                      # after putting tools/ on sys.path
    spec = _paths.geopixe_path('test_data/fits/CSIRO/donut2x-2-whole.spec',
                               given=args.spec)
"""
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
ENV = 'SCRAPPYFIT_GEOPIXE_DATA'


def roots():
    return [r for r in os.environ.get(ENV, '').split(os.pathsep) if r]


def find(rel):
    """rel under the first root that has it, or None."""
    for r in roots():
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return None


def geopixe_path(rel, given=None):
    """`given` if set, else rel under a SCRAPPYFIT_GEOPIXE_DATA root; exits
    with a message naming both when neither exists."""
    if given:
        return given
    p = find(rel)
    if p is None:
        sys.exit('%s not found: pass its path on the command line, or add the '
                 'directory holding it to %s (now %r)'
                 % (rel, ENV, os.environ.get(ENV, '')))
    return p


def lab_spectrum(name, root=None):
    """A lab reference spectrum: root/tests/data/name when root (another
    checkout) has it, else wherever tests/labdata.py finds it; exits when
    neither has it."""
    if root:
        p = os.path.join(root, 'tests', 'data', name)
        if os.path.isfile(p):
            return p
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import labdata
    p = labdata.lab_spectrum(name)
    if p is None:
        sys.exit('lab spectrum %s not found: set %s (tests/labdata.py)'
                 % (name, labdata.LAB_SPECTRA_ENV))
    return str(p)
