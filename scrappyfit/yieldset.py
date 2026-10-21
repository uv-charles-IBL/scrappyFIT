"""The sample's GeoPIXE .yield files, and where to find them.

Why a SET of files
------------------
GeoPIXE quantifies every element from one column of the yield table, the
layer marked UNKNOWN (pixe_fit.pro:414, conc = area / yield at :767). It has
no per-element layer choice: a multilayer file gives the right answer only
for the elements that live in its UNKNOWN layer, and the way to quantify
another layer is to load another file. A layered sample is therefore
described by one file per UNKNOWN layer - and, when the stack differs across
the map (silver plateau, voids, a bare strip), one set per region. Run
287424's files are named for exactly that:

    pvsk287424_<region>_UNKNOWN-<layer>.yield

and that is the pattern parsed here. A file that does not follow it is a set
of one, its layer read from the file.

Where they are found
--------------------
GeoPIXE keeps no link from a spectrum to its yield except the yield path in
its fit setup (.pcm, fit_setup.pro:2598-2607), and no search path. Ours, in
order, first hit wins:

  1. the files remembered for this data file (chosen in Sample model);
  2. files beside the data whose name contains the data file's stem
     (287424.lmf -> pvsk287424_bare_UNKNOWN-ITO.yield), or the one
     session.find_yield_beside() accepts;
  3. the same rule in each folder of the search path.

Both the remembered choices and the search path live in a small JSON store,
~/.scrappyfit/yields.json, or the file SCRAPPYFIT_YIELD_STORE names.

Files whose .provenance.json says their yields are in RELATIVE units (the
lab's ratio-only files) are never picked up by the search: they cannot give
a concentration.
"""
import json
import os
import pathlib
import re

STORE_ENV = 'SCRAPPYFIT_YIELD_STORE'

_NAME = re.compile(r'^(?P<stem>.+?)_(?P<region>[^_]+)_UNKNOWN-(?P<layer>.+)'
                   r'\.yield$', re.IGNORECASE)


# -- the store -----------------------------------------------------------

def store_path():
    env = os.environ.get(STORE_ENV)
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / '.scrappyfit' / 'yields.json'


def _read_store():
    try:
        d = json.loads(store_path().read_text(encoding='utf-8'))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_store(d):
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding='utf-8')


def _key(data_path):
    try:
        return os.path.normcase(str(pathlib.Path(data_path).resolve()))
    except (OSError, ValueError, TypeError):
        return os.path.normcase(str(data_path))


def search_path():
    """The folders searched for a data file's yields, in order."""
    sp = _read_store().get('search_path') or []
    return [str(s) for s in sp if isinstance(s, str) and s]


def set_search_path(dirs):
    d = _read_store()
    d['search_path'] = [str(x) for x in dirs if str(x).strip()]
    _write_store(d)
    return search_path()


def remember(data_path, files, region=''):
    """Keep the yield files chosen for a data file (and the region)."""
    d = _read_store()
    d.setdefault('data', {})[_key(data_path)] = dict(
        files=[str(f) for f in files], region=region or '')
    _write_store(d)


def remembered(data_path):
    """(files, region) remembered for a data file, or ([], '')."""
    e = (_read_store().get('data') or {}).get(_key(data_path)) or {}
    files = [f for f in e.get('files') or [] if isinstance(f, str)]
    return files, str(e.get('region') or '')


def forget(data_path):
    d = _read_store()
    if (d.get('data') or {}).pop(_key(data_path), None) is not None:
        _write_store(d)


# -- one file, and a set -------------------------------------------------

def provenance(path):
    """The .provenance.json beside a .yield, as a dict ({} if none)."""
    p = str(path)
    if not p.lower().endswith('.yield'):
        return {}
    sc = pathlib.Path(p[:-len('.yield')] + '.provenance.json')
    try:
        d = json.loads(sc.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def is_relative_units(path):
    """True when the file's .provenance.json says its yields are RELATIVE."""
    u = provenance(path).get('yield_units', '')
    return isinstance(u, str) and 'RELATIVE' in u


def m54_only(path):
    """True when the provenance says the M yield is the M5 + M4 share only
    (key 'pb_m', as the lab's yield_v2_gp.py writes it)."""
    u = provenance(path).get('pb_m', '')
    return isinstance(u, str) and 'M5+M4' in u.replace(' ', '')


class YieldEntry:
    """One .yield file: its region, its UNKNOWN layer, and its contents."""

    def __init__(self, path):
        self.path = str(path)
        m = _NAME.match(os.path.basename(self.path))
        self.region = m.group('region') if m else ''
        self._layer = m.group('layer') if m else None
        self._data = None

    @property
    def data(self):
        """The file read by io.gpyield.read_yield (first yield set)."""
        if self._data is None:
            from .io.gpyield import read_yield
            self._data = read_yield(self.path)[0]
        return self._data

    @property
    def unknown(self):
        """GeoPIXE's UNKNOWN layer index, 1-based, clamped into the stack."""
        d = self.data
        n = len(d['layers'])
        return min(max(int(d.get('unknown') or 1), 1), n)

    @property
    def unknown_layer(self):
        return self.data['layers'][self.unknown - 1]

    @property
    def layer(self):
        """The UNKNOWN layer's name: from the file name, else the file."""
        if self._layer is None:
            try:
                self._layer = (self.unknown_layer.get('name')
                               or 'layer %d' % self.unknown)
            except Exception:
                self._layer = '?'
        return self._layer

    @property
    def name(self):
        return os.path.basename(self.path)

    def __repr__(self):
        return 'YieldEntry(%r, region=%r, layer=%r)' % (
            self.name, self.region, self.layer)


class YieldSet:
    """Yield files for one sample, grouped by region."""

    def __init__(self, paths, folder=None):
        seen, self.all = set(), []
        for p in paths:
            k = os.path.normcase(os.path.abspath(str(p)))
            if k not in seen:
                seen.add(k)
                self.all.append(YieldEntry(p))
        self.all.sort(key=lambda e: (e.region, e.layer.lower(), e.name))
        dirs = {os.path.dirname(e.path) for e in self.all}
        self.folder = folder or (dirs.pop() if len(dirs) == 1 else '')

    def __len__(self):
        return len(self.all)

    @property
    def paths(self):
        return [e.path for e in self.all]

    @property
    def regions(self):
        """Region names from the file names, sorted; [''] for unpatterned."""
        return sorted({e.region for e in self.all})

    def default_region(self):
        r = self.regions
        return r[0] if len(r) == 1 else None

    def entries(self, region=None):
        """Files for a region; unpatterned files ('' region) apply to all."""
        if region is None:
            return list(self.all)
        return [e for e in self.all if e.region in (region, '')]

    def describe(self, region=None):
        es = self.entries(region)
        where = self.folder or 'several folders'
        return '%d file%s in %s%s' % (
            len(es), '' if len(es) == 1 else 's', where,
            (', region %s' % region) if region else '')


# -- finding a data file's set -------------------------------------------

def _matches(stem, name):
    if len(stem) < 3 or not name.lower().endswith('.yield'):
        return False
    return re.search(r'(?<![0-9])%s(?![0-9])' % re.escape(stem), name,
                     re.IGNORECASE) is not None


def _scan(folder, stem):
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(folder, n)
        if _matches(stem, n) and os.path.isfile(p) and \
                not is_relative_units(p):
            out.append(p)
    return out


def find_yield_set(data_path, beside=None):
    """(YieldSet, source) for a data file, or (None, what was searched).

    beside: a single .yield already accepted next to the data (the rule in
    session.find_yield_beside), added to the files found there.
    Never raises.
    """
    try:
        p = pathlib.Path(data_path)
        stem = p.stem
    except (TypeError, ValueError):
        return None, 'no data file'
    files, region = remembered(p)
    files = [f for f in files if os.path.isfile(f)]
    if files:
        ys = YieldSet(files)
        ys.remembered_region = region
        return ys, 'remembered for this data file'
    here = _scan(str(p.parent), stem)
    if beside and beside not in here:
        here.append(beside)
    if here:
        return YieldSet(here), 'found beside the data'
    sp = search_path()
    for d in sp:
        hits = _scan(d, stem)
        if hits:
            return YieldSet(hits, folder=d), 'found on the yield search path'
    return None, ('no .yield named for "%s" beside the data or on the search '
                  'path (%s)' % (stem, '; '.join(sp) if sp else 'empty'))
