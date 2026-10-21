"""OMDAQ run log reader - where the integrated charge actually lives.

The LMF block header carries a dose counter, but it counts digitiser pulses,
not charge, and the conversion is not recorded in the file. OMDAQ writes the
integrated charge to a run log instead, in two equivalent forms that sit
beside the data:

    _<date>_<project>_RUNLOG.TXT       one stanza per run
    _<date>_<project>_RUNLOG_00.CSV    the same thing as a table

Both give 'Q (nC)' per run, along with the ion, the beam energy, the scan
size and which detector was on which ADC. The CSV is preferred when present
because it needs no parsing heuristics.

Pairing a log against its LMFs also calibrates the digitiser: the LMF pulse
count and the logged charge are proportional, and the slope is the pulse
size. Once measured for a session it applies to every run in that session,
which is what makes absolute quantification possible for runs whose log has
since been lost.
"""

import csv
import glob
import os
import re


class Run:
    __slots__ = ('number', 'caption', 'ion', 'MeV', 'scan', 'size',
                 'start', 'run_time', 'charge_nC', 'lmf_Mb', 'lmf_fmt', 'adc')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.adc is None:
            self.adc = {}

    @property
    def charge_uC(self):
        return None if self.charge_nC is None else self.charge_nC / 1000.0

    def __repr__(self):
        return '<run %s %r Q=%s nC>' % (self.number, (self.caption or '')[:28],
                                        self.charge_nC)


def read_csv(path):
    """The RUNLOG_00.CSV form. Returns {run number: Run}."""
    out = {}
    with open(path, newline='', encoding='latin-1') as fh:
        rows = list(csv.reader(fh))
    # the first line is 'Folder:,<path>'; the header is the first row whose
    # opening cell is exactly 'Run'
    hdr = None
    for i, r in enumerate(rows):
        if r and r[0].strip() == 'Run':
            hdr = [c.strip() for c in r]
            body = rows[i + 1:]
            break
    if hdr is None:
        return out

    def col(name):
        return hdr.index(name) if name in hdr else None

    ci = {k: col(k) for k in ('Run', 'Caption', 'Ion', 'MeV', 'Scan',
                              'Size (um)', 'StartTime', 'RunTime', 'Q (nC)',
                              'LMF Mb', 'LMF Fmt')}
    adc_cols = [(h, i) for i, h in enumerate(hdr)
                if re.match(r'ADC\d+ detector$', h)]
    for r in body:
        if not r or not r[0].strip().isdigit():
            continue

        def g(key, cast=str):
            i = ci.get(key)
            if i is None or i >= len(r) or r[i].strip() == '':
                return None
            try:
                return cast(r[i].strip())
            except ValueError:
                return None

        adc = {}
        for h, i in adc_cols:
            if i < len(r) and r[i].strip():
                adc[h.split()[0]] = r[i].strip()
        n = int(r[0])
        out[n] = Run(number=n, caption=g('Caption'), ion=g('Ion'),
                     MeV=g('MeV', float), scan=g('Scan'), size=g('Size (um)'),
                     start=g('StartTime'), run_time=g('RunTime'),
                     charge_nC=g('Q (nC)', float), lmf_Mb=g('LMF Mb', float),
                     lmf_fmt=g('LMF Fmt', int), adc=adc)
    return out


_RUN = re.compile(r'^Run:\s*(\d+)')
_Q = re.compile(r'^Q \(nC\):\s*([-\d.eE+]+)')
_BEAM = re.compile(r'^(\S+)\s+([\d.]+)\s+MeV')
_ADC = re.compile(r'^ADC(\d+) detector:\s*(\S+)')


def read_txt(path):
    """The RUNLOG.TXT form, for when the CSV is missing."""
    out = {}
    cur = None
    for raw in open(path, encoding='latin-1'):
        line = raw.strip()
        m = _RUN.match(line)
        if m:
            cur = Run(number=int(m.group(1)), adc={})
            out[cur.number] = cur
            expect_caption = True
            continue
        if cur is None:
            continue
        m = _Q.match(line)
        if m:
            cur.charge_nC = float(m.group(1))
            continue
        m = _BEAM.match(line)
        if m:
            cur.ion, cur.MeV = m.group(1), float(m.group(2))
            continue
        m = _ADC.match(line)
        if m:
            cur.adc['ADC%s' % m.group(1)] = m.group(2)
            continue
        if line.startswith('Start time:'):
            cur.start = line.split(':', 1)[1].strip()
        elif line.startswith('Run time:'):
            cur.run_time = line.split(':', 1)[1].strip()
        elif line.startswith('Map:'):
            cur.scan, cur.size = 'Map', line.split(':', 1)[1].strip()
        elif cur.caption is None and line and not line.startswith('='):
            cur.caption = line
    return out


def read(path):
    """Either form, chosen by extension."""
    return (read_csv if path.lower().endswith('.csv') else read_txt)(path)


def find_for(data_path):
    """The run log sitting beside a data file, CSV preferred. None if absent."""
    d = os.path.dirname(os.path.abspath(data_path))
    for pat in ('*RUNLOG*.CSV', '*RUNLOG*.csv', '*RUNLOG*.TXT', '*RUNLOG*.txt'):
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    return None


def run_number(path):
    """The run number from an LMF filename, or None."""
    m = re.search(r'(\d{4,})', os.path.basename(path))
    return int(m.group(1)) if m else None


def charge_for(data_path, log=None):
    """Integrated charge in uC for a data file, from its run log. None if the
    log is missing or the run is not in it - never a guess."""
    log_path = log or find_for(data_path)
    if not log_path or not os.path.exists(log_path):
        return None
    n = run_number(data_path)
    if n is None:
        return None
    r = read(log_path).get(n)
    return None if r is None else r.charge_uC


def calibrate_digitiser(lmf_paths, log_path, min_charge_nC=1.0):
    """Microcoulomb per LMF dose count, from runs that have both.

    The LMF counter and the logged charge should be strictly proportional -
    the counter is a pulse train from the current digitiser and each pulse is
    a fixed quantum of charge. Fitting a slope through the origin therefore
    measures the quantum, and the scatter about it says whether the two really
    are measuring the same thing.

    Returns a dict with the slope in uC per count, the fractional RMS scatter,
    the correlation, and the runs used. Runs with negligible charge are
    dropped because they carry no information and would dominate a ratio.
    """
    from . import lmf as _lmf

    runs = read(log_path)
    pts = []
    for p in lmf_paths:
        n = run_number(p)
        r = runs.get(n)
        if r is None or r.charge_nC is None or r.charge_nC < min_charge_nC:
            continue
        try:
            c = _lmf.clocks(p)
        except Exception:
            continue
        if c['charge_counts'] <= 0:
            continue
        pts.append((n, c['charge_counts'], r.charge_uC, c['duration_s']))
    if len(pts) < 3:
        return dict(uC_per_count=None, n=len(pts), points=pts,
                    reason='fewer than three usable runs')

    sx = sum(p[1] * p[1] for p in pts)
    sxy = sum(p[1] * p[2] for p in pts)
    slope = sxy / sx if sx else None

    resid = [(p[2] - slope * p[1]) / p[2] for p in pts if p[2]]
    rms = (sum(r * r for r in resid) / len(resid)) ** 0.5 if resid else None

    n = len(pts)
    mx = sum(p[1] for p in pts) / n
    my = sum(p[2] for p in pts) / n
    cov = sum((p[1] - mx) * (p[2] - my) for p in pts)
    vx = sum((p[1] - mx) ** 2 for p in pts)
    vy = sum((p[2] - my) ** 2 for p in pts)
    rho = cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None

    return dict(uC_per_count=slope, frac_rms=rms, correlation=rho,
                n=n, points=pts)
