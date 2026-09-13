"""Amptek DPPMCA .mca text spectra.

Blocks between <<TAG>> and <<END>> or <<TAG END>>. The header carries
LIVE_TIME, REAL_TIME and START_TIME; <<DP5 CONFIGURATION>> the DPP
settings (TPEA peaking time, GAIN, THSL/THFA thresholds, TECS) and
<<DPP STATUS>> the fast and slow counts and the TEC temperature. All of
that is returned because it is what decides whether a spectrum is usable:
on 12 Sep 2026 a fast threshold of 289 left the fast channel with 2,573
counts against 1.66 M slow, so pile-up rejection was off and no dead time
was recorded.
"""

import re

import numpy as np


def read_mca(path):
    """Return (counts, info). info has header, config and status dicts."""
    t = open(path, errors='ignore').read().replace('\r', '')
    m = re.search(r'<<DATA>>\n(.*?)\n<<END>>', t, re.S)
    if not m:
        raise ValueError('%s: no <<DATA>> block' % path)
    counts = np.array([int(float(v)) for v in m.group(1).split()])
    head = dict(re.findall(r'^([A-Z_]+) - (.*)$', t.split('<<ROI>>')[0], re.M))
    conf = {}
    m = re.search(r'<<DP5 CONFIGURATION>>\n(.*?)<<DP5 CONFIGURATION END>>', t, re.S)
    if m:
        for line in m.group(1).split('\n'):
            mm = re.match(r'([A-Z0-9]+)=([^;]*);\s*(.*)', line.strip())
            if mm:
                conf[mm.group(1)] = (mm.group(2).strip(), mm.group(3).strip())
    status = {}
    m = re.search(r'<<DPP STATUS>>\n(.*?)<<DPP STATUS END>>', t, re.S)
    if m:
        for line in m.group(1).split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                status[k.strip()] = v.strip()
    live = float(head.get('LIVE_TIME', 'nan'))
    real = float(head.get('REAL_TIME', 'nan'))
    return counts, dict(header=head, config=conf, status=status,
                        live_s=live, real_s=real, path=str(path))


def warnings(info):
    """Plain-language problems with the DPP settings, for the log."""
    out = []
    st, cf = info['status'], info['config']
    try:
        fast = float(st.get('Fast Count', 'nan').replace(',', ''))
        slow = float(st.get('Slow Count', 'nan').replace(',', ''))
        if fast < 0.5 * slow:
            out.append('fast channel counted %.0f against %.0f slow: fast '
                       'threshold (THFA=%s) too high, pile-up rejection and '
                       'dead time are not working'
                       % (fast, slow, cf.get('THFA', ('?',))[0]))
    except ValueError:
        pass
    try:
        tec = float(re.sub(r'[^0-9.]', '', st.get('TEC Temp', '')))
        tset = float(cf.get('TECS', ('nan',))[0])
        if tec > tset + 30:
            out.append('TEC reads %.0f K against a set point of %.0f K'
                       % (tec, tset))
    except ValueError:
        pass
    return out
