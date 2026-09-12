"""Ga2O3 device runs: scrappyFIT fitted areas against GeoPIXE's window maps.

GeoPIXE left window maps (.2D) for these runs rather than fits. A window map
is the count in a fixed channel range, so it carries whatever else falls in
that range - background, tails of neighbours, escape peaks - and it is what
GeoPIXE's operator was looking at. The fitted area is the deconvolved line.
Their ratio per run says how much the window was over- or under-reading.

The .2D decoder is verified pixel-identical against the LMF on all 39 maps,
so 'GeoPIXE window' below is exactly the number GeoPIXE displayed.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scrappyfit.io import lmf, omdaq2d, runlog      # noqa: E402
from scrappyfit.session import Session              # noqa: E402

D = r'C:\Users\Charles\Desktop\GeoPIXE-main\data\Ga2O3\nigels sample analysis'
G = r'C:\Users\Charles\Desktop\GeoPIXE Install\GeoPIXE'
CAL = (0.0091636, -2.07536)
DET = os.path.join(G, 'Amptek_25_C1_deflector.detector')
# As is in the Ga-rich runs (380006, 380008): As Ka at 10.54 keV, 2700 and
# 4200 counts. Without it the fit broadened every peak to cover it.
ELS = [(31, 1), (31, 2), (79, 2), (79, 3), (22, 1), (13, 1), (14, 1), (29, 1),
       (33, 1), (33, 2)]
SHOW = ['Ti', 'Cu', 'Ga', 'AuL', 'AuM', 'As']
WIN = {'Ti': 'TiKa', 'Cu': 'CuKa', 'Ga': 'GaKa', 'AuL': 'AuLa'}


def caption(run):
    for log in glob.glob(os.path.join(D, '**', '*RUNLOG_00.CSV'), recursive=True):
        r = runlog.read(log).get(int(run))
        if r:
            return r.caption or '', r.charge_uC
    return '', None


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(D, '**', '*.lmf'), recursive=True)):
        run = os.path.basename(p)[:6]
        cap, q_log = caption(run)
        c = lmf.clocks(p)
        q = c['charge_counts'] * 1e-4
        # three runs have an overflowed log charge; the dose counter is used
        if q_log and q_log < 1e4:
            q = q_log
        s = Session()
        s.load(p)
        s.set_calibration(*CAL)
        s.load_efficiency([e for e in s.builtin_efficiencies()
                           if 'deflector' in e][0])
        s.load_detector(DET)
        s.options.e_low, s.options.e_high = 1.0, 12.0
        try:
            r = s.run_fit(ELS)
        except Exception as ex:
            print('%s FIT FAILED %s' % (run, ex))
            continue
        ar = s.areas()
        win = {}
        for k, tag in WIN.items():
            f = glob.glob(os.path.join(D, '**', '%sP0%s.2D' % (run, tag)),
                          recursive=True)
            if f:
                img, _ = omdaq2d.decode(f[0])
                win[k] = int(img.sum())
        rows.append((run, cap, q, r.reduced_chi2, ar, win))
        print('%s  chi2 %6.2f  Q %.3f uC  %s' % (run, r.reduced_chi2, q, cap[:50]))
        for E, ex, sg, cands in s.unexplained_peaks(2):
            if sg >= 8.0:
                print('        unexplained %.2f keV %6.0f counts %5.1f sigma  %s'
                      % (E, ex, sg, ', '.join(cands)))

    print()
    print('FITTED AREA per uC, and the ratio to GeoPIXE window map where one exists')
    print('%-7s %8s | %9s %6s | %9s %6s | %9s %6s | %9s %6s | %9s | %9s'
          % ('run', 'chi2', 'Ti', 'w/f', 'Cu', 'w/f', 'Ga', 'w/f', 'AuL', 'w/f',
             'AuM', 'As'))
    print('-' * 100)
    for run, cap, q, chi, ar, win in rows:
        cells = ['%-7s %8.2f' % (run, chi)]
        for k in SHOW:
            a = ar.get(k, 0.0) / q if q else float('nan')
            w = win.get(k)
            rat = (w / ar[k]) if (w and ar.get(k, 0) > 0) else None
            if k in ('AuM', 'As'):
                cells.append('%9.0f' % a)
            else:
                cells.append('%9.0f %6s' % (a, ('%.2f' % rat) if rat else '-'))
        print(' | '.join(cells))
    print()
    print('w/f = window counts / fitted area. Above 1 the window over-reads')
    print('(background, neighbours, escape peaks); below 1 it under-reads')
    print('(the window is narrower than the line).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
