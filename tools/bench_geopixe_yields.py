"""Benchmark this package's yields against GeoPIXE's own -REF.yield output.

Run it against the test_data tree that ships with GeoPIXE:

    python tools/bench_geopixe_yields.py [path-to-GeoPIXE/test_data]

Three things this exists to catch, all of which it has caught:

  * a units error, which shows up as a ratio nowhere near one. This is how
    the 60x absolute-scale bug was found, and how it stays found.
  * a wrong line-grouping convention, which shows up as a STEP at Z = 28.
    GeoPIXE's .yield files store the MAJOR LINE - the unresolved Ka group
    below Z = 28, Ka1 alone above it, where the line table splits Ka1 from
    Ka2. Comparing element totals instead puts a 1.53x step right there and
    lifts the median from 1.02 to 1.48.
  * a layer model applied wrongly. Every layer has to be present, with the
    element assigned to the layer whose yield is being compared. Treating a
    2.6 mg/cm2 surface film as a thick target overstates its yield 3x.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scrappyfit.io.gpyield import read_yield                     # noqa: E402
from scrappyfit.physics.gpdb import Database                     # noqa: E402
from scrappyfit.physics.geometry import YIELD_TO_GEOPIXE         # noqa: E402
from scrappyfit.physics.layers import Layer, LayeredYieldModel   # noqa: E402

DEFAULT = r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data'


def stats(zs, rs):
    zs, rs = np.asarray(zs), np.asarray(rs)
    med = float(np.median(rs))
    slope = float(np.polyfit(zs, rs, 1)[0])
    return dict(n=len(rs), median=med,
                spread=100 * (rs.max() - rs.min()) / med,
                trend=1000 * slope / med)


def bench_one(path, db, mac='henke1993', fy='krause'):
    Y = read_yield(path)
    y0 = Y[0] if isinstance(Y, (list, tuple)) else Y
    name = os.path.basename(path)

    if y0.get('z1') != 1:
        print('%-44s SKIP  photon/heavy-ion source (z1=%s)'
              % (name[:44], y0.get('z1')))
        return

    z2 = np.asarray(y0['z2'])
    sh = np.asarray(y0['shell'])
    row = np.asarray(y0['yield'])
    if row.ndim == 1:
        row = row[None, :]
    zk = [int(z) for z in z2[sh == 1]]

    stack = [Layer(list(L['Z']), list(L['F']), float(L['thick']),
                   'layer%d' % i) for i, L in enumerate(y0['layers'])]
    lym = LayeredYieldModel(db)
    E0 = y0['e_beam']
    th = y0.get('theta', 135.0)

    print('%s   %.2f MeV, %d layer(s): %s'
          % (name, E0, len(stack),
             ', '.join('%s %.4g mg/cm2' % (f, L.thick)
                       for f, L in zip(y0.get('formula') or
                                       [''] * len(stack), stack))))

    for k in range(min(len(stack), row.shape[0])):
        # per_major_line=True: GeoPIXE stores the major line (see the module
        # docstring). in_layer pins every element into the layer whose stored
        # yields are being compared against.
        mine = lym.yields(stack, zk, E0=E0, theta_deg=th, mac=mac, fy=fy,
                          n_steps=900, per_major_line=True,
                          in_layer={z: k for z in zk})
        gp = row[k]
        zs, rs = [], []
        for i, (z, s_) in enumerate(zip(z2, sh)):
            if s_ != 1:
                continue
            m = mine.get(int(z))
            if m is None or not np.isfinite(m) or m <= 0:
                continue
            if i >= len(gp) or gp[i] <= 0:
                continue
            zs.append(int(z))
            rs.append(m * YIELD_TO_GEOPIXE / float(gp[i]))
        if len(rs) < 5:
            print('    layer %d: too few comparable elements' % k)
            continue
        st = stats(zs, rs)
        print('    layer %d  n=%2d  median %.4f  spread %6.1f%%  '
              'trend %+6.1f%%/10Z'
              % (k, st['n'], st['median'], st['spread'], st['trend']))


def main(root=None):
    root = root or DEFAULT
    if not os.path.isdir(root):
        print('no test_data at %s' % root)
        return 1
    db = Database()
    files = sorted(glob.glob(os.path.join(root, '**', '*.yield'),
                             recursive=True))
    if not files:
        print('no .yield files under %s' % root)
        return 1
    seen = set()
    for f in files:
        # -REF is the reference output; skip the non-REF twin of the same run
        base = os.path.basename(f).replace('-REF.yield', '.yield')
        if base in seen:
            continue
        seen.add(base)
        try:
            bench_one(f, db)
        except Exception as ex:
            print('%-44s FAILED %s' % (os.path.basename(f)[:44], ex))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
