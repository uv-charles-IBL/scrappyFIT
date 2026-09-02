"""Benchmark my yields against GeoPIXE's own reference outputs."""
import sys, glob, os
sys.path.insert(0, r'C:\Users\Charles\Desktop\scrappyFIT')
import numpy as np
from scrappyfit.io.gpyield import read_yield
from scrappyfit.physics.gpdb import Database
from scrappyfit.physics.layers import LayeredYieldModel, Layer
from scrappyfit.physics.geometry import YIELD_TO_GEOPIXE

D = r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\yields'
db = Database()

files = sorted(glob.glob(os.path.join(D, '*-REF.yield')))
files += sorted(glob.glob(
    r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\fits\APS\*.yield'))
files += sorted(glob.glob(
    r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\fits\CSIRO\*.yield'))

for f in files:
    try:
        Y = read_yield(f)
    except Exception as ex:
        print('%-46s READ FAILED %s' % (os.path.basename(f)[:44], ex))
        continue
    y0 = Y[0] if isinstance(Y, (list, tuple)) else Y
    z1 = y0.get('z1')
    e0 = y0.get('e_beam')
    nl = y0.get('nl')
    print('=' * 78)
    print('%s' % os.path.basename(f))
    print('   beam Z1=%s  E=%s  theta=%s  layers=%s  formula=%s'
          % (z1, e0, y0.get('theta'), nl,
             (y0.get('formula') or [''])[:2]))
    if z1 != 1:
        print('   -> photon or heavy-ion source, my model is protons only: SKIP')
        continue
    L = y0['layers'][0]
    z2, sh = np.asarray(y0['z2']), np.asarray(y0['shell'])
    row = np.asarray(y0['yield'])
    row = row[0] if row.ndim == 2 else row
    zk = [int(z) for z in z2[sh == 1]]
    lym = LayeredYieldModel(db)
    lay = Layer(list(L['Z']), list(L['F']), 1e4, 'matrix')
    try:
        mine = lym.yields([lay], zk, E0=e0, theta_deg=y0.get('theta', 135.0),
                          mac='henke1993', fy='krause', n_steps=900)
    except Exception as ex:
        print('   model failed:', ex)
        continue
    rows = []
    for i, (z, s_) in enumerate(zip(z2, sh)):
        if s_ != 1:
            continue
        z = int(z)
        m = mine.get(z)
        if m is None or not np.isfinite(m) or m <= 0:
            continue
        if i >= len(row) or row[i] <= 0:
            continue
        pred = m * YIELD_TO_GEOPIXE
        rows.append((z, float(row[i]), pred, pred / float(row[i])))
    if not rows:
        print('   no comparable elements')
        continue
    rat = np.array([r[3] for r in rows])
    zz = np.array([r[0] for r in rows])
    print('   %d elements   mine/GeoPIXE: median %.4f  [%.3f - %.3f]'
          % (len(rows), np.median(rat), rat.min(), rat.max()))
    hv = rat[zz >= 15]
    if len(hv):
        print('   Z>=15 only  : median %.4f  spread %.1f%%'
              % (np.median(hv), 100 * (hv.max() - hv.min()) / np.median(hv)))
    print('   %-4s %-4s %12s %12s %8s' % ('Z', 'sym', 'GeoPIXE', 'mine', 'ratio'))
    for z, g, p, r in rows[:10]:
        print('   %-4d %-4s %12.5g %12.5g %8.3f' % (z, db.sym[z], g, p, r))
