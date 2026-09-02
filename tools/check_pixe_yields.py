"""How well do our PIXE yields agree with GeoPIXE, above the light elements?

Only proton yield files count here (z1 == 1). Element totals both sides, since
a .yield stores element totals.
"""
import glob
import os
import sys
sys.path.insert(0, r'C:\Users\Charles\Desktop\scrappyFIT')
import numpy as np
from scrappyfit.io.gpyield import read_yield
from scrappyfit.physics.gpdb import Database
from scrappyfit.physics.geometry import YIELD_TO_GEOPIXE
from scrappyfit.physics.layers import Layer, LayeredYieldModel

db = Database()
ROOTS = [r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data',
         r'F:\GeoPIXE']
files = []
for r in ROOTS:
    files += glob.glob(os.path.join(r, '**', '*.yield'), recursive=True)

seen, per_z = set(), {}
print('%-46s %6s %8s %8s %8s' % ('file', 'MeV', 'Z>=20', 'Z>=26', 'spread'))
print('-' * 82)
for f in sorted(files):
    base = os.path.basename(f).replace('-REF.yield', '.yield')
    if base in seen:
        continue
    seen.add(base)
    try:
        Y = read_yield(f)
    except Exception:
        continue
    y0 = Y[0] if isinstance(Y, (list, tuple)) else Y
    if y0.get('z1') != 1 or not y0.get('layers'):
        continue
    z2, sh = np.asarray(y0['z2']), np.asarray(y0['shell'])
    row = np.asarray(y0['yield'])
    row = row[0] if row.ndim == 2 else row
    zk = [int(z) for z in z2[sh == 1]]
    stack = [Layer(list(L['Z']), list(L['F']), float(L['thick']), 'L%d' % i)
             for i, L in enumerate(y0['layers'])]
    lym = LayeredYieldModel(db)
    try:
        mine = lym.yields(stack, zk, E0=y0['e_beam'],
                          theta_deg=y0.get('theta', 135.0), mac='mixed',
                          fy='krause', n_steps=900, per_major_line=False,
                          in_layer={z: 0 for z in zk})
    except Exception:
        continue
    rr = {}
    for i, (z, s_) in enumerate(zip(z2, sh)):
        if s_ != 1:
            continue
        z = int(z)
        m = mine.get(z)
        if m is None or not np.isfinite(m) or m <= 0:
            continue
        if i >= len(row) or row[i] <= 0:
            continue
        rr[z] = m * YIELD_TO_GEOPIXE / float(row[i])
        per_z.setdefault(z, []).append(rr[z])
    if not rr:
        continue
    a20 = [v for z, v in rr.items() if z >= 20]
    a26 = [v for z, v in rr.items() if z >= 26]
    if not a20:
        continue
    print('%-46s %6.2f %8.3f %8.3f %7.1f%%'
          % (base[:46], y0['e_beam'], np.median(a20),
             np.median(a26) if a26 else float('nan'),
             100 * (max(a20) - min(a20)) / np.median(a20)))

print()
print('Pooled across every proton yield file, by element:')
print('%-5s %-4s %8s %8s %s' % ('el', 'Z', 'median', 'n', 'range'))
allz = sorted(per_z)
for z in allz:
    if z < 18 or z % 2:
        continue
    v = np.array(per_z[z])
    print('%-5s %-4d %8.3f %8d   %.3f - %.3f'
          % (db.sym[z], z, np.median(v), len(v), v.min(), v.max()))

pool20 = np.array([x for z in allz if z >= 20 for x in per_z[z]])
pool26 = np.array([x for z in allz if z >= 26 for x in per_z[z]])
print()
print('Z >= 20 : median %.4f, %d comparisons, %.0f%% within 20%% of GeoPIXE'
      % (np.median(pool20), len(pool20),
         100 * np.mean(np.abs(pool20 - 1) < 0.2)))
print('Z >= 26 : median %.4f, %d comparisons, %.0f%% within 20%% of GeoPIXE'
      % (np.median(pool26), len(pool26),
         100 * np.mean(np.abs(pool26 - 1) < 0.2)))
