"""Concentration cross-check against the worked example.

GeoPIXE's donut2x fit used kimberlite.yield, which ships with it. Feeding our
fitted areas through GEOPIXE'S OWN yields isolates the fitter: any difference
left is ours, not the yield model's.
"""
import os
import sys
sys.path.insert(0, r'C:\Users\Charles\Desktop\scrappyFIT')
import numpy as np
from scrappyfit.io.gpfit import area_scale, read_fit_results
from scrappyfit.io.gpspec import calibration_from_pfr, read_spec
from scrappyfit.io.gpyield import read_yield
from scrappyfit.session import Session

B = r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\fits\CSIRO'
G = r'C:\Users\Charles\Desktop\GeoPIXE Install\GeoPIXE'
pfr = os.path.join(B, 'donut2x-2-whole-REF.pfr')
rec = read_fit_results(pfr)[0]
cal = calibration_from_pfr(pfr)
spec, _ = read_spec(os.path.join(B, 'donut2x-2-whole.spec'))

s = Session()
s.load_spectrum(spec, cal=cal, label='donut2x')
s.options.e_low, s.options.e_high = rec['setup']['elow'], rec['setup']['ehigh']
s.options.mac = 'mixed'
s.load_detector(os.path.join(G, 'Extra Detector Files', 'Canberra-34.detector'))
s.options.noise = rec['parameters']['noise']
s.options.fano = rec['parameters']['fano']

pairs, seen, gp_area, gp_conc, sc = [], set(), {}, {}, {}
for i in range(rec['n_els']):
    if rec['mask'][i] == 0:
        continue
    nm = (rec['name'][i] or '').split()[0]
    gp_area[nm] = float(rec['area'][i])
    gp_conc[nm] = float(rec['conc'][i])
    v = area_scale(rec, i, s.db)
    if v:
        sc[nm] = v
    Z = s.db.z.get(nm.lower())
    sh = int(rec['shell'][i]) or 1
    if Z and (Z, sh) not in seen:
        seen.add((Z, sh))
        pairs.append((Z, sh))

r = s.run_fit(pairs)
mine = s.areas()

# GeoPIXE's own yields, keyed by element
Y = read_yield(os.path.join(B, 'kimberlite.yield'))
y0 = Y[0] if isinstance(Y, (list, tuple)) else Y
z2, sh2 = np.asarray(y0['z2']), np.asarray(y0['shell'])
row = np.asarray(y0['yield'])
row = row[0] if row.ndim == 2 else row
gy = {}
for i, (z, sv) in enumerate(zip(z2, sh2)):
    if sv == 1 and i < len(row) and row[i] > 0:
        gy[int(z)] = float(row[i])

print('Concentrations through GEOPIXE\'S OWN kimberlite.yield')
print('so only the fitted areas differ.')
print()
print('%-5s %13s %13s %8s   %8s' % ('el', 'GeoPIXE ppm', 'scrappyFIT',
                                    'ratio', 'area rat'))
print('-' * 58)

# one scale factor, set so Fe matches - the absolute constant (charge, solid
# angle) is not in these files, and it cancels for every RATIO below
ref = 'Fe'
Zref = s.db.z[ref.lower()]
k = None
rows = []
for nm in sorted(gp_conc, key=lambda x: -gp_conc[x]):
    Z = s.db.z.get(nm.lower())
    if Z is None or Z not in gy or nm not in mine or mine[nm] <= 0:
        continue
    conc_rel = mine[nm] / gy[Z]
    rows.append((nm, conc_rel))
d = dict(rows)
if ref in d and gp_conc.get(ref):
    k = gp_conc[ref] / d[ref]
rat = []
for nm, cr in rows:
    mc = cr * k
    g = gp_conc[nm]
    ar = (mine[nm] * sc.get(nm, 1.0) / gp_area[nm]) if gp_area.get(nm) else float('nan')
    if g > 0:
        rat.append(mc / g)
    print('%-5s %13.4f %13.4f %8.3f   %8.3f'
          % (nm, g, mc, mc / g if g > 0 else float('nan'), ar))
rat = np.array(rat)
print('-' * 58)
print('median %.3f   within 20%%: %.0f%%   within 2x: %.0f%%'
      % (np.median(rat), 100 * np.mean(np.abs(rat - 1) < 0.2),
         100 * np.mean((rat > 0.5) & (rat < 2))))
print()
print('(normalised on Fe, since charge and solid angle are not in these '
      'files; every other element is then a real test)')
