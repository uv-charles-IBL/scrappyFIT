"""Fit the spectrum in Nigel's GeoPIXE export and overlay GeoPIXE's own fit."""
import sys
sys.path.insert(0, r'C:\Users\Charles\Desktop\scrappyFIT')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scrappyfit.session import Session

CSV = (r'C:\Users\Charles\Desktop\GeoPIXE-main\data\Ga2O3'
       r'\nigels sample analysis\csv data\sample 1 300 um bar PIXE plot.csv')
d = np.genfromtxt(CSV, delimiter=',', skip_header=1)
E, y, snip, gfit, gpu, gesc = (d[:, k] for k in range(6))
a = float(np.median(np.diff(E)))
b = float(E[0])
print('run 354016 export: %d ch, %.6f keV/ch, offset %.4f, %.0f counts, '
      'E %.2f..%.2f' % (len(E), a, b, y.sum(), E[0], E[-1]))
print('GeoPIXE fit columns: fit %.0f, pileup %.0f, escapes %.0f, SNIP %.0f'
      % (gfit.sum(), gpu.sum(), gesc.sum(), snip.sum()))

s = Session()
s.load_spectrum(y, cal=(a, b), label='354016')
db = s.db
pk, _ = find_peaks(y, height=y.max() * 0.005, distance=6)
print('\nstrongest peaks:')
cands = {}
for i in sorted(pk, key=lambda j: -y[j])[:14]:
    e = E[i]
    best = None
    for Z in range(6, 93):
        for sh, lab in ((1, ''), (2, 'L'), (3, 'M')):
            le = db.line_energy(Z, sh)
            if le > 0 and abs(le - e) < 0.045:
                if best is None or abs(le - e) < abs(best[1] - e):
                    best = (db.sym[Z] + lab, le)
    print('   %7.3f keV %8.0f  %s' % (e, y[i], best[0] if best else '?'))

# the element set GeoPIXE's windows used, plus what the peaks show
els = ['O', 'Al', 'Si', 'Ti', 'Cu', 'Ga', 'Au']
s.options.e_low = 0.9
s.estimate_resolution()
s.options.e_high = min(E[-1], 14.0)
s.load_efficiency([p for p in s.builtin_efficiencies() if 'deflector' in p][0])
try:
    r = s.run_fit([(db.z['al'], 1), (db.z['si'], 1),
                   (db.z['ti'], 1), (db.z['cu'], 1), (db.z['ga'], 1),
                   (db.z['k'], 1), (db.z['au'], 2), (db.z['au'], 3)])
except Exception as ex:
    print('FIT FAILED:', ex)
    sys.exit(1)
m = np.asarray(r.model, float)
print('\nscrappyFIT chi2 %.3f over %.2f-%.2f keV' % (r.reduced_chi2,
                                                     s.options.e_low,
                                                     s.options.e_high))
win = (E >= s.options.e_low) & (E <= s.options.e_high)
gchi = np.sum((y[win] - gfit[win]) ** 2 / np.maximum(gfit[win], 1)) / win.sum()
print('GeoPIXE fit column, same chi2 definition: %.3f' % gchi)
print('\nareas:', {k: round(v) for k, v in s.areas().items() if v > 0})

fig, (ax, axr) = plt.subplots(2, 1, figsize=(12.5, 7.5), dpi=130,
                              gridspec_kw=dict(height_ratios=[3, 1]),
                              sharex=True)
q = win
ax.semilogy(E[q], np.maximum(y[q], 0.4), color='#222', lw=0.9, label='data')
ax.semilogy(E[q], np.maximum(gfit[q], 0.4), color='#1F4FD8', lw=1.3,
            label='GeoPIXE fit (chi2 %.2f)' % gchi)
ax.semilogy(E[q], np.maximum(m[q], 0.4), color='#D62828', lw=1.3,
            label='scrappyFIT (chi2 %.2f)' % r.reduced_chi2)
ax.set_ylabel('counts')
ax.set_title('run 354016 (Nigel, Ga2O3 device): the same spectrum fitted '
             'by both', fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.2)
sg = np.sqrt(np.maximum(y, 1))
axr.plot(E[q], ((y - gfit) / sg)[q], color='#1F4FD8', lw=0.7, label='GeoPIXE')
axr.plot(E[q], ((y - m) / sg)[q], color='#D62828', lw=0.7, label='scrappyFIT')
axr.axhline(0, color='k', lw=0.6)
axr.set_ylim(-12, 12)
axr.set_xlabel('energy, keV'); axr.set_ylabel('residual, sigma')
axr.legend(fontsize=8, frameon=False); axr.grid(alpha=0.2)
fig.tight_layout()
out = r'C:\Users\Charles\Desktop\scrappyFIT\docs\nigel_354016_fit.png'
fig.savefig(out, facecolor='white')
print('figure', out)
