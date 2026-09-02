"""Fit a GeoPIXE example spectrum here and put the two results side by side.

    python tools/compare_to_geopixe.py [.spec] [.pfr]

Defaults to the CSIRO donut2x example that ships with GeoPIXE.

What the comparison can and cannot show
---------------------------------------
It compares FITTED PEAK AREAS honestly, because both codes are fitting the
same counts in the same channels and nothing physical stands between the data
and the answer.

Concentrations are a weaker test. Turning an area into a concentration needs
the yield model, the detector efficiency and the filter, and GeoPIXE's
numbers here came from its own .yield file and its own detector definition.
A concentration difference therefore mixes a fitting difference with a
first-principles one, and the two cannot be separated from this file alone.
So areas are reported first and given more weight.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scrappyfit.io.gpfit import read_fit_results               # noqa: E402
from scrappyfit.io.gpspec import (calibration_from_pfr,        # noqa: E402
                                  read_spec, verify)
from scrappyfit.session import Session                          # noqa: E402

BASE = r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\fits\CSIRO'
SPEC = os.path.join(BASE, 'donut2x-2-whole.spec')
PFR = os.path.join(BASE, 'donut2x-2-whole-REF.pfr')


def main(spec_path=SPEC, pfr_path=PFR):
    if not (os.path.exists(spec_path) and os.path.exists(pfr_path)):
        print('example files not found:\n  %s\n  %s' % (spec_path, pfr_path))
        return 1

    rec = read_fit_results(pfr_path)[0]
    cal = calibration_from_pfr(pfr_path)
    spec, info = read_spec(spec_path)

    print('=' * 78)
    print('GeoPIXE reference : %s' % os.path.basename(pfr_path))
    print('  %d elements, chi %.4g, %d iterations, fit range %.3f - %.3f keV'
          % (rec['n_els'], rec['fit']['chi'], rec['fit']['n_its'],
             rec['setup']['elow'], rec['setup']['ehigh']))
    print('spectrum          : %s' % os.path.basename(spec_path))
    print('  %d channels, %.0f total counts' % (info['n_channels'],
                                                info['total']))
    if cal is None:
        print('  no calibration recovered from the .pfr; cannot proceed')
        return 1
    print('  calibration %.7f keV/ch, offset %+.6f keV' % cal)

    s = Session()
    s.load_spectrum(spec, cal=cal, label=os.path.basename(spec_path))
    rows, worst = verify(spec, cal, s.db)
    print('  calibration check against known lines, worst %.0f eV:' % worst)
    for ch, e, c, lab, d in rows[:6]:
        print('     ch %-5d %7.3f keV  %-9.0f %-8s %+5.0f eV'
              % (ch, e, c, lab, d))

    # Fit the same elements GeoPIXE fitted, over the same range.
    want = []
    for i in range(rec['n_els']):
        if rec['mask'][i] == 0:
            continue
        nm = (rec['name'][i] or '').split()[0]
        Z = s.db.z.get(nm.lower())
        if Z:
            want.append((Z, int(rec['shell'][i]) or 1))
    seen, els = set(), []
    for Z, sh in want:
        if (Z, sh) not in seen:
            seen.add((Z, sh))
            els.append((Z, sh))

    s.options.e_low = max(rec['setup']['elow'], cal[1] + cal[0])
    s.options.e_high = min(rec['setup']['ehigh'],
                           cal[0] * (len(spec) - 1) + cal[1])
    print()
    print('fitting %d elements over %.3f - %.3f keV'
          % (len(els), s.options.e_low, s.options.e_high))
    try:
        res = s.run_fit(els)
    except Exception as ex:
        print('FIT FAILED: %s' % ex)
        return 1
    print('  chi2 %.4g   (GeoPIXE reported chi %.4g)'
          % (res.reduced_chi2, rec['fit']['chi']))

    mine_area = s.areas()
    gp_area = {}
    for i in range(rec['n_els']):
        if rec['mask'][i] == 0:
            continue
        nm = (rec['name'][i] or '').split()[0]
        gp_area[nm] = float(rec['area'][i])

    print()
    print('FITTED PEAK AREAS - the like-for-like comparison')
    print('%-6s %14s %14s %9s' % ('el', 'GeoPIXE', 'scrappyFIT', 'ratio'))
    print('-' * 48)
    rat = []
    for nm in sorted(gp_area, key=lambda k: -gp_area[k]):
        g = gp_area[nm]
        m = mine_area.get(nm)
        if m is None or g <= 0:
            continue
        rat.append(m / g)
        print('%-6s %14.4g %14.4g %9.3f' % (nm, g, m, m / g))
    if rat:
        rat = np.array(rat)
        print('-' * 48)
        print('median ratio %.4f, %d elements, %.0f%% within a factor 2'
              % (np.median(rat), len(rat),
                 100.0 * np.mean((rat > 0.5) & (rat < 2.0))))

    gp_conc = {}
    for i in range(rec['n_els']):
        if rec['mask'][i] == 0:
            continue
        nm = (rec['name'][i] or '').split()[0]
        gp_conc[nm] = float(rec['conc'][i])
    print()
    print('GeoPIXE CONCENTRATIONS, ppm (its yield model, for reference)')
    print('%-6s %14s %12s %12s' % ('el', 'conc ppm', 'error', 'mdl'))
    for i in sorted(range(rec['n_els']),
                    key=lambda j: -float(rec['conc'][j]))[:12]:
        if rec['mask'][i] == 0:
            continue
        print('%-6s %14.4f %12.4f %12.4f'
              % ((rec['name'][i] or '?').split()[0], rec['conc'][i],
                 rec['error'][i], rec['mdl'][i]))
    return 0


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else SPEC
    b = sys.argv[2] if len(sys.argv) > 2 else PFR
    sys.exit(main(a, b))
