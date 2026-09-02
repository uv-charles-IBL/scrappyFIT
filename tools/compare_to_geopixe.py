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

from scrappyfit.io.gpfit import (area_scale,                   # noqa: E402
                                 read_fit_results)
from scrappyfit.io.gpdetector import (efficiency_curve,        # noqa: E402
                                      read_detector, read_filter)
from scrappyfit.io.gpspec import (calibration_from_pfr,        # noqa: E402
                                  read_spec, verify)
from scrappyfit.session import Session                          # noqa: E402

BASE = r'C:\Users\Charles\Desktop\GeoPIXE-main\test_data\fits\CSIRO'
SPEC = os.path.join(BASE, 'donut2x-2-whole.spec')
PFR = os.path.join(BASE, 'donut2x-2-whole-REF.pfr')


GEO_INSTALL = r"C:\Users\Charles\Desktop\GeoPIXE Install\GeoPIXE"


def _named_files(pfr_path, suffix):
    """Filenames with the given suffix mentioned anywhere in a .pfr.

    gpfit.py stops decoding before the detector and filter structs, so
    rec['detector'] is not populated. The names are still in the file as
    plain strings, and scanning for them is reliable in a way that guessing
    struct offsets is not. Missing this silently is expensive: without it
    the donut2x comparison runs with a SILICON crystal against a spectrum
    taken on germanium.
    """
    import re
    raw = open(pfr_path, 'rb').read()
    pat = re.compile((r'[ -~]{3,}' + re.escape(suffix)).encode(), re.I)
    return [m.group().decode('latin-1') for m in pat.finditer(raw)]


def _locate(name):
    for d in (os.path.join(GEO_INSTALL, 'Extra Detector Files'), GEO_INSTALL):
        p = os.path.join(d, os.path.basename(name))
        if os.path.exists(p):
            return p
    return None


def _find_detector(pfr_path):
    """The .detector the .pfr names, if it is on this machine."""
    for nm in _named_files(pfr_path, '.detector'):
        p = _locate(nm)
        if p:
            return read_detector(p)
    return None


def _find_filters(pfr_path):
    out = []
    for nm in _named_files(pfr_path, '.filter'):
        p = _locate(nm)
        if p:
            out += read_filter(p)
    return out


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

    # Use the detector GeoPIXE actually used. The .pfr names it, and it
    # matters: Canberra-34 is a GERMANIUM crystal behind 200 um of aluminium,
    # so escape peaks are Ge (9.886 keV) rather than Si, and nothing below
    # about 5 keV reaches the crystal at all.
    det = _find_detector(pfr_path)
    if det:
        s.options.crystal_Z = det['crystal']['Z'][0]
        s.options.crystal_thick_um = det['crystal']['thick'] / 5.32 * 10.0
        s.options.mac = 'mixed'           # Henke stops at 30 keV
        s._escape = None
        flt = _find_filters(pfr_path)
        print('  detector %s: crystal Z=%d, %.4g mg/cm2, %d absorber(s)'
              % (os.path.basename(det['path']), det['crystal']['Z'][0],
                 det['crystal']['thick'], len(det['absorbers'] or [])))
        if flt:
            print('  external filter(s): %s'
                  % ', '.join('%.4g mg/cm2 Z=%s'
                              % (f['thick'], f['Z'][0]) for f in flt))
        # The efficiency is not only for concentrations. It weights each
        # LINE of every element, and behind 200 um of aluminium it varies
        # 3.6x between Fe Ka and Fe Kb - without it no fit can satisfy both.
        Eax = cal[0] * np.arange(len(spec)) + cal[1]
        eff, _, _ = efficiency_curve(det, s.db, np.clip(Eax, 0.2, None),
                                     mac='mixed', filters=flt)
        eff = np.nan_to_num(eff)
        s.efficiency = lambda e: float(np.interp(e, Eax, eff))
        print('  efficiency: %.5f at Fe Ka, %.5f at Fe Kb (ratio %.2f)'
              % (s.efficiency(6.404), s.efficiency(7.058),
                 s.efficiency(7.058) / max(s.efficiency(6.404), 1e-12)))
    else:
        print('  WARNING: the detector named in the .pfr was not found, so '
              'this runs on the default silicon crystal')

    # GeoPIXE's fitted width parameters are already in OUR units:
    # FWHM in channels = sqrt(noise^2 + fano^2 (E - e_low)).
    par = rec.get('parameters') or {}
    if 'noise' in par and 'fano' in par:
        s.options.noise, s.options.fano = par['noise'], par['fano']
        print('  width from the .pfr: noise %.4f, Fano %.4f'
              % (par['noise'], par['fano']))
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

    scale = {}
    for i in range(rec['n_els']):
        if rec['mask'][i] == 0:
            continue
        nm = (rec['name'][i] or '').split()[0]
        sc = area_scale(rec, i, s.db)
        if sc:
            scale[nm] = sc

    print()
    print('FITTED PEAK AREAS, both on GeoPIXE per-line convention')
    print('%-6s %14s %14s %9s' % ('el', 'GeoPIXE', 'scrappyFIT', 'ratio'))
    print('-' * 48)
    rat = []
    for nm in sorted(gp_area, key=lambda k: -gp_area[k]):
        g = gp_area[nm]
        m = mine_area.get(nm)
        if m is None or g <= 0 or m <= 0:
            continue
        # A .pfr area is the named LINE, not the element total - see
        # gpfit.area_scale. Bring ours onto the same convention.
        br = scale.get(nm, 1.0)
        rat.append(m * br / g)
        print('%-6s %14.4g %14.4g %9.3f' % (nm, g, m * br, m * br / g))
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
