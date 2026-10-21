"""Does the click order (detector before or after the range) change a fit?

Fits run 404004 (SiC, tests/data) with C N O Al Si over 0.17-6.0 keV, the
.detector loaded at the default E low 0.20 and the range set afterwards:

  old  noise kept as converted at 0.20 (before the FitOptions.e_low setter)
  new  noise re-referenced to 0.17 by the setter
  ref  range set first, then the detector

new and ref start from the same width (to the last bit or two). Then the
same 'ref' recipe is refitted from start noise values a few ulp and a few
tenths of a channel apart, to show how far the result moves on its own.
Usage: python tools/width_elow_order.py
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scrappyfit.session import Session  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / 'tests' / 'data'
DET = DATA / 'Amptek_25_C1_deflector.detector'
CAL = (0.0019756398864463256, -0.4522001604198639)
ELS = ['C', 'N', 'O', 'Al', 'Si']


def run(order, noise=None):
    s = Session()
    s.load(_paths.lab_spectrum('404004_sic.txt'))
    s.set_calibration(*CAL)
    s.load_efficiency([p for p in s.builtin_efficiencies()
                       if 'deflector' in p.lower()][0])
    if order == 'ref':
        s.options.e_low, s.options.e_high = 0.17, 6.0
        s.load_detector(str(DET))
    else:
        s.load_detector(str(DET))
        if order == 'old':
            s.options._e_low = 0.17         # bypass the setter: old behaviour
        else:
            s.options.e_low = 0.17
        s.options.e_high = 6.0
    if noise is not None:
        s.options.noise = noise
    noise0 = s.options.noise
    r = s.run_fit(ELS)
    return noise0, r


def show(label, n0, r):
    print('%-14s start noise %.17g ch  chi2r %.4f  noise %.4f fano %.4f  %s'
          % (label, n0, r.reduced_chi2, r.pars.a[0], r.pars.a[1],
             ' '.join('%s=%.0f' % (n, a) for n, a in zip(r.names, r.areas)
                      if n in ELS)))


if __name__ == '__main__':
    for order in ('old', 'new', 'ref'):
        show(order, *run(order))
    n = run('ref')[0]
    for k, x in enumerate((math.nextafter(n, 0), math.nextafter(n, 99),
                           n - 0.1, n + 0.1, n - 0.3, n + 0.3)):
        show('ref start %d' % k, *run('ref', noise=x))
