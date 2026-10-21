"""Write tests/data/synthetic_quartz.txt: a quartz-like test spectrum made by
scrappyFIT's own model, with no laboratory data in it.

usage: python -I tools/make_synthetic_spectrum.py [output path]

The lab's reference spectra stay off the public repository (they are not
ours to publish). The fit and GUI tests still need something real-shaped to
run on CI and on a fresh clone, and this is it. The spectrum is the sum of:

  * the K-line components Session.build_components makes for C, N, O, F,
    Na, Mg, Al and Si, drawn with FitOptions' default starting width and
    tail at the quartz golden calibration (0.0016984 keV/ch, -0.37225 keV),
    with hand-picked round areas (AREAS below; not fitted to any data);
  * a smooth falling continuum, 0 below 0.15 keV;
  * Poisson noise from a fixed seed.

Its counts depend on the database and line table, so rerun this if those
change. The tests only assume the format (two columns, channel and counts,
4096 channels) and that the elements' peaks are there.

Written 2026-10-20 (fix/repo-portability).
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from scrappyfit.fitting import fit as _fit          # noqa: E402
from scrappyfit.session import Session              # noqa: E402

N_CHANNELS = 4096
CAL = (0.0016984, -0.37225)      # keV/ch, keV: tests/test_golden_spectra.py
SEED = 20261020
AREAS = {                        # counts per element, chosen by hand
    'C': 20000, 'N': 3000, 'O': 300000, 'F': 2000,
    'Na': 15000, 'Mg': 8000, 'Al': 20000, 'Si': 400000,
}


def continuum(E):
    """Counts per channel: a falling bremsstrahlung-like shape."""
    b = 400.0 * np.exp(-E / 1.2) + 15.0 * np.exp(-E / 6.0)
    return np.where(E > 0.15, b, 0.0)


def make():
    s = Session()
    s.load_spectrum(np.zeros(N_CHANNELS), cal=CAL, label='synthetic quartz')
    o = s.options
    a, b = s.cal
    eoc = 0.7 * o.e_low + 0.3 * o.e_high
    pars = _fit.ShapePars(o.noise, o.fano, (eoc - b) / a, 1.0 / a,
                          tail_amp=o.tail_amp, tail_len=o.tail_len,
                          e_low=o.e_low, e_high=o.e_high)
    model = continuum(a * np.arange(N_CHANNELS) + b)
    for c in s.build_components(list(AREAS)):
        sym = c.name.split()[0]
        prof = c.profile(pars, N_CHANNELS)
        if prof.sum() > 0:
            model = model + AREAS[sym] * prof / prof.sum()
    rng = np.random.default_rng(SEED)
    return rng.poisson(np.maximum(model, 0.0))


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(
        ROOT, 'tests', 'data', 'synthetic_quartz.txt')
    counts = make()
    with open(out, 'w', newline='\n') as f:
        for ch, c in enumerate(counts):
            f.write('%d %d\n' % (ch, c))
    print('wrote %s: %d channels, %d counts' % (out, len(counts),
                                                 int(counts.sum())))


if __name__ == '__main__':
    main(sys.argv)
