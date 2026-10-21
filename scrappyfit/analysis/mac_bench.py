"""Benchmark MAC databases against samples of known composition.

The method
----------
A MAC database cannot be judged in the abstract; it has to be judged by
whether it makes a fit of a KNOWN material come out right. Two samples here
carry independent ground truth:

  quartz 287427    SiO2 exactly - O 53.26 / Si 46.74 wt%. A two-element test
                   with no assumptions at all.
  chondrite 287428 mean H-chondrite composition. Looser, but it spans Mg to
                   Fe so it tests a wide energy range at once.

For each database the same spectrum is fitted with the same lines, converted
to concentrations through the same yield model, and normalised to 100 wt%.
Only the mass attenuation coefficients differ. The score is the mean of
|ln(measured / true)| over the elements with known values - a log measure so
that being 2x high and 2x low count equally.

Quartz is the cleaner discriminator: oxygen at 525 eV and silicon at 1740 eV
sit on opposite sides of the 1 keV floor below which XCOM and Scofield
tabulate nothing, so the O/Si ratio is directly sensitive to the choice.
"""
import sys, numpy as np
from ..fitting import fit as F
from ..physics.gpdb import Database
from ..fitting.background import snip
from ..physics.layers import Layer, LayeredYieldModel
from ..physics import lfix
ROOT = None      # resolved by scrappyfit.config
DATASETS = ['henke1993', 'sabbatuccisalvat2016', 'ffast', 'mixed']

def load_efficiency(path):
    """Detector efficiency curve from a GeoPIXE EFF3 export."""
    E, f = [], []
    for l in open(path, errors='ignore'):
        if l.startswith('EFF3'):
            tk = l.split()
            E.append(float(tk[1])); f.append(float(tk[2]))
    E, f = np.array(E), np.array(f)
    return lambda e: float(np.interp(np.clip(e, E[0], E[-1]), E, f))


eff = None      # set by the caller, via load_efficiency()


def fit_areas(db, spec, A, B, zlist, extra=(), elo=0.20, ehi=6.50):
    bk = snip(spec, A, B, elo, ehi, passes=3, use_low_stats=True)
    comps = []
    for Z in zlist:
        L = db.line_list(Z, 1)
        if not L:
            continue
        c = F.Component(db.sym[Z], L)
        c.tail_amp_fn = lambda E: 0.08
        c.tail_len_fn = lambda E: 1.0
        comps.append(c)
    for Z, sh, nm in extra:
        L = db.line_list(Z, sh)
        if not L:
            continue
        c = F.Component(nm, L)
        c.tail_amp_fn = lambda E: 0.70 if sh == 2 else 0.08
        c.tail_len_fn = lambda E: 3.5 if sh == 2 else 1.0
        comps.append(c)
    r = F.fit_spectrum(spec, A, B, comps, elo, ehi, noise=32.74, fano=28.05,
                       tail_amp=0.08, tail_len=1.0, background=bk,
                       refine=('cal', 'width', 'tail', 'shelf', 'contact'),
                       nonneg='strict')
    return dict(zip(r.names, r.areas)), r.reduced_chi2


def concentrations(db, areas, zlist, matrix_z, matrix_w, thick, mac, fy='krause'):
    lym = LayeredYieldModel(db)
    lay = Layer(matrix_z, matrix_w, max(thick, 1.0), 'm')
    yy = lym.yields([lay], zlist, E0=1.0, theta_deg=135.0, mac=mac, fy=fy,
                    n_steps=900)
    out = {}
    for Z in zlist:
        nm = db.sym[Z]
        if nm not in areas or areas[nm] <= 0:
            continue
        if not np.isfinite(yy[Z]) or yy[Z] <= 0:
            continue
        b = max(i for _, i in db.line_list(Z, 1))
        out[Z] = areas[nm] * b / (yy[Z] * eff(db.line_energy(Z)))
    t = sum(out.values())
    return {Z: 100.0 * v / t for Z, v in out.items()} if t > 0 else {}


def score(meas, truth):
    d = [abs(np.log(meas[Z] / truth[Z])) for Z in truth
         if Z in meas and meas[Z] > 0]
    return (float(np.mean(d)), len(d)) if d else (float('nan'), 0)
