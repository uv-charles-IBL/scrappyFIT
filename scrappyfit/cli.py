"""Command line, no window: fit one or more spectra with one set of settings.

    scrappyfit-cli SPECTRUM [SPECTRUM ...] --elements C,N,O,Si --out DIR
                   [--cal gain,offset] [--range lo,hi] [--efficiency PATH]
                   [--quantify]

A thin front end to batch.run_batch: every file gets the same options, the
per-run exports and batch_summary.csv land in DIR, and the digest is printed.
Element tokens are symbols for the K lines; "FeL" or "Fe:L" means the L
lines and "PbM" the M lines, as in the GUI's --elements.

The window is `scrappyfit` (or `python -m scrappyfit`).
"""
import argparse
import sys


def _pairs(spec, db):
    out = []
    for tok in spec.split(','):
        tok = tok.strip().replace(':', '')
        if not tok:
            continue
        sh = 1
        if len(tok) > 1 and tok[-1] in 'LM' and tok[:-1].lower() in db.z:
            sh = 2 if tok[-1] == 'L' else 3
            tok = tok[:-1]
        Z = db.z.get(tok.lower())
        if Z is None:
            raise SystemExit('unknown element %r' % tok)
        out.append((Z, sh))
    return out


def _two(text, what):
    try:
        a, b = (float(x) for x in text.split(','))
    except ValueError:
        raise SystemExit('--%s takes two numbers, a,b' % what)
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='scrappyfit-cli',
        description='Fit PIXE spectra without the window (batch.run_batch).')
    ap.add_argument('paths', nargs='+', help='.lmf, .spec, .mca or .txt files')
    ap.add_argument('--elements', required=True,
                    help='comma-separated, e.g. C,N,O,Si or Fe,FeL,PbM')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--cal', help='gain,offset (keV/channel, keV) for all files')
    ap.add_argument('--range', help='fit range lo,hi in keV')
    ap.add_argument('--efficiency', help='efficiency curve file')
    ap.add_argument('--quantify', action='store_true',
                    help='quantify as well (needs --efficiency)')
    ns = ap.parse_args(argv)
    if ns.quantify and not ns.efficiency:
        ap.error('--quantify needs --efficiency')

    from . import batch
    from .session import FitOptions, Session

    opts = FitOptions()
    if ns.range:
        opts.e_low, opts.e_high = _two(ns.range, 'range')
    elements = _pairs(ns.elements, Session().db)
    res = batch.run_batch(
        ns.paths, elements, ns.out, options=opts,
        calibration=_two(ns.cal, 'cal') if ns.cal else None,
        efficiency=ns.efficiency, quantify=ns.quantify,
        progress=lambda i, n, p: print('[%d/%d] %s' % (i + 1, n, p)))
    print(batch.summarise(res))
    return 0 if res and all(r.ok for r in res) else 1


if __name__ == '__main__':
    sys.exit(main())
