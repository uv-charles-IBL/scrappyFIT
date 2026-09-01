"""Batch processing: the same analysis across many runs.

The reason this exists is that a single fit is rarely the question. The
question is usually "how does this element behave across the twenty runs from
that session", and doing that by hand through a GUI both wastes time and
guarantees the settings drift between files.

A batch therefore fixes ONE set of options and applies it to every input, then
writes a single summary table alongside the per-run exports. If a run needs
different settings it does not belong in the same batch - that is the point.

Failures do not abort the run. A file that will not load or will not fit is
recorded with its error and the batch continues, because discovering at file
19 of 20 that the whole thing died at file 3 is the worst outcome.
"""

import csv
import pathlib
import traceback

from .session import FitOptions, Session


class BatchResult:
    """Outcome for one input file."""

    def __init__(self, path, ok, label='', chi2=None, areas=None,
                 concentrations=None, error='', exported=()):
        self.path = str(path)
        self.ok = ok
        self.label = label
        self.chi2 = chi2
        self.areas = areas or {}
        self.concentrations = concentrations or {}
        self.error = error
        self.exported = list(exported)

    def __repr__(self):
        return ('<%s %s%s>' % ('OK  ' if self.ok else 'FAIL',
                               pathlib.Path(self.path).name,
                               '' if self.ok else ': ' + self.error[:60]))


def run_batch(paths, elements, out_dir, options=None, calibration=None,
              efficiency=None, adc=0, quantify=False, mask_fn=None,
              progress=None):
    """Fit every file in `paths` with identical settings.

    elements     as for Session.run_fit
    out_dir      per-run exports plus batch_summary.csv land here
    calibration  (gain, offset) applied to every run, or None to trust each
                 file's own header. Forcing one calibration across a session
                 is usually right - they were acquired on one setup - and it
                 removes a per-file variable from the comparison.
    efficiency   path to a curve; required if quantify is True
    mask_fn      optional callable(session) -> (mask, name) or None, applied
                 before fitting. This is how a batch analyses one phase
                 across many samples rather than the bulk of each.
    progress     optional callable(index, total, path)

    Returns a list of BatchResult in input order.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = options or FitOptions()
    results = []

    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), str(p))
        s = Session(options=opts)
        try:
            s.load(str(p), adc=adc)
            if calibration:
                s.set_calibration(*calibration)
            if efficiency:
                s.load_efficiency(efficiency)
            name = ''
            if mask_fn is not None:
                got = mask_fn(s)
                if got is not None:
                    mask, name = got
                    s.set_mask(mask, name)
            r = s.run_fit(elements)
            conc = {}
            if quantify:
                try:
                    conc = {s.db.sym[Z]: v for Z, v in s.quantify().items()}
                except Exception as ex:
                    conc = {}
                    s_err = 'quantify skipped: %s' % ex
                    print(s_err)
            written = s.export(out)
            results.append(BatchResult(p, True, s.label, r.reduced_chi2,
                                       s.areas(), conc, '', written))
        except Exception as ex:
            results.append(BatchResult(p, False, error='%s: %s'
                                       % (type(ex).__name__, ex)))
            traceback.print_exc()

    _write_summary(out / 'batch_summary.csv', results, quantify)
    return results


def _write_summary(path, results, quantify):
    """One row per run, one column per element. Wide rather than long, because
    the normal next step is to open it in a spreadsheet and compare a column
    down the runs."""
    keys = set()
    for r in results:
        keys |= set(r.concentrations if quantify else r.areas)
    keys = sorted(keys)
    head = ['file', 'label', 'ok', 'chi2_reduced'] + keys + ['error']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in results:
            src = r.concentrations if quantify else r.areas
            w.writerow([pathlib.Path(r.path).name, r.label,
                        int(r.ok), '' if r.chi2 is None else '%.4f' % r.chi2]
                       + ['%.6g' % src[k] if k in src else '' for k in keys]
                       + [r.error])
    return path


def summarise(results):
    """Human-readable digest, for a log pane or a terminal."""
    ok = [r for r in results if r.ok]
    lines = ['%d of %d succeeded' % (len(ok), len(results))]
    for r in results:
        if r.ok:
            lines.append('  OK    %-28s chi2=%.3f  %d components'
                         % (pathlib.Path(r.path).name, r.chi2, len(r.areas)))
        else:
            lines.append('  FAIL  %-28s %s'
                         % (pathlib.Path(r.path).name, r.error))
    return '\n'.join(lines)
