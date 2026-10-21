"""Run the J3 GUI steps on one checkout and dump the fit, to compare versions.

usage: python -I tools/gui_fit_dump.py <repo root> <out.json> [nopreview]

Run it against two checkouts (e.g. a git worktree of the old commit) and
compare the JSON. Used 2026-10-18 for the E-low CHANGELOG entry.
"""
import json
import os
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
sys.path.insert(0, root)
from PyQt5 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication([])
from scrappyfit.gui.app import MainWindow  # noqa: E402

DATA = os.path.join(root, 'tests', 'data')
win = MainWindow()
if len(sys.argv) > 3 and hasattr(win, 'chk_preview'):
    win.chk_preview.setChecked(False)


def settle():
    while True:
        app.processEvents()
        j = getattr(win, 'jobs', None)
        if j is None or not (j.busy or win.preview_pending):
            break
        time.sleep(0.01)
    app.processEvents()


win.open_path(_paths.lab_spectrum('404004_sic.txt', root))
win.ed_gain.setText('%.10g' % 0.0019756398864463256)
win.ed_off.setText('%.10g' % -0.4522001604198639)
win.ed_gain.editingFinished.emit()
win.ed_elo.setText('0.17')
win.ed_ehi.setText('6')
win.ed_ehi.editingFinished.emit()
settle()
QtWidgets.QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (os.path.join(DATA, 'Amptek_25_C1_deflector.detector'), ''))
win.on_load_detector()
settle()
idx = [i for i in range(win.cmb_eff.count())
       if 'deflector' in win.cmb_eff.itemText(i).lower()][0]
win.cmb_eff.setCurrentIndex(idx)
settle()
db = win.session.db
for sym in ['C', 'N', 'O', 'Al', 'Si']:
    win.elements.table.buttons[sym].click()
settle()
s = win.session
state = {'noise': s.options.noise, 'fano': s.options.fano,
         'start_pars': None if s.start_pars is None else list(s.start_pars.a),
         'range': [s.options.e_low, s.options.e_high], 'cal': list(s.cal)}
t0 = time.perf_counter()
win.on_fit()
settle()
ms = 1e3 * (time.perf_counter() - t0)
r = s.fit
json.dump({'ms': ms, 'state_before_fit': state, 'chi2': repr(float(r.chi2)),
           'names': list(r.names), 'areas': [repr(float(a)) for a in r.areas],
           'pars': [repr(float(x)) for x in r.pars.a]}, open(out, 'w'),
          indent=0)
print(out, '%.0f ms' % ms, r.reduced_chi2)
win.close()
