"""Drive the GUI through a complete analysis, capturing the real window.

Quartz 287427 is the test case because it has exact ground truth - SiO2,
O 53.26 / Si 46.74 wt% - so every number can be checked rather than merely
looked at.
"""
import os
import sys

sys.path.insert(0, '.')
from PyQt5 import QtWidgets, QtCore

OUT = 'docs/drive'
os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication([])
from scrappyfit.gui.app import MainWindow

w = MainWindow()
w.resize(1560, 980)
w.show()
app.processEvents()

shots = []


def shot(name, widget=None):
    app.processEvents()
    tgt = widget or w
    p = os.path.join(OUT, '%d_%s.png' % (len(shots) + 1, name))
    tgt.grab().save(p)
    shots.append(p)
    return p


def log_tail(n=4):
    return '\n'.join('      ' + l
                     for l in w.log.toPlainText().splitlines()[-n:])


print('=' * 72)
print('STEP 1  load the quartz list-mode file')
w.session.load(r'C:\Users\Charles\Desktop\GeoPIXE-main\data\lmf\287427.lmf')
w.lbl_file.setText('287427.lmf')
w.refresh_spectrum()
print('   %s: %d channels, %.0f counts, %d events with positions'
      % (w.session.label, len(w.session.spectrum), w.session.spectrum.sum(),
         len(w.session.events[0])))
shot('loaded')

print('STEP 2  calibrate and check against known lines')
w.ed_gain.setText('0.0016984')
w.ed_off.setText('-0.37225')
w.on_cal_changed()
w.preset(['O', 'Si', 'Al', 'Na'], [], [])
w.on_check_cal()
print(log_tail(6))

print('STEP 3  detector efficiency')
for i in range(w.cmb_eff.count()):
    if 'deflector' in (w.cmb_eff.itemText(i) or ''):
        w.cmb_eff.setCurrentIndex(i)
        break
print(log_tail(1))

print('STEP 4  suggest elements from the spectrum')
w.on_suggest()
print(log_tail(8))
shot('suggested', w.elements)

print('STEP 5  choose elements deliberately (pruning the suggestions)')
w.preset(['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'K', 'Ca', 'Fe'],
         [], [])
print('   selected: %s' % w.elements.table.summary())
shot('elements', w.elements)

print('STEP 6  fit')
w.on_fit()
print(log_tail(1))
shot('fitted')

print('STEP 7  label every peak')
w.on_label_peaks()
print(log_tail(1))
shot('labelled')

print('STEP 8  sample model - type the known SiO2 matrix')
w.sample.rb_known.setChecked(True)
w.sample.tbl.setRowCount(0)
w.sample._add('O', '53.26')
w.sample._add('Si', '46.74')
w.sample.ed_thick.setText('')
w.sample.ed_beam.setText('1.0')
w.sample.ed_theta.setText('135.0')
print('   %s' % w._sample_summary())
shot('sample_model', w.sample)

print('STEP 9  quantify')
w.on_quantify()
rows = w.session.concentration_table()
print('   TRUE SiO2:  O 53.26   Si 46.74')
for nm, v, rel in rows[:6]:
    print('      %-3s %8.3f wt%%  +-%.1f%%' % (nm, v, rel))
shot('quantified')

print('STEP 10  map fluorine and mask the inclusions')
w.cmb_mapel.setCurrentText('F')
w.spin_bin.setValue(4)
w.chk_sub.setChecked(False)
w.on_map()
w.spin_thr.setValue(97)
w.on_threshold()
print(log_tail(2))
w.on_fit()
a = w.session.areas()
print('   masked region: chi2 %.2f   F %.0f  Al %.0f  Si %.0f'
      % (w.session.fit.reduced_chi2, a.get('F', 0), a.get('Al', 0),
         a.get('Si', 0)))
shot('masked_map')
w.tabs.setCurrentIndex(0)
w.refresh_spectrum()
shot('masked_spectrum')

print('STEP 11  back to the whole field, export everything')
w.on_clear_mask()
w.on_fit()
w.on_quantify()
outdir = os.path.abspath('docs/drive/export')
files = w.session.export(outdir)
from scrappyfit.io.gpda_write import from_session as dam
from scrappyfit.io.gpyield_write import from_session as yld
dam(w.session, os.path.join(outdir, 'quartz.dam'))
yld(w.session, os.path.join(outdir, 'quartz.yield'),
    matrix=w.sample.matrix(), thickness=w.sample.thickness(),
    beam_MeV=w.sample.beam(), theta_deg=w.sample.theta())
w.spec.save_image(os.path.join(OUT, 'final_spectrum.png'), dpi=150,
                  title='Quartz 287427 - scrappyFIT')
print('   %d files in %s' % (len(os.listdir(outdir)), outdir))
for f in sorted(os.listdir(outdir)):
    print('      ' + f)
shot('final')

print('=' * 72)
print('screenshots:')
for p in shots:
    print('   ' + p)
