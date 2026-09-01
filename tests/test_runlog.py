"""OMDAQ run log parsing, and the digitiser calibration it makes possible."""

import glob
import os

import pytest

from scrappyfit.io import runlog

# the 15-Dec-2023 session is the only one here with a log AND its LMFs
SESSION = r'C:\Users\Charles\Desktop\GeoPIXE Install\GeoPIXE\20231215'
have_session = os.path.isdir(SESSION)
needs_session = pytest.mark.skipif(not have_session,
                                   reason='15-Dec-2023 session not present')

CSV_TEXT = '''Folder:,D:\\Microbeam Data\\NULL\\20231215\\
Run,Caption,Ion,MeV,Scan,"Size (um)",StartTime,RunTime,"Q (nC)","LMF Mb","LMF Fmt","ADC0 detector","ADC0 signal","ADC1 detector","ADC1 signal"
258304,"PFOS sample 1",1H1,1.000,Map,"500 um","15-Dec-2023 11:58","12:00:21 AM",0.000,0.0,1,Amptek,PIXE,RBS,RBS
258305,"PFOS sample 1",1H1,1.000,Map,"500 um","15-Dec-2023 12:00","12:01:06 AM",102.300,0.0,1,Amptek,PIXE,RBS,RBS
'''

TXT_TEXT = '''D:\\Microbeam Data\\NULL\\20231215\\_15-Dec-2023_NULL_RUNLOG
=====================================================
Run: 258305
PFOS sample 1 2 uc broad beam
1H1 1.000 MeV
Map: 500 um
Start time: 15-Dec-2023 12:00
Run time: 12:01:06 AM
Q (nC): 102.300
LMF Mb: 0.0 Fmt = 1
ADC0 detector: Amptek (PIXE)
ADC1 detector: RBS (RBS)
'''


def test_csv_gives_charge_in_uc(tmp_path):
    p = tmp_path / 'x_RUNLOG_00.CSV'
    p.write_text(CSV_TEXT, encoding='latin-1')
    runs = runlog.read(str(p))
    assert set(runs) == {258304, 258305}
    assert runs[258305].charge_nC == pytest.approx(102.3)
    # nC in the log, uC in the interface - the unit slip that would be worth
    # a factor of a thousand in a concentration
    assert runs[258305].charge_uC == pytest.approx(0.1023)
    assert runs[258305].MeV == pytest.approx(1.0)
    assert runs[258305].adc['ADC0'] == 'Amptek'


def test_txt_and_csv_agree(tmp_path):
    c = tmp_path / 'a_RUNLOG_00.CSV'
    t = tmp_path / 'a_RUNLOG.TXT'
    c.write_text(CSV_TEXT, encoding='latin-1')
    t.write_text(TXT_TEXT, encoding='latin-1')
    rc = runlog.read(str(c))[258305]
    rt = runlog.read(str(t))[258305]
    assert rc.charge_nC == pytest.approx(rt.charge_nC)
    assert rc.MeV == pytest.approx(rt.MeV)


def test_zero_charge_is_zero_not_missing(tmp_path):
    """A run that really did collect no charge must not look like a run whose
    charge is unknown - one is a fact, the other is a gap."""
    p = tmp_path / 'x_RUNLOG_00.CSV'
    p.write_text(CSV_TEXT, encoding='latin-1')
    r = runlog.read(str(p))[258304]
    assert r.charge_nC == 0.0
    assert r.charge_uC == 0.0


def test_missing_log_returns_none_rather_than_guessing(tmp_path):
    f = tmp_path / '999999.lmf'
    f.write_bytes(b'\x00' * 16)
    assert runlog.find_for(str(f)) is None
    assert runlog.charge_for(str(f)) is None


def test_run_number_from_filename():
    assert runlog.run_number(r'C:\data\287427.lmf') == 287427
    assert runlog.run_number('nope.lmf') is None


@needs_session
def test_digitiser_quantum_is_one_tenth_nanocoulomb():
    """The LMF dose counter against the logged charge, over a real session.

    If this ever stops holding, absolute concentrations taken from the dose
    counter are wrong by the ratio, so it is worth asserting rather than
    remembering.
    """
    log = runlog.find_for(os.path.join(SESSION, '258304.lmf'))
    assert log is not None
    lmfs = sorted(glob.glob(os.path.join(SESSION, '*.lmf')))
    cal = runlog.calibrate_digitiser(lmfs, log)
    assert cal['n'] >= 10
    assert cal['correlation'] > 0.9999
    assert cal['frac_rms'] < 0.02
    assert cal['uC_per_count'] == pytest.approx(1.0e-4, rel=0.01)
