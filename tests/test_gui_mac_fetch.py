"""GUI: choosing the Sabbatucci table, which is not shipped, offers to fetch it.

No real dialog and no network: QMessageBox.question and
scrappyfit.io.sabbatucci.fetch are monkeypatched, and SCRAPPYFIT_DATA_DIR
points at an empty folder so the table is missing even on a machine that has
fetched it.
"""
import pytest

from scrappyfit import config

pytestmark = pytest.mark.skipif(
    config.database_path(required=False) is None,
    reason='no GeoPIXE database available')

QtWidgets = pytest.importorskip('PyQt5.QtWidgets')

SAB = 'sabbatuccisalvat2016'


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    monkeypatch.setenv('SCRAPPYFIT_DATA_DIR', str(tmp_path / 'data'))
    # the bundled database: an external GeoPIXE tree may hold its own copy
    import scrappyfit
    import pathlib
    monkeypatch.setenv('SCRAPPYFIT_DB', str(
        pathlib.Path(scrappyfit.__file__).parent / 'resources' / 'database'))
    from scrappyfit.gui.app import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w.close()


def _answer(monkeypatch, yes):
    QMB = QtWidgets.QMessageBox
    asked = []

    def question(*a, **k):
        asked.append(a[2] if len(a) > 2 else '')
        return QMB.Yes if yes else QMB.No
    monkeypatch.setattr(QMB, 'question', staticmethod(question))
    return asked


def _fetch(monkeypatch, fn):
    from scrappyfit.io import sabbatucci
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return fn()
    monkeypatch.setattr(sabbatucci, 'fetch', fake)
    return calls


def test_declining_the_download_reverts_the_box(window, monkeypatch):
    asked = _answer(monkeypatch, yes=False)
    calls = _fetch(monkeypatch, lambda: pytest.fail('fetched after a No'))
    window.cmb_mac.setCurrentText(SAB)
    assert asked and 'AGPL-3.0' in asked[0] and 'EGSnrc' in asked[0]
    assert not calls
    assert window.cmb_mac.currentText() == 'mixed'
    assert window.cmb_mac.isEnabled()
    assert 'declined' in window.log.toPlainText()


def test_accepting_runs_the_fetch_as_a_job_and_keeps_the_choice(
        qtbot, window, monkeypatch, tmp_path):
    from scrappyfit.io import sabbatucci
    _answer(monkeypatch, yes=True)

    def install():
        # the job runs this on the worker thread; the box is locked meanwhile
        assert not window.cmb_mac.isEnabled()
        p = sabbatucci.installed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Element    Fe   26    55.847000\nData\n'
                     '  1.000000E+00  1.000000E+03\n'
                     '  1.000000E+01  1.000000E+02\nEndElement\nEnd\n')
        return p
    calls = _fetch(monkeypatch, install)
    window.cmb_mac.setCurrentText(SAB)
    qtbot.waitUntil(lambda: window.cmb_mac.isEnabled(), timeout=10000)
    assert calls == [1]
    assert window.cmb_mac.currentText() == SAB
    assert window.session.db.mac_installed(SAB)
    assert 'Installed' in window.log.toPlainText()


def test_a_failed_download_reverts_and_says_why(qtbot, window, monkeypatch):
    _answer(monkeypatch, yes=True)

    def boom():
        raise OSError('no route to host')
    _fetch(monkeypatch, boom)
    window.cmb_mac.setCurrentText('ffast')
    window.cmb_mac.setCurrentText(SAB)
    qtbot.waitUntil(lambda: window.cmb_mac.isEnabled(), timeout=10000)
    assert window.cmb_mac.currentText() == 'ffast'
    log = window.log.toPlainText()
    assert 'download failed' in log and 'no route to host' in log
