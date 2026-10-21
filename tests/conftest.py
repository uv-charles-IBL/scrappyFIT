"""Shared test setup.

The GUI journeys need a Qt platform that draws nowhere. Set it before
pytest-qt creates the QApplication; CI sets the same variable.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


import pytest


@pytest.fixture(autouse=True)
def _isolated_yield_store(tmp_path, monkeypatch):
    """Remembered yield files and the yield search path live in
    ~/.scrappyfit/yields.json; a test must never read or write the user's."""
    monkeypatch.setenv('SCRAPPYFIT_YIELD_STORE',
                       str(tmp_path / 'yield_store.json'))
