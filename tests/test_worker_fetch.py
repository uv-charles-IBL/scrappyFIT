"""JobRunner with a 'fetch' job (the Sabbatucci download): other jobs queue
behind it as usual, and a live estimate never cancels it."""
import threading

import pytest

pytest.importorskip('PyQt5.QtCore')


@pytest.fixture
def jr():
    from scrappyfit.gui.worker import JobRunner
    r = JobRunner()
    yield r
    r.shutdown()


def _download(gate, tag='table'):
    # like the GUI's fetch work: never polls ctx.should_stop()
    def fn(ctx):
        gate.wait(5)
        return tag
    return fn


def test_a_preview_during_a_fetch_runs_after_it_and_does_not_cancel_it(
        qtbot, jr):
    from scrappyfit.gui.worker import Job
    gate, got, cancelled = threading.Event(), [], []
    fetch = jr.submit(Job('fetch', _download(gate), on_done=got.append,
                          on_cancel=lambda: cancelled.append('fetch')))
    jr.submit(Job('preview', lambda ctx: 'estimate', on_done=got.append))
    assert not fetch.stop.is_set()      # never told to stop by an estimate
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert got == ['table', 'estimate'] and cancelled == []


def test_a_fit_during_a_fetch_queues_and_runs(qtbot, jr):
    from scrappyfit.gui.worker import Job
    gate, got = threading.Event(), []
    jr.submit(Job('fetch', _download(gate), on_done=got.append))
    jr.submit(Job('fit', lambda ctx: 'fit', on_done=got.append))
    assert jr.running_heavy()
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert got == ['table', 'fit']


def test_a_preview_does_not_drop_a_queued_fetch(qtbot, jr):
    """The fetch waits behind a running estimate; a newer estimate must not
    supersede it (the user agreed to a download, not to an overlay). The
    newer estimate is the one dropped."""
    from scrappyfit.gui.worker import Job
    gate, got, cancelled = threading.Event(), [], []
    jr.submit(Job('preview', _download(gate, 'old estimate')))
    fetch = jr.submit(Job('fetch', lambda ctx: 'table', on_done=got.append,
                          on_cancel=lambda: cancelled.append('fetch')))
    late = jr.submit(Job('preview', lambda ctx: 'estimate',
                         on_done=got.append))
    assert late is None and jr.pending is fetch
    gate.set()
    qtbot.waitUntil(lambda: not jr.busy, timeout=5000)
    assert cancelled == [] and got == ['table']
