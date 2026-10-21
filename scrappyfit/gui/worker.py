"""One background thread for everything slow: fits, batches, the live overlay.

The GUI submits a Job and gets on with drawing; the job's function runs on a
single worker thread and its result comes back to the GUI thread through a
signal. One thread, one job at a time, so two jobs never touch a Session
together.

  - `submit(job)` starts it, or queues it behind the one running.
  - A queued 'preview' is replaced by a newer one: only the latest estimate
    is ever worth drawing. A preview running when a fit is submitted is told
    to stop; it does so at its next component (a SNIP background it is
    building finishes first), and the fit starts after it.
  - A 'fetch' (a download the user agreed to) is never stopped or
    superseded by a preview: a preview arriving while one is queued is
    dropped, and one arriving while one runs is queued behind it (where a
    newer preview still replaces it). A fetch ignores its stop flag, so
    shutdown() during a download waits for it, up to its timeout.
  - A job dropped from the queue, by cancel() or by a newer submission,
    still gets its on_cancel, so whoever locked the window for it unlocks.
  - `cancel()` asks the running fit or batch to stop. Cancellation is
    cooperative: the job's function polls `ctx.should_stop()` (the fit does
    so at every residual evaluation) and raises FitCancelled.

Signals, all delivered on the GUI thread:
  started(kind)                        a job began
  progress(kind, text, done, total)    total 0 means "busy, no count"
  finished(kind, result)               the function returned (a batch that
                                       was cancelled returns what it did)
  failed(kind, traceback_text)         it raised
  cancelled(kind)                      it raised FitCancelled on request
  idle()                               nothing running or queued
"""
import threading
import time
import traceback

from PyQt5 import QtCore

from ..fitting.fit import FitCancelled
from . import observe

_ORPHANS = []       # worker threads that outlived shutdown's wait


class Job:
    def __init__(self, kind, fn, on_done=None, on_fail=None, on_cancel=None,
                 tag=None):
        self.kind = kind            # 'fit', 'batch', 'preview', 'fetch'
        self.fn = fn                # fn(ctx) -> result, runs on the worker
        self.on_done = on_done      # called on the GUI thread with the result
        self.on_fail = on_fail
        self.on_cancel = on_cancel
        self.tag = tag              # caller's bookkeeping, e.g. a generation
        self.stop = threading.Event()
        self.t0 = None


class _Ctx:
    """What a job's function may use: a stop flag and a progress line."""

    def __init__(self, job, emit):
        self._job = job
        self._emit = emit

    def should_stop(self):
        return self._job.stop.is_set()

    def progress(self, text, done=0, total=0):
        self._emit(self._job, str(text), int(done), int(total))


class _Runner(QtCore.QObject):
    run = QtCore.pyqtSignal(object)
    done = QtCore.pyqtSignal(object, object)
    fail = QtCore.pyqtSignal(object, str)
    stopped = QtCore.pyqtSignal(object)
    step = QtCore.pyqtSignal(object, str, int, int)

    def __init__(self):
        super().__init__()
        self.run.connect(self._run)

    @QtCore.pyqtSlot(object)
    def _run(self, job):
        ctx = _Ctx(job, self.step.emit)
        try:
            out = job.fn(ctx)
        except FitCancelled:
            self.stopped.emit(job)
            return
        except Exception:
            self.fail.emit(job, traceback.format_exc())
            return
        # Returned normally: deliver it even if a stop was asked for. A batch
        # stopped between files returns the runs it finished, and a fit whose
        # last pass needed no refinement is complete. on_done can read
        # job.stop to tell.
        self.done.emit(job, out)


class JobRunner(QtCore.QObject):
    started = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, str, int, int)
    finished = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str)
    cancelled = QtCore.pyqtSignal(str)
    idle = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QtCore.QThread()
        self._thread.setObjectName('scrappyfit-worker')
        self._runner = _Runner()
        self._runner.moveToThread(self._thread)
        self._runner.done.connect(self._on_done)
        self._runner.fail.connect(self._on_fail)
        self._runner.stopped.connect(self._on_stopped)
        self._runner.step.connect(self._on_step)
        self._thread.start()
        self.current = None
        self.pending = None
        self._closed = False

    # -- state --------------------------------------------------------

    @property
    def busy(self):
        return self.current is not None or self.pending is not None

    def running(self, *kinds):
        """True while a job of one of these kinds runs or is queued."""
        return any(j is not None and j.kind in kinds
                   for j in (self.current, self.pending))

    def thread_running(self):
        return self._thread.isRunning()

    # -- control ------------------------------------------------------

    def submit(self, job):
        """Start `job`, or queue it. Returns it, or None if it was dropped."""
        if self._closed:
            return None
        cur = self.current
        if cur is None:
            self._start(job)
            return job
        if job.kind == 'preview':
            if self.running_heavy():
                # a fit or batch owns the session; its result supersedes
                # any estimate, so drop this one
                self._drop(job, '%s running' % cur.kind)
                return None
            if self.pending is not None and self.pending.kind != 'preview':
                # a queued fetch is work the user asked for; the estimate
                # is the one that gives way
                self._drop(job, '%s queued' % self.pending.kind)
                return None
            if self.pending is not None:
                self._drop(self.pending, 'superseded')
            self.pending = job
            if cur.kind == 'preview':
                cur.stop.set()          # the running one is already stale
            return job
        # a fit or batch: stop a running preview and go next
        if cur.kind == 'preview':
            cur.stop.set()
        if self.pending is not None:
            self._drop(self.pending, 'superseded by %s' % job.kind)
        self.pending = job
        return job

    def running_heavy(self):
        return self.running('fit', 'batch')

    def cancel(self):
        """Stop the running job and drop the queued one.

        A dropped job still gets its on_cancel, so a caller that locked the
        window when it submitted always hears back and can unlock it.
        """
        if self.pending is not None:
            job, self.pending = self.pending, None
            self._drop(job, 'cancelled while queued')
        if self.current is not None:
            self.current.stop.set()
            observe.log_event('job cancel requested', kind=self.current.kind)

    def shutdown(self, timeout_ms=30000):
        """Stop and join the worker. Nothing is delivered after this."""
        self._closed = True
        self.pending = None
        if self.current is not None:
            self.current.stop.set()
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            # never let a QThread be destroyed while it runs; keep it alive
            # until the process ends
            observe.log_event('worker thread did not stop', level=40)
            _ORPHANS.append((self._thread, self._runner))

    # -- internals ----------------------------------------------------

    def _start(self, job):
        self.current = job
        job.t0 = time.perf_counter()
        observe.log_event('job start', kind=job.kind)
        self.started.emit(job.kind)
        self._runner.run.emit(job)

    def _drop(self, job, why):
        observe.log_event('job dropped', kind=job.kind, reason=why)
        if job.on_cancel is not None:
            job.on_cancel()
        self.cancelled.emit(job.kind)

    def _ms(self, job):
        return round(1e3 * (time.perf_counter() - job.t0), 1)

    def _finish(self, job, call):
        """Run a finished job's callbacks with the runner already free, so a
        callback that submits work (a redraw scheduling an estimate) is
        treated like any other submission, then start what was queued."""
        if self._closed or job is not self.current:
            return
        self.current = None
        nxt, self.pending = self.pending, None
        try:
            call()
        finally:
            if nxt is not None and self.current is not None                     and nxt.kind == 'preview':
                self._drop(nxt, 'superseded')    # the callback started newer work
            elif nxt is not None:
                self.submit(nxt)
            elif self.current is None and self.pending is None:
                self.idle.emit()

    def _on_step(self, job, text, done, total):
        if job is self.current and not self._closed:
            self.progress.emit(job.kind, text, done, total)

    def _on_done(self, job, result):
        observe.log_event('job done', kind=job.kind, ms=self._ms(job))

        def call():
            if job.on_done is not None:
                job.on_done(result)
            self.finished.emit(job.kind, result)
        self._finish(job, call)

    def _on_fail(self, job, tb):
        observe.log_event('job failed', level=40, kind=job.kind,
                          ms=self._ms(job), traceback=tb)

        def call():
            if job.on_fail is not None:
                job.on_fail(tb)
            self.failed.emit(job.kind, tb)
        self._finish(job, call)

    def _on_stopped(self, job):
        observe.log_event('job cancelled', kind=job.kind, ms=self._ms(job))

        def call():
            if job.on_cancel is not None:
                job.on_cancel()
            self.cancelled.emit(job.kind)
        self._finish(job, call)
