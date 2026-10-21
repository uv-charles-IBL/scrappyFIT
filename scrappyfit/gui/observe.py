"""What the GUI did, written down: the action log and the error hooks.

Every user action goes to ~/.scrappyfit/logs/<YYYY-MM-DD>.log as one JSON
line (``SCRAPPYFIT_LOG_DIR`` moves it). An action's line holds its name, its
arguments, how long it took, and a context snapshot (file, calibration, fit
range, elements). Read in order, the log is enough to repeat what was done
before something broke.

Unhandled exceptions, including ones raised inside Qt slots, and Qt's own
critical messages go to the same file at ERROR level. They also raise a
non-modal dialog, and the program keeps running. PyQt5 aborts on an
exception in a slot only while sys.excepthook is the default, so installing
a hook here is also what keeps one bad click from killing the session.
"""
import datetime
import functools
import inspect
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback

LOGGER = logging.getLogger('scrappyfit.gui')

_state = {'handler': None, 'old_hook': None, 'old_thread_hook': None,
          'old_qt': None, 'installed': False, 'dialogs': [], 'notifier': None}


def resolve_log_dir(path=None):
    d = path or os.environ.get('SCRAPPYFIT_LOG_DIR') or os.path.join(
        os.path.expanduser('~'), '.scrappyfit', 'logs')
    return pathlib.Path(d)


class _JsonLines(logging.Formatter):
    def format(self, rec):
        out = {'t': datetime.datetime.fromtimestamp(rec.created)
               .isoformat(timespec='milliseconds'),
               'level': rec.levelname, 'event': rec.getMessage()}
        out.update(getattr(rec, 'fields', {}) or {})
        if rec.exc_info:
            out['traceback'] = ''.join(traceback.format_exception(*rec.exc_info))
        return json.dumps(out, default=_jsonable)


class _DailyFile(logging.Handler):
    """One file per calendar day. Kept open and flushed after every record,
    so a crash loses nothing; reopened when the date changes, so a session
    that runs past midnight rolls over."""

    def __init__(self, folder):
        super().__init__()
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.setFormatter(_JsonLines())
        self._fh = None
        self._open_path = None

    def path(self):
        return self.folder / ('%s.log' % datetime.date.today().isoformat())

    def emit(self, rec):
        try:
            line = self.format(rec)
            p = self.path()
            if p != self._open_path:
                self.close_file()
                self._fh = open(p, 'a', encoding='utf-8')
                self._open_path = p
            self._fh.write(line + '\n')
            self._fh.flush()
        except Exception:
            self.handleError(rec)

    def close_file(self):
        if self._fh is not None:
            self._fh.close()
        self._fh = self._open_path = None

    def close(self):
        self.close_file()
        super().close()


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return '<array %s>' % (x.shape,)
    except ImportError:
        pass
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, pathlib.PurePath):
        return str(x)
    return repr(x)


def log_event(event, level=logging.INFO, **fields):
    LOGGER.log(level, event, extra={'fields': fields})


def handler_count():
    return sum(isinstance(h, _DailyFile) for h in LOGGER.handlers)


# ------------------------------------------------------------- error hooks

def _show_error(title, text):
    """Raise the warning box on the GUI thread, whichever thread failed.

    Qt widgets may only be made on the GUI thread, and an exception or a Qt
    message can arrive from the worker. The notifier lives on the GUI thread,
    so emitting its signal from elsewhere queues the dialog over to it.
    """
    n = _state.get('notifier')
    if n is not None:
        try:
            n.show.emit(str(title), str(text))
        except RuntimeError:
            pass                        # notifier gone with the app


def _make_notifier():
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None

    class _Notifier(QtCore.QObject):
        show = QtCore.pyqtSignal(str, str)

    n = _Notifier()
    n.moveToThread(app.thread())
    n.show.connect(_dialog)
    return n


def _dialog(title, text):
    """A non-modal box, so the window stays usable behind it."""
    try:
        from PyQt5 import QtCore, QtWidgets
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title,
                                    text[:2000])
        box.setDetailedText(text)
        box.setModal(False)
        box.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        box.show()
        _state['dialogs'].append(box)
        del _state['dialogs'][:-5]
    except Exception:
        pass


def _excepthook(etype, value, tb):
    LOGGER.error('unhandled exception', exc_info=(etype, value, tb),
                 extra={'fields': {'type': etype.__name__, 'message': str(value)}})
    _show_error('scrappyFIT: something went wrong',
                '%s: %s\n\nThe session is still running. The details are in '
                '%s.' % (etype.__name__, value, _log_path_text()))


def _thread_hook(args):
    _excepthook(args.exc_type, args.exc_value, args.exc_traceback)


def _qt_handler(mode, context, message):
    from PyQt5 import QtCore
    level = {QtCore.QtDebugMsg: logging.DEBUG,
             QtCore.QtInfoMsg: logging.INFO,
             QtCore.QtWarningMsg: logging.WARNING,
             QtCore.QtCriticalMsg: logging.ERROR,
             QtCore.QtFatalMsg: logging.CRITICAL}.get(mode, logging.WARNING)
    log_event('qt message', level, message=message,
              file=context.file, line=context.line)
    if level >= logging.ERROR:
        _show_error('scrappyFIT: Qt error', message)


def _log_path_text():
    h = _state['handler']
    return str(h.path()) if h else 'the log'


def install(log_dir=None):
    """Attach the file log and the error hooks. Calling it twice is harmless."""
    folder = resolve_log_dir(log_dir)
    if _state['handler'] is not None and _state['handler'].folder != folder:
        uninstall()
    if _state['handler'] is None:
        h = _DailyFile(folder)
        LOGGER.addHandler(h)
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.propagate = False
        _state['handler'] = h
    if not _state['installed']:
        _state['old_hook'] = sys.excepthook
        sys.excepthook = _excepthook
        _state['old_thread_hook'] = threading.excepthook
        threading.excepthook = _thread_hook
        try:
            from PyQt5 import QtCore
            _state['old_qt'] = QtCore.qInstallMessageHandler(_qt_handler)
            _state['notifier'] = _make_notifier()
        except ImportError:
            pass
        _state['installed'] = True
        log_event('session start', python=sys.version.split()[0],
                  argv=sys.argv, log=str(_state['handler'].path()))
    return _state['handler'].path()


def uninstall():
    if _state['installed']:
        sys.excepthook = _state['old_hook'] or sys.__excepthook__
        threading.excepthook = (_state['old_thread_hook']
                                or threading.__excepthook__)
        try:
            from PyQt5 import QtCore
            QtCore.qInstallMessageHandler(_state['old_qt'])
        except ImportError:
            pass
        _state['notifier'] = None
        _state['installed'] = False
    h = _state['handler']
    if h is not None:
        LOGGER.removeHandler(h)
        h.close()
        _state['handler'] = None


# ------------------------------------------------------------ action log

def _short(v, limit=200):
    s = v if isinstance(v, (int, float, bool, type(None))) else None
    if s is not None or isinstance(v, (list, tuple, dict)):
        try:
            txt = json.dumps(v, default=_jsonable)
            return v if len(txt) <= limit else txt[:limit] + '...'
        except Exception:
            pass
    txt = str(v)
    return txt if len(txt) <= limit else txt[:limit] + '...'


def user_action(fn):
    """Log a GUI handler: name, arguments, duration, outcome, and context.

    Qt passes a signal's own arguments (clicked's `checked`) to whatever is
    connected. The wrapper drops the ones the handler does not declare, so
    decorating a no-argument slot does not change what it receives.

    The owning window may define `_log_context()` (state worth recording
    with every action) and `_refuse_while_busy(name)` (True to skip an
    action that would change the session under a running job).
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[1:]
    var = any(p.kind == p.VAR_POSITIONAL for p in params)
    npos = len([p for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    names = [p.name for p in params]

    @functools.wraps(fn)
    def wrapper(self, *args, **kw):
        if not var:
            args = args[:npos]
        name = fn.__name__
        refuse = getattr(self, '_refuse_while_busy', None)
        if refuse is not None and refuse(name):
            log_event('action refused', action=name, reason='job running')
            return None
        shown = {n: _short(a) for n, a in zip(names, args)}
        shown.update({k: _short(v) for k, v in kw.items()})
        t0 = time.perf_counter()
        ok, err = True, None
        try:
            return fn(self, *args, **kw)
        except Exception as ex:
            ok, err = False, '%s: %s' % (type(ex).__name__, ex)
            raise
        finally:
            ctx = {}
            get = getattr(self, '_log_context', None)
            if get is not None:
                try:
                    ctx = get()
                except Exception as ex:
                    ctx = {'context_error': str(ex)}
            fields = dict(action=name, args=shown,
                          ms=round(1e3 * (time.perf_counter() - t0), 1),
                          ok=ok, ctx=ctx)
            if err:
                fields['error'] = err
            log_event('action', **fields)
    wrapper.__wrapped_action__ = True
    return wrapper
