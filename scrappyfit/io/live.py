"""Reading an OMDAQ list-mode file while it is still being written.

This is easier than it sounds, and the reason is the format. An LMF is a fixed
header followed by 8192-byte blocks, each self-contained: five int32 counters
then 2043 event records. A block is either wholly present or not there yet.

So a live reader does not need to parse a partial structure or coordinate with
the writer. It remembers how many whole blocks it has consumed, and whenever
the file has grown by at least one more, it reads that. No locking, no shared
memory, no OMDAQ cooperation required - it works on a file being written by
another process on a network share.

What it does NOT do is guarantee it sees every block the instant it lands.
Poll interval sets the latency. For watching a map fill in, one second is
imperceptible; for a count-rate readout, use a shorter one.

    w = LiveLMF('run.lmf')
    while acquiring:
        if w.poll():
            spectrum = w.spectrum(adc=0)
            xs, ys, es = w.events(adc=0)
"""

import os
import time

import numpy as np

from .lmf import ADC_MASK, ADC_SHIFT, BLOCK, E_MASK, NHDR_I32, REC, header_size


class LiveLMF:
    """Incremental reader over a growing list-mode file.

    Accumulates a spectrum and, optionally, the full event list. Keeping every
    event costs memory - a long run is tens of millions of them - so
    keep_events can be turned off when only the spectrum and count rate are
    wanted.
    """

    def __init__(self, path, keep_events=True, max_adc=8):
        self.path = str(path)
        self.keep_events = keep_events
        self.max_adc = max_adc
        self._hdr = None
        self.blocks_read = 0
        self.charge = 0
        self.elapsed = 0
        self.spectra = np.zeros((max_adc, 4096), dtype=np.int64)
        self._ev = {a: [[], [], []] for a in range(max_adc)} if keep_events else None
        self.last_poll = None
        self.last_new_blocks = 0

    # -- internals ------------------------------------------------------

    def _ensure_header(self):
        if self._hdr is None:
            # header_size needs a whole number of blocks present, which is not
            # true the instant acquisition starts; retry until it is
            self._hdr = header_size(self.path)
        return self._hdr

    def _available_blocks(self):
        try:
            n = os.path.getsize(self.path)
        except OSError:
            return 0
        h = self._ensure_header()
        return max((n - h) // BLOCK, 0)

    # -- public ---------------------------------------------------------

    def poll(self):
        """Consume any whole blocks that have appeared. Returns how many."""
        try:
            avail = self._available_blocks()
        except Exception:
            return 0
        new = avail - self.blocks_read
        self.last_poll = time.time()
        self.last_new_blocks = 0
        if new <= 0:
            return 0

        h = self._ensure_header()
        with open(self.path, 'rb') as fh:
            fh.seek(h + self.blocks_read * BLOCK)
            raw = np.frombuffer(fh.read(new * BLOCK), dtype='u1')
        got = len(raw) // BLOCK
        if got <= 0:
            return 0
        raw = raw[:got * BLOCK].reshape(got, BLOCK)

        counters = raw[:, :4 * NHDR_I32].copy().view('>i4')
        # LMF counters are little-endian on the platforms we have seen; read
        # both ways and take the one that is monotonic and plausible
        le = raw[:, :4 * NHDR_I32].copy().view('<i4')
        counters = le if abs(int(le[:, 0].sum())) < abs(int(counters[:, 0].sum())) \
            else counters
        self.charge += int(counters[:, 0].sum())
        if counters.shape[1] > 1:
            self.elapsed = int(counters[-1, 1])

        rec = raw[:, 4 * NHDR_I32:].copy().view(REC)
        e = rec['e'].ravel()
        good = (e & E_MASK) != E_MASK
        en = (e[good] & E_MASK).astype(np.int32)
        ad = ((e[good] >> ADC_SHIFT) & ADC_MASK).astype(np.int8)
        xs = rec['x'].ravel()[good]
        ys = rec['y'].ravel()[good]

        for a in range(self.max_adc):
            m = ad == a
            if not m.any():
                continue
            self.spectra[a] += np.bincount(en[m], minlength=4096)
            if self.keep_events:
                self._ev[a][0].append(xs[m])
                self._ev[a][1].append(ys[m])
                self._ev[a][2].append(en[m])

        self.blocks_read += got
        self.last_new_blocks = got
        return got

    def spectrum(self, adc=0):
        return self.spectra[adc].astype(float)

    def events(self, adc=0):
        """(x, y, energy) accumulated so far. Empty arrays if not kept."""
        if not self.keep_events:
            return (np.array([], np.int32),) * 3
        cols = self._ev[adc]
        if not cols[0]:
            return (np.array([], np.int32),) * 3
        return tuple(np.concatenate(c) for c in cols)

    def total_counts(self, adc=0):
        return int(self.spectra[adc].sum())

    def status(self):
        return ('%d blocks, %d events in ADC0, charge %d'
                % (self.blocks_read, self.total_counts(0), self.charge))

    def watch(self, interval=1.0, on_update=None, stop=None):
        """Block and poll until `stop()` returns True.

        on_update(reader, n_new_blocks) is called only when something arrived,
        so a callback that redraws will not redraw a static plot forever.
        """
        while True:
            if stop is not None and stop():
                return
            n = self.poll()
            if n and on_update is not None:
                on_update(self, n)
            time.sleep(interval)


def attach_to_session(session, path, adc=0, keep_events=True):
    """Point a Session at a growing file and return the reader.

    Calling reader.poll() then session.refresh_from(reader) keeps the session
    current. Deliberately explicit rather than automatic - a fit should happen
    when asked, not silently on data that changed underneath it.
    """
    r = LiveLMF(path, keep_events=keep_events)
    r.poll()
    session.reset_data()
    session.path = str(path)
    session.label = os.path.basename(str(path)) + ' [live]'
    session.adc = adc
    refresh_session(session, r, adc)
    return r


def refresh_session(session, reader, adc=0):
    """Copy the reader's current state into a Session."""
    session.spectrum = reader.spectrum(adc)
    session.full_spectrum = session.spectrum.copy()
    session.charge = float(reader.charge)
    if reader.keep_events:
        x, y, e = reader.events(adc)
        session.events = (x.astype(np.int32), y.astype(np.int32),
                          e.astype(np.int32)) if len(x) else None
    session.invalidate()
    return session
