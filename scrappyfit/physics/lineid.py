"""X-ray line identification and element suggestion.

Two jobs, and the second is the interesting one.

**Identify** answers "what could be at 2.31 keV". That is a lookup, and the
only real design question is how to rank the answers, because at any energy in
a crowded spectrum there are a dozen candidate lines and most are absurd. The
ranking here weighs proximity against the line's own intensity and against
whether it is the element's principal line: a sulfur Ka at 8 eV away should
outrank a lanthanide Lb3 at 3 eV away, because one of those is a line you
would actually see and the other is a fifth-order transition of an element
that would announce itself elsewhere.

**Suggest** answers "which elements should I add to this fit". That is
different, and stronger, because it uses the whole spectrum rather than one
energy. An element is worth suggesting when SEVERAL of its lines line up with
observed peaks in roughly the right intensity ratio. One coincidence is
nothing - in a light-element spectrum every channel is within 60 eV of
something - but three lines in the right proportions is an element.

This is the same reasoning the screening tests in physics/screen.py apply
after a fit. Doing it before the fit as well means the suspect assignments are
less likely to get in.

Cost: peak finding is one pass over 4096 channels, and matching is a few
thousand comparisons. Negligible - it can run on every redraw.
"""

import numpy as np

# Principal lines, used to weight a candidate. An element's Ka is what you see
# first; its Lb4 is not evidence of anything on its own.
PRINCIPAL = {'Ka1', 'Ka_', 'Ka2', 'La1', 'La_', 'Ma1', 'Ma'}
STRONG = PRINCIPAL | {'Kb1', 'Kb_', 'Lb1', 'Lb_'}


class Candidate:
    __slots__ = ('Z', 'symbol', 'line', 'energy', 'intensity', 'shell',
                 'delta_eV', 'score')

    def __init__(self, Z, symbol, line, energy, intensity, shell, delta_eV,
                 score):
        self.Z, self.symbol, self.line = Z, symbol, line
        self.energy, self.intensity, self.shell = energy, intensity, shell
        self.delta_eV, self.score = delta_eV, score

    def label(self):
        return '%s %s' % (self.symbol, self.line)

    def __repr__(self):
        return ('<%s %s %.4f keV  %+.0f eV  I=%.3f  score %.2f>'
                % (self.symbol, self.line, self.energy, self.delta_eV,
                   self.intensity, self.score))


def _shell_of(name):
    return {'K': 1, 'L': 2, 'M': 3}.get(name[:1].upper(), 0)


def identify(db, energy_keV, window_eV=80.0, z_range=(5, 92),
             min_intensity=0.02, limit=12):
    """Candidate lines near an energy, best first.

    window_eV  how far to look. Default 80 eV is about one detector resolution
               at low energy, which is the honest search radius - anything
               further is not what you clicked on.
    """
    out = []
    for Z in range(max(z_range[0], 1), min(z_range[1], 92) + 1):
        E = db.lineE.get(Z)
        if not E:
            continue
        I = db.lineI.get(Z, {})
        for nm, e in E.items():
            if e <= 0:
                continue
            d = (e - energy_keV) * 1000.0
            if abs(d) > window_eV:
                continue
            inten = I.get(nm, 0.0)
            if inten < min_intensity:
                continue
            # proximity, weighted by how believable the line is
            prox = np.exp(-0.5 * (d / (window_eV / 2.0)) ** 2)
            weight = 1.0 if nm in PRINCIPAL else (0.6 if nm in STRONG else 0.25)
            out.append(Candidate(Z, db.sym[Z], nm, e, inten, _shell_of(nm),
                                 d, prox * weight * (0.3 + inten)))
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def find_peaks(counts, cal, background=None, e_low=0.15, e_high=None,
               min_sigma=4.0, min_separation_eV=60.0):
    """Peaks worth explaining, as a list of (energy_keV, net_counts, sigma).

    Significance is measured against the local background rather than against
    zero, because in a PIXE spectrum the continuum is large and a peak that is
    small compared with it is not a detection.
    """
    y = np.asarray(counts, dtype=float)
    a, b = cal
    E = a * np.arange(len(y)) + b
    if background is None:
        background = _rough_background(y)
    bk = np.asarray(background, dtype=float)[:len(y)]
    net = y - bk
    noise = np.sqrt(np.maximum(bk, 1.0))

    hi = e_high if e_high is not None else E[-1]
    ok = (E >= e_low) & (E <= hi)
    sig = np.where(ok, net / noise, 0.0)

    sep = max(int(min_separation_eV / 1000.0 / max(a, 1e-9)), 2)
    peaks = []
    i = 1
    while i < len(y) - 1:
        if sig[i] >= min_sigma:
            lo = max(i - sep, 0)
            hh = min(i + sep + 1, len(y))
            j = lo + int(np.argmax(sig[lo:hh]))
            if j == i:
                w0 = max(i - sep, 0)
                w1 = min(i + sep + 1, len(y))
                peaks.append((float(E[i]), float(net[w0:w1].sum()),
                              float(sig[i])))
                i += sep
                continue
        i += 1
    return peaks


def _rough_background(y, iterations=24):
    """A cheap SNIP, only for peak finding. The real background comes from
    fitting.background; this exists so peak detection works before any fit
    and does not need options set."""
    z = np.log(np.log(np.maximum(y, 0) + 1.0) + 1.0)
    n = len(z)
    for w in range(1, iterations):
        lo = np.roll(z, w)
        hi = np.roll(z, -w)
        lo[:w] = z[:w]
        hi[-w:] = z[-w:]
        z = np.minimum(z, 0.5 * (lo + hi))
    return np.exp(np.exp(z) - 1.0) - 1.0


def suggest_elements(db, peaks, z_range=(5, 92), window_eV=70.0,
                     shells=(1, 2, 3), min_score=0.35, limit=25,
                     significant=0.10):
    """Which elements would explain these peaks.

    The scoring exists to reject coincidences, which in a light-element
    spectrum are everywhere - at 70 eV tolerance almost every channel is near
    something. Three rules do the work:

    1. **The element's strongest visible line must be matched.** If tin's
       La is absent, a stray match on its Lb4 is not tin. This single rule
       removes most of the nonsense, because a real element announces itself
       on its principal line first.

    2. **Score is the fraction of the element's visible intensity that is
       accounted for**, so an element with three visible lines has to explain
       all three, not one.

    3. **Multi-line matches are rewarded** over single-line ones, since one
       coincidence is cheap and three in the right proportions is not. Light
       elements which genuinely have only one line are not penalised for it -
       the bonus is a multiplier on top, not a requirement.

    Returns [(symbol, shell, score, n_matched)], best first.
    """
    if not peaks:
        return []
    pe = np.array([p[0] for p in peaks])
    pnet = np.array([max(p[1], 1.0) for p in peaks])
    # How much of the spectrum each peak represents. Without this an element
    # matching a 400-count peak scores the same as one matching the million-
    # count peak next to it, and in a quartz spectrum that put indium above
    # silicon. Compressed with a fourth root so a dominant peak counts for
    # more without swamping everything else.
    pw = (pnet / pnet.max()) ** 0.25
    emin, emax = pe.min() - 0.15, pe.max() + 0.15

    def nearest(e):
        d = np.abs(pe - e) * 1000.0
        k = int(np.argmin(d))
        return float(d[k]), float(pw[k])

    def dist(e):
        return nearest(e)[0]

    out = []
    for Z in range(max(z_range[0], 1), min(z_range[1], 92) + 1):
        for sh in shells:
            try:
                lines = db.line_list(Z, sh)
            except Exception:
                lines = []
            if not lines:
                continue
            visible = [(e, i) for e, i in lines if emin <= e <= emax]
            if not visible:
                continue
            denom = sum(i for _, i in visible)
            if denom <= 0:
                continue

            # rule 1: the strongest visible line has to be there
            strongest = max(visible, key=lambda t: t[1])
            if dist(strongest[0]) > window_eV:
                continue

            # rule 2: every line the element SHOULD show must be accounted
            # for. This is the rule that matters. A heavy element's L series
            # is dense enough that one of its lines will land near something
            # in any spectrum; requiring all of its strong lines to appear is
            # what separates a real element from a coincidence.
            matched, missed, n = 0.0, 0.0, 0
            for e, inten in visible:
                d, weight = nearest(e)
                if d <= window_eV:
                    prox = np.exp(-0.5 * (d / (window_eV / 2.0)) ** 2)
                    matched += inten * prox * weight
                    n += 1
                elif inten >= significant:
                    missed += inten

            # fraction of the element's expected emission actually seen,
            # penalised by whatever is conspicuously absent
            score = (matched - missed) / denom
            if n > 1:
                score *= 1.0 + 0.25 * np.log(n)
            if score >= min_score:
                out.append((db.sym[Z], sh, float(score), n))
    out.sort(key=lambda t: -t[2])
    return out[:limit]


def describe(db, energy_keV, window_eV=80.0, limit=8):
    """Human-readable identification, for a status bar or a log."""
    cands = identify(db, energy_keV, window_eV=window_eV, limit=limit)
    if not cands:
        return 'nothing tabulated within %.0f eV of %.4f keV' % (window_eV,
                                                                 energy_keV)
    head = '%.4f keV - candidates:' % energy_keV
    rows = ['   %-8s %8.4f keV  %+5.0f eV   rel.int %.3f'
            % (c.label(), c.energy, c.delta_eV, c.intensity) for c in cands]
    return head + '\n' + '\n'.join(rows)
