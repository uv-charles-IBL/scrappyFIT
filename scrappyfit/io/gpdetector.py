"""Reader for GeoPIXE .detector files, and the efficiency they describe.

read_detector.pro walks a version-branched header, but the part that matters
is regular: everything is built from make_layer.pro's struct

    layer = {N: int, Z: int[32], F: float[32], thick: float, name: string}

which is 66 XDR words before the name, with thick already in mg/cm^2
(make_layer converts microns using the density). The absorbers and the
crystal are all layers, so the file can be read by locating those blocks and
checking they are self-consistent: N must be small, Z[0] a real element, the
weight fractions must sum to about 1, and the thickness must be positive.

Verified against Canberra-34.detector, the detector GeoPIXE's own donut2x
example was measured on:

    Be window        Be   5.7696   mg/cm2   (~31 um)
    Al collimator    Al   191.629  mg/cm2   (~710 um)
    Ge dead layer    Ge   0.223128 mg/cm2   (~0.42 um)
    Au contact       Au   0.000193 mg/cm2   (~0.1 nm)
    crystal          Ge   5338     mg/cm2   (~1.0 mm)

The collimator is the one judgement call. At 710 um of aluminium it is
opaque to everything below about 20 keV, so it cannot be in the photon path -
it is an APERTURE defining the solid angle, not a filter. Including it as an
absorber would predict essentially zero efficiency across the whole PIXE
range. is_aperture() flags it and efficiency() drops it by default.
"""

import os
import re

import numpy as np

LAYER_WORDS = 66          # N + Z[32] + F[32] + thick


def _mu(db, Z, F, e, mac):
    """mu for a compound, NaN when the dataset does not reach this energy.

    Henke stops at 30 keV, so a high-energy detector model needs FFAST or
    XCOM. Returning 0 here would make the crystal look transparent and the
    efficiency zero, which is indistinguishable from a real result."""
    v = db.mu_compound(Z, F, e, mac)
    return float('nan') if v is None else float(v)


def _layer_at(i4, v4, k):
    """Decode a layer at word k, or None if it does not look like one."""
    if k + LAYER_WORDS > len(i4):
        return None
    n = int(i4[k])
    if not (1 <= n <= 8):
        return None
    Z = [int(z) for z in i4[k + 1:k + 33]][:n]
    F = [float(f) for f in v4[k + 33:k + 65]][:n]
    thick = float(v4[k + LAYER_WORDS - 1])
    if not all(1 <= z <= 92 for z in Z):
        return None
    if not all(np.isfinite(F)) or sum(F) <= 0:
        return None
    if abs(sum(F) - 1.0) > 0.05:
        return None
    if not np.isfinite(thick) or thick < 0 or thick > 1e6:
        return None
    return dict(N=n, Z=Z, F=F, thick=thick)


def read_detector(path):
    """Return a dict with 'absorbers' (list of layers) and 'crystal'.

    Layers are found by scanning for self-consistent blocks rather than by
    replaying every version branch of read_detector.pro. Each hit is checked
    against the name strings in the file so the caller can see what matched.
    """
    raw = open(path, 'rb').read()
    n = len(raw) // 4
    i4 = np.frombuffer(raw[:n * 4], '>i4')
    v4 = np.frombuffer(raw[:n * 4], '>f4')
    names = [s.decode('latin-1')
             for s in re.findall(rb'[A-Za-z][ -~]{2,}', raw)]

    layers, k = [], 1
    while k < n - LAYER_WORDS:
        lay = _layer_at(i4, v4, k)
        if lay is not None:
            layers.append((k, lay))
            k += LAYER_WORDS
        else:
            k += 1

    # attach the nearest preceding descriptive string to each layer
    for idx, (k, lay) in enumerate(layers):
        lay['name'] = names[idx * 2] if idx * 2 < len(names) else ''

    out = dict(version=int(i4[0]), path=path, names=names,
               absorbers=[l for _, l in layers[:-1]],
               crystal=layers[-1][1] if layers else None)
    if layers:
        out.update(_scalars_after(raw, i4, v4, layers[-1][0]))
    return out


def _scalars_after(raw, i4, v4, k_crystal):
    """The scalar block that follows the crystal layer.

    read_detector.pro reads, in order:

        crystal (a layer)
        diameter, density, distance, source
        gamma_factor, w0, w1, resolution
        aeff, beff
        tail {F, B, amp, L, S}

    The layer ends with its thickness at k+65, then an XDR string for the
    name, then these fifteen floats. Skipping the string by its own length
    is exact, so no scanning is needed here.

    On Canberra-34 this yields density 5.338, gamma_factor 0.24 (germanium,
    against silicon's 0.022), w1 0.002089 - which matches the w1 in the .pfr
    written from it - and tail {F 12 um, B 9850 um, amp 0.035, L 0.6978,
    S 0.064}, matching the .pfr's tail exactly.
    """
    p = (k_crystal + LAYER_WORDS) * 4          # just past 'thick'
    n = int(np.frombuffer(raw[p:p + 4], '>i4')[0]) if p + 4 <= len(raw) else 0
    if 0 < n < 256:
        p += 4
        if (p + 4 <= len(raw)
                and int(np.frombuffer(raw[p:p + 4], '>i4')[0]) == n):
            p += 4                              # doubled-length form
        p += (n + 3) // 4 * 4                   # the padded bytes
    else:
        p += 4                                  # empty string

    need = 15
    k = p // 4
    if k + need > len(v4):
        return {}
    f = [float(x) for x in v4[k:k + need]]
    out = dict(diameter=f[0], density=f[1], distance=f[2], source=f[3],
               gamma_factor=f[4], w0=f[5], w1=f[6], resolution=f[7],
               aeff=f[8], beff=f[9],
               tail=dict(F=f[10], B=f[11], amp=f[12], L=f[13], S=f[14]))
    # Sanity: a detector density and a tail amplitude have known ranges. If
    # they are wrong the string skip was wrong, and silently returning
    # nonsense here would poison every lineshape.
    if not (0.1 < out['density'] < 25.0 and 0.0 <= out['tail']['amp'] < 1.0):
        return {}
    return out


def is_aperture(layer, opaque_mg_cm2=100.0):
    """True for a layer thick enough that nothing in the PIXE range survives.

    A collimator is recorded among the absorbers but is not in the photon
    path - it defines the solid angle. The threshold sits at 100 mg/cm2 to
    separate Canberra-34's 191.6 mg/cm2 (710 um) aluminium collimator from a
    genuine 200 um aluminium filter at 54.9 mg/cm2. Thickness alone is a weak
    discriminator, which is why this is applied only to a detector's own
    absorber list and never to filters the caller passed in.
    """
    return float(layer.get('thick', 0.0)) >= opaque_mg_cm2


def efficiency_curve(det, db, energies, mac='henke1993', filters=(),
                     drop_apertures=True):
    """Intrinsic efficiency: what fraction of photons at each energy is both
    transmitted to the crystal and absorbed in it.

        eff(E) = product(exp(-mu_i rho t_i)) x (1 - exp(-mu_cryst t_cryst))

    filters: extra layers, for anything outside the detector such as the
    200 um Al the donut2x example ran behind.
    """
    E = np.asarray(energies, float)
    eff = np.ones_like(E)
    used, skipped = [], []

    # Aperture rejection applies ONLY to the detector's own absorber list.
    # Anything in `filters` was passed deliberately by the caller and is in
    # the beam by construction - the 200 um Al of the donut2x example is
    # 54.9 mg/cm2, thick enough to trip any threshold that also catches the
    # 191.6 mg/cm2 collimator, so guessing from thickness alone cannot work.
    for lay in list(det.get('absorbers') or []):
        if drop_apertures and is_aperture(lay):
            skipped.append(lay)
            continue
        used.append(lay)
        mu = np.array([db.mu_compound(lay['Z'], lay['F'], e, mac) or 0.0
                       for e in E])
        eff *= np.exp(-mu * lay['thick'] * 1e-3)

    for lay in list(filters):
        used.append(lay)
        mu = np.array([db.mu_compound(lay['Z'], lay['F'], e, mac) or 0.0
                       for e in E])
        eff *= np.exp(-mu * lay['thick'] * 1e-3)

    cry = det.get('crystal')
    if cry:
        mu = np.array([_mu(db, cry['Z'], cry['F'], e, mac) for e in E])
        # A missing MAC returns NaN rather than 0. Zero would silently mean
        # 'the crystal absorbs nothing', i.e. zero efficiency, which looks
        # like a real answer; NaN says the table does not reach this energy.
        eff = eff * (1.0 - np.exp(-mu * cry['thick'] * 1e-3))

    return eff, used, skipped


def read_filter(path):
    """A GeoPIXE .filter file, as a list of layers - same block layout."""
    det = read_detector(path)
    lays = list(det.get('absorbers') or [])
    if det.get('crystal'):
        lays.append(det['crystal'])
    return lays


def describe(det):
    rows = ['%s  version %d' % (os.path.basename(det['path']), det['version'])]
    for lay in det.get('absorbers') or []:
        rows.append('   absorber Z=%-12s thick %10.6g mg/cm2%s'
                    % (','.join(str(z) for z in lay['Z']), lay['thick'],
                       '   APERTURE, not in the photon path'
                       if is_aperture(lay) else ''))
    c = det.get('crystal')
    if c:
        rows.append('   crystal  Z=%-12s thick %10.6g mg/cm2'
                    % (','.join(str(z) for z in c['Z']), c['thick']))
    if det.get('tail'):
        t = det['tail']
        rows.append('   density %.4g g/cm3, gamma %.4g, resolution %.4g keV'
                    % (det.get('density', 0), det.get('gamma_factor', 0),
                       det.get('resolution', 0)))
        rows.append('   tail F %.4g um, B %.4g um, amp %.4g, L %.4g, S %.4g'
                    % (t['F'], t['B'], t['amp'], t['L'], t['S']))
    return '\n'.join(rows)
