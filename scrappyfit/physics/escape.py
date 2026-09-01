"""
Detector escape peaks, ported from escape_fraction.pro / escape_energy.pro.

When a characteristic X-ray is absorbed in the crystal, the crystal's own K
X-ray can escape, leaving a peak at E - E_escape. On a Ge detector that is a
9.886 keV displacement, which lands escape peaks from strong lines directly on
top of other elements' peaks. Not modelling them makes the fit redistribute
that intensity into whatever real element happens to sit there.

    E_escape        Ge  Ka1 = 9.886 keV   (Kb1 for the beta escape)
                    Si  Ka_ = 1.740 keV   (no beta escape)

    EF = (1 - MU_DET*ln(1 + MU_E/MU_DET)/MU_E) / tanh(0.5*MU_E*D)
    fraction = detector.gamma * EF * ba

    MU_DET  mass attenuation of the crystal at the escape energy
    MU_E    mass attenuation of the crystal at the line energy
    D       crystal thickness / cos(tilt), in g/cm^2
    ba      1.0 for alpha escape; 0.133 for Ge beta escape, 0 otherwise

Only lines above the crystal's K edge produce escapes, which is why the
fraction is zero below ~11.1 keV on Ge and ~1.84 keV on Si.
"""

import numpy as np


class EscapeModel:
    def __init__(self, db, crystal_Z, crystal_thick_mgcm2, gamma=1.0,
                 mac_dataset=None):
        """crystal_thick_mgcm2 is detector.crystal.thick, as GeoPIXE stores it.
        gamma is detector.GAMMA, the escape scaling factor."""
        self.db = db
        self.Z = int(crystal_Z)
        self.thick = float(crystal_thick_mgcm2)
        self.gamma = float(gamma)
        self.mac = mac_dataset

        if self.Z == 32:                      # Ge
            self.e_alpha = db.lineE.get(32, {}).get('Ka1', 9.886)
            self.e_beta = db.lineE.get(32, {}).get('Kb1', 10.982)
            self.ba_beta = 0.133
        elif self.Z == 14:                    # Si
            self.e_alpha = db.lineE.get(14, {}).get('Ka_', 1.740)
            self.e_beta = 0.0
            self.ba_beta = 0.0
        else:
            self.e_alpha = 0.0
            self.e_beta = 0.0
            self.ba_beta = 0.0

        self.k_edge = db.edge.get((self.Z, 'K'), 0.0)

    def _mu(self, E):
        if self.mac:
            m = self.db.mu(self.Z, E, self.mac)
            if m is not None:
                return m
        # fall back to the Elam photo table via the tabulated set
        m = self.db.mu(self.Z, E, 'henke1993')
        return m if m is not None else 0.0

    def fraction(self, E, beta=False, tilt_deg=0.0):
        """Escape fraction for a line at energy E (keV)."""
        e_loss = self.e_beta if beta else self.e_alpha
        if e_loss < 0.01:
            return 0.0
        if not (self.k_edge < E < 100.0):
            return 0.0
        ba = self.ba_beta if beta else 1.0
        if ba <= 0.0:
            return 0.0

        ct = np.cos(np.radians(tilt_deg))
        mu_det = self._mu(e_loss)
        mu_e = self._mu(E)
        if mu_det <= 0 or mu_e <= 0:
            return 0.0

        D = self.thick * 0.001 / ct          # mg/cm^2 -> g/cm^2
        ef = ((1.0 - mu_det * np.log(1.0 + mu_e / mu_det) / mu_e)
              / np.tanh(0.5 * mu_e * D))
        return float(self.gamma * ef * ba)

    def escape_lines(self, lines, tilt_deg=0.0, min_frac=1e-5):
        """Given [(E, intensity), ...] return the escape lines they produce.

        Escape lines belong to the SAME fitted component as their parent, with
        intensity scaled by the escape fraction - they are not free parameters.
        Tying them is what makes them useful: they constrain the parent's area
        rather than adding another degree of freedom.
        """
        out = []
        for E, inten in lines:
            for beta in (False, True):
                f = self.fraction(E, beta=beta, tilt_deg=tilt_deg)
                if f <= min_frac:
                    continue
                e_loss = self.e_beta if beta else self.e_alpha
                e_esc = E - e_loss
                if e_esc > 0.05:
                    out.append((e_esc, inten * f))
        return out
