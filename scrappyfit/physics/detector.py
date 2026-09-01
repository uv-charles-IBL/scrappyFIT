"""Detector efficiency, read from a curve exported once from GeoPIXE.

GeoPIXE's .detector files are IDL XDR structs. Rather than reimplement that
serialisation, the efficiency curve is exported once from GeoPIXE as ASCII
(energy keV, efficiency) and interpolated here. That keeps the instrument
response identical to GeoPIXE's own while staying licence-free afterwards.

Re-export with the probe_eff routine in the scratchpad if the detector or
the MAC dataset changes - efficiency depends on both.
"""
import math, os


class Efficiency:
    def __init__(self, path):
        self.E, self.eff = [], []
        for line in open(path, errors='ignore'):
            t = line.split()
            if len(t) == 2 and not line.startswith('#'):
                try:
                    self.E.append(float(t[0])); self.eff.append(float(t[1]))
                except ValueError:
                    pass
        self.name = os.path.basename(path)

    def __call__(self, E_keV):
        E, y = self.E, self.eff
        if E_keV <= E[0]:
            return y[0]
        if E_keV >= E[-1]:
            return y[-1]
        for i in range(1, len(E)):
            if E[i] >= E_keV:
                if y[i - 1] <= 0 or y[i] <= 0:
                    return max(y[i - 1], 0.0)
                f = ((math.log(E_keV) - math.log(E[i - 1])) /
                     (math.log(E[i]) - math.log(E[i - 1])))
                return math.exp(math.log(y[i - 1]) + f *
                                (math.log(y[i]) - math.log(y[i - 1])))
        return y[-1]
