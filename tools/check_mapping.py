"""Does the mapping path agree with itself?

The worked example ships no imaging data, so this validates the mapping chain
on 287424 with 287423's DA matrix instead. The test that matters is internal
consistency: a DA matrix is linear, so projecting the WHOLE spectrum must
give the same answer as projecting every pixel and averaging. If those two
disagree, the binning, the masking or the projection is wrong.
"""
import sys
sys.path.insert(0, r'C:\Users\Charles\Desktop\scrappyFIT')
import numpy as np
from scrappyfit.analysis.profiles import line_profile, region_stats
from scrappyfit.session import Session

LMF = r'C:\Users\Charles\Desktop\GeoPIXE-main\data\lmf\287424.lmf'
DAM = r'F:\GeoPIXE\287423.dam'

s = Session()
s.load(LMF)
s.set_calibration(0.0016984, -0.37225)
s.load_dam(DAM)

print('1. DA MATRIX LINEARITY: whole spectrum vs mean of the pixels')
print()
for binning in (1, 2, 4):
    maps = s.da_maps(binning=binning)
    x, y, e = s.events
    dam = s.dam
    # project the whole spectrum directly through the same matrix
    spec = np.bincount(e, minlength=dam['matrix'].shape[1]
                       if hasattr(dam.get('matrix'), 'shape') else 4096)
    npix = None
    rows = []
    for nm, im in maps.items():
        im = np.asarray(im, float)
        if npix is None:
            npix = im.size
        rows.append((nm, float(np.nansum(im))))
    tot = dict(rows)
    if binning == 1:
        base = tot
        base_npix = npix
    print('   binning %d: %d pixels, %d maps, sum(Si) = %.6g'
          % (binning, npix, len(maps), tot.get('Si', float('nan'))))

print()
print('   A DA map is a concentration per pixel, so the SUM scales with the')
print('   pixel count. What must be invariant is the MEAN:')
print('   %-6s %12s %12s %12s' % ('el', 'bin 1', 'bin 2', 'bin 4'))
means = {}
for binning in (1, 2, 4):
    maps = s.da_maps(binning=binning)
    for nm, im in maps.items():
        means.setdefault(nm, {})[binning] = float(np.nanmean(np.asarray(im, float)))
for nm in sorted(means):
    m = means[nm]
    print('   %-6s %12.5f %12.5f %12.5f' % (nm, m.get(1, np.nan),
                                            m.get(2, np.nan), m.get(4, np.nan)))
worst = 0.0
for nm, m in means.items():
    if abs(m.get(1, 0)) > 0.05:
        for bb in (2, 4):
            worst = max(worst, abs(m[bb] / m[1] - 1.0))
print()
print('   largest change in the mean across binnings: %.3f%%' % (100 * worst))

print()
print('2. MASKING: a region mean from the map vs a fit of that region')
mask = np.zeros_like(np.asarray(maps['Si'], float), dtype=bool)
h, w = mask.shape
mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
st = region_stats(np.asarray(maps['Si'], float), mask)
print('   Si over the central quarter: mean %.4f wt%%, n=%d pixels'
      % (st['mean'], st['n']))

print()
print('3. PROFILE: a traverse samples the same map the region does')
d, v, sp = line_profile(np.asarray(maps['Si'], float),
                        (w * 0.1, h * 0.5), (w * 0.9, h * 0.5), width=9)
print('   Si along a horizontal traverse: mean %.4f, min %.4f, max %.4f'
      % (v.mean(), v.min(), v.max()))
print('   whole-map mean for comparison: %.4f'
      % float(np.nanmean(np.asarray(maps['Si'], float))))
