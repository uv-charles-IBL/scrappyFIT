"""GeoPIXE .dai reader: the image block is found by counting back from the end."""
import os

import numpy as np
import pytest

from scrappyfit.io.gpdai import read_dai

DAI = r'C:\Users\Charles\Desktop\GeoPIXE-main\data\Topaz\287427.dai'


@pytest.mark.skipif(not os.path.exists(DAI), reason='sample .dai not present')
def test_287427_layout():
    d = read_dai(DAI)
    assert d['version'] == -56
    assert (d['xsize'], d['ysize']) == (256, 256)
    assert d['names'][0] == 'Back' and d['names'][-1] == 'sum'
    assert len(d['names']) == 19
    # 40,103 floats of header and flux maps precede the images; the first
    # reader guessed byte 596 and got half of one element glued to half of
    # the next
    assert d['image_offset'] == 161008
    assert d['has_errors']
    si, o = d['images']['Si'], d['images']['O']
    assert si.shape == (256, 256)
    # quartz: oxygen and silicon are the whole field and neither map is
    # split into a bright half and a dark half
    # (the wrong decode gave a ratio of 0.005; a real gradient gives 0.87)
    assert abs(si[:128].mean() / si[128:].mean() - 1) < 0.3
    assert abs(o[:128].mean() / o[128:].mean() - 1) < 0.3
    assert d['dam'].lower().endswith('.dam')
    assert d['source'].lower().endswith('287427.lmf')
