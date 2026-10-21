"""DA weights must not depend on how one pure spectrum happens to be scaled."""
import numpy as np
from scrappyfit.io.gpda_write import build_matrix


def test_weights_invariant_to_one_huge_row():
    rng = np.random.default_rng(0); ch = np.arange(400.0)
    A = np.array([np.exp(-0.5 * ((ch - c) / 12.0) ** 2) for c in (100, 140, 250)])
    A /= A.sum(axis=1)[:, None]
    bg = np.exp(-ch / 300.0)
    M1 = build_matrix(np.vstack([A, bg]), ['a', 'b', 'c', 'bg'])
    M2 = build_matrix(np.vstack([A, bg * 1e9]), ['a', 'b', 'c', 'bg'])
    spec = 500 * A[0] + 300 * A[1] + 50 * A[2] + 2.0 * bg + rng.normal(0, 0.01, 400)
    r1, r2 = M1[:3] @ spec, M2[:3] @ spec
    assert np.allclose(r1, [500, 300, 50], rtol=0.02) and np.allclose(r1, r2, rtol=1e-6)
