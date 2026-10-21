"""L-shell line selection must follow GeoPIXE's list_line_index.pro:
only L-alpha is lumped below Z 52; L-beta and L-gamma are always individual."""

from scrappyfit.physics.gpdb import Database

db = Database()


def _near(lines, e, tol=0.005):
    return any(abs(x - e) <= tol and i > 0 for x, i in lines)


def test_silver_l_beta2_and_gamma1_are_at_their_own_energies():
    L = db.line_list(47, 2)
    assert _near(L, 3.347)          # Ag Lb2
    assert _near(L, 3.519)          # Ag Lg1
    assert not _near(L, 3.192)      # no lumped Lb_ centroid
    assert not _near(L, 3.609)      # no lumped Lg_ centroid


def test_l_alpha_still_lumped_below_split_and_split_above():
    assert _near(db.line_list(47, 2), 2.983)        # Ag La_ group
    I = db.line_list(53, 2)                         # iodine: La1 + La2
    assert _near(I, 3.938) and _near(I, 3.926)


def test_intensities_sum_to_one():
    for Z in (35, 47, 49, 50, 53, 82):
        L = db.line_list(Z, 2)
        assert abs(sum(i for _, i in L) - 1.0) < 1e-9
