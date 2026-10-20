from fractions import Fraction as F

from scripts.check_policy_coupling import (
    marginal,
    path_law,
    shared_law,
    verify,
)


def test_fresh_common_innovations_preserve_variable_length_path_laws():
    law = shared_law()
    assert sum(law.values()) == 1
    for branch in (0, 1):
        assert marginal(law, branch) == path_law(branch)
    # Independent hand-derived first branch probabilities.
    assert marginal(law, 0)[(0,)] == F(7, 8)
    assert marginal(law, 0)[(1, 0, 0)] == F(21, 512)
    assert {len(path) for pair in law for path in pair} == {1, 2, 3}


def test_reusing_one_uniform_within_a_branch_changes_the_policy_law():
    law = shared_law(reuse=True)
    assert marginal(law, 0).get((1, 0, 0), 0) == 0
    assert marginal(law, 0) != path_law(0)


def test_unbiasedness_does_not_imply_lower_variance():
    report = verify()
    assert report["coupling_variance_changes_have_opposite_signs"]
