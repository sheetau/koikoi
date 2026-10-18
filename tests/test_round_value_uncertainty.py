import numpy as np
import pytest

from scripts.round_transition_values import backward_values
from scripts.round_value_uncertainty import bootstrap_values, deal_weights


def test_shared_deal_bootstrap_matches_explicit_resampled_kernels():
    rng = np.random.default_rng(40130001)
    shape = (8, 2, 2, 59, 2, 5)
    # Different point/dealer/model/seat transitions prevent an accidental symmetry test.
    points = rng.integers(-2, 63, size=shape, dtype=np.int32)
    dealers = rng.integers(0, 2, size=shape, dtype=np.int32)
    terminal = (points <= 0) | (points >= 60)
    terminal[7] = True
    weights = deal_weights(5, 7, rng)
    weights[:, 0] = 1
    actual = bootstrap_values(points, dealers, terminal, weights)
    for b in range(7):
        sampled = [np.empty_like(a) for a in (points, dealers, terminal)]
        for r in range(8):
            indices = np.repeat(np.arange(5), weights[r, b])
            for dst, source in zip(sampled, (points, dealers, terminal)):
                dst[r] = source[r][..., indices]
        expected = backward_values(*sampled)
        np.testing.assert_allclose(actual[b], expected, rtol=0, atol=1e-15)
    assert actual.shape == (7, 8, 2, 2, 59, 2)
    assert (np.abs(actual) <= 1).all() and not np.array_equal(actual[0], actual[1])
    bad = weights.copy()
    bad[0, 0, 0] = 0
    with pytest.raises(ValueError, match="weights"):
        bootstrap_values(points, dealers, terminal, bad)
