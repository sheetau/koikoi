import itertools

import numpy as np
import pytest

from scripts.diagnose_categorical_sensitivity import envelope


def test_binary_and_uniform_closed_forms():
    p = np.array([0, 0.01, 0.2, 0.5, 0.99, 1])
    np.testing.assert_allclose(
        envelope(np.stack([p, 1 - p], axis=1)),
        2 * np.sqrt(2) * p * (1 - p),
        atol=1e-15,
    )
    assert envelope(np.ones((1, 1)))[0] == 0
    for count in (2, 4, 6, 8):
        np.testing.assert_allclose(
            envelope(np.full((1, count), 1 / count)), 1 / np.sqrt(count)
        )


def test_envelope_matches_direct_softmax_finite_differences():
    probabilities = np.array([[0.2, 0.3, 0.5], [0.01, 0.98, 0.01]])
    epsilon = 1e-5
    for p in probabilities:
        logits = np.log(p)
        norms = []
        for q in itertools.product((-1, 0, 1), repeat=len(p)):
            gradient = []
            for a in range(len(p)):
                values = []
                for direction in (-1, 1):
                    z = logits.copy()
                    z[a] += direction * epsilon
                    probability = np.exp(z) / np.exp(z).sum()
                    values.append(sum(float(x) * y for x, y in zip(probability, q)))
                gradient.append((values[1] - values[0]) / (2 * epsilon))
            norms.append(np.linalg.norm(gradient))
        np.testing.assert_allclose(envelope(p[None])[0], max(norms), atol=2e-11)


def test_invalid_distribution_rejected():
    for invalid in (np.zeros((1, 2)), [[-0.1, 1.1]], [[np.nan]], np.ones((1, 9)) / 9):
        with pytest.raises(ValueError):
            envelope(invalid)
