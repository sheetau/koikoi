"""Independent exhaustive two-attempt experiment for the MSE calculation."""

from fractions import Fraction
from itertools import combinations

import numpy as np

from scripts.benchmark_mixed_history_sampler import expected_errors


def test_expected_error_matches_all_two_attempt_paths():
    pool = tuple(range(5))
    hands = tuple(combinations(pool, 2))
    support = ((0, 1), (0, 2), (3, 4))
    target = [sum(c in h for h in support) / Fraction(3) for c in pool]

    def error(samples):
        return (
            sum(
                (sum(c in h for h in samples) / Fraction(len(samples)) - target[c]) ** 2
                for c in pool
            )
            / 5
        )

    iid = mixed = success = Fraction(0)
    for first in hands:
        for second in hands:
            accepted = [h for h in (first, second) if h in support]
            if accepted:
                iid += error(accepted) / 100
                success += Fraction(1, 100)
            if first not in support and second in support:
                mixed += error([second]) / 100
            elif first in support:
                current = second if second in support else first
                mixed += Fraction(1, 5) * error([first, current]) / 100
        if first in support:
            for outgoing in first:
                for incoming in set(pool) - set(first):
                    candidate = tuple(sorted((set(first) - {outgoing}) | {incoming}))
                    current = candidate if candidate in support else first
                    mixed += Fraction(4, 5) * error([first, current]) / (10 * 6)
    result = expected_errors(pool, 2, support, budget=2)
    np.testing.assert_allclose(
        result["initialization_failure_probability"], float(1 - success), atol=1e-14
    )
    np.testing.assert_allclose(
        result["independent_conditional_mse"], float(iid / success), atol=1e-14
    )
    np.testing.assert_allclose(
        result["mixed_conditional_mse"], float(mixed / success), atol=1e-14
    )
