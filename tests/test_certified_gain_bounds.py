from fractions import Fraction as F
from itertools import product
from math import comb

import pytest

from scripts.certified_gain_bounds import (
    binomial_lower,
    binomial_tail,
    certified_mean_lower,
)


def pmf(n, k, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def test_tail_matches_independent_exhaustive_bernoulli_sequences():
    for n in range(1, 7):
        for p in (F(0), F(1, 7), F(1, 2), F(6, 7), F(1)):
            by_count = [F(0)] * (n + 1)
            for sequence in product((0, 1), repeat=n):
                mass = F(1)
                for outcome in sequence:
                    mass *= p if outcome else 1 - p
                by_count[sum(sequence)] += mass
            for k in range(n + 1):
                assert binomial_tail(n, k, p) == sum(by_count[k:])


def test_lower_limit_rounds_outward_including_extreme_counts():
    alpha, bits = F(1, 20), 24
    for n in (1, 7, 100):
        assert binomial_lower(n, 0, alpha, bits) == 0
        for k in {1, n // 2 + 1, n}:
            lower = binomial_lower(n, k, alpha, bits)
            assert binomial_tail(n, k, lower) <= alpha
            assert binomial_tail(n, k, lower + F(1, 2**bits)) > alpha
    assert binomial_lower(4, 4, F(1, 16), bits) == F(1, 2)


def test_exact_finite_sample_coverage_of_binomial_limits():
    alpha = F(1, 10)
    for n in range(1, 11):
        limits = [binomial_lower(n, k, alpha, 12) for k in range(n + 1)]
        for numerator in range(21):
            p = F(numerator, 20)
            failure = sum(pmf(n, k, p) for k in range(n + 1) if limits[k] > p)
            assert failure <= alpha


def test_mean_bound_exact_coverage_for_discrete_nonnegative_populations():
    # Enumerate every ordered sample, not a Monte Carlo coverage check.
    support, n, alpha = (F(0), F(1, 4), F(1)), 4, F(1, 5)
    grid = (F(1, 8), F(1, 2), F(1))
    bounds = {}
    for sample in product(range(3), repeat=n):
        values = [support[i] for i in sample]
        result = certified_mean_lower(values, grid, expected_n=n, alpha=alpha, bits=12)
        assert result.empirical_mean == sum(values) / n
        assert 0 <= result.mean_lower <= result.empirical_mean
        bounds[sample] = result.mean_lower
    for probabilities in ((F(1, 2), F(1, 3), F(1, 6)), (F(4, 5), F(1, 10), F(1, 10))):
        mean = sum(x * p for x, p in zip(support, probabilities))
        failure = F(0)
        for sample, lower in bounds.items():
            mass = F(1)
            for i in sample:
                mass *= probabilities[i]
            if lower > mean:
                failure += mass
        assert failure <= alpha


def test_complete_denominator_zero_fallback_and_threshold_equality():
    result = certified_mean_lower(
        [0, F(1, 4), F(1, 2), 1], [F(1, 4), F(1, 2), 1], expected_n=4, alpha=F(1, 20)
    )
    assert [row.successes for row in result.thresholds] == [3, 2, 1]
    assert result.empirical_mean == F(7, 16)
    assert result.mean_lower > 0
    zero = certified_mean_lower([0] * 4, [F(1, 4)], expected_n=4, alpha=F(1, 20))
    assert zero.mean_lower == zero.empirical_mean == 0
    for values, thresholds, expected_n, alpha in (
        ([0, 1], [F(1, 2)], 3, F(1, 20)),
        ([-1, 1], [F(1, 2)], 2, F(1, 20)),
        ([0, 2], [F(1, 2)], 2, F(1, 20)),
        ([0.0, 1], [F(1, 2)], 2, F(1, 20)),
        ([0, 1], [], 2, F(1, 20)),
        ([0, 1], [0], 2, F(1, 20)),
        ([0, 1], [1, F(1, 2)], 2, F(1, 20)),
        ([0, 1], [1, 1], 2, F(1, 20)),
        ([0, 1], [1], 2, 0),
    ):
        with pytest.raises(ValueError):
            certified_mean_lower(values, thresholds, expected_n=expected_n, alpha=alpha)
    for n, k, alpha, bits in (
        (0, 0, F(1, 20), 8),
        (2, 3, F(1, 20), 8),
        (2, 1, 1, 8),
        (2, 1, F(1, 20), 0),
    ):
        with pytest.raises(ValueError):
            binomial_lower(n, k, alpha, bits)
