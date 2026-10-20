"""Exact conservative lower bounds for nonnegative conditional improvements.

Inputs must be IID prefixes from the law used by the exact value evaluator.
Thresholds, sample size and the family error allocation must precede sampling.
This module neither establishes those assumptions nor defines a study.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise


def rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("Use exact integer or Fraction inputs")
    return Fraction(value)


def binomial_tail(n, k, probability):
    """Return P[Binomial(n, probability) >= k] with exact integer arithmetic."""
    if type(n) is not int or n < 1 or type(k) is not int or not 0 <= k <= n:
        raise ValueError("Expected n >= 1 and 0 <= k <= n")
    p = rational(probability)
    if not 0 <= p <= 1:
        raise ValueError("Probability outside [0, 1]")
    if k == 0 or p == 1:
        return Fraction(1)
    if p == 0:
        return Fraction(0)
    a, b = p.numerator, p.denominator
    term, total = (b - a) ** n, 0
    for j in range(n + 1):
        if j >= k:
            total += term
        if j < n:
            term = term * (n - j) * a // ((j + 1) * (b - a))
    return Fraction(total, b**n)


def binomial_lower(n, k, alpha, bits=32):
    """One-sided Clopper--Pearson lower limit rounded down to a dyadic grid."""
    error = rational(alpha)
    if not 0 < error < 1:
        raise ValueError("Error probability must lie strictly between 0 and 1")
    if type(bits) is not int or bits < 1:
        raise ValueError("Expected positive integer precision")
    # Validate even the zero-success case.
    binomial_tail(n, k, Fraction(0))
    if k == 0:
        return Fraction(0)
    denominator = 1 << bits
    low, high = 0, denominator
    while high - low > 1:
        mid = (low + high) // 2
        if binomial_tail(n, k, Fraction(mid, denominator)) <= error:
            low = mid
        else:
            high = mid
    return Fraction(low, denominator)


@dataclass(frozen=True)
class ThresholdBound:
    threshold: Fraction
    successes: int
    probability_lower: Fraction


@dataclass(frozen=True)
class GainBound:
    n: int
    alpha: Fraction
    empirical_mean: Fraction
    mean_lower: Fraction
    thresholds: tuple[ThresholdBound, ...]


def certified_mean_lower(values, thresholds, *, expected_n, alpha, bits=32):
    """Bound E[Delta] for IID exact Delta in [0, 1], including fallback zeros.

    Allocate alpha equally across the fixed thresholds. Simultaneous survival
    bounds imply E[Delta] >= sum_j (t_j - t_{j-1}) P[Delta >= t_j].
    Callers must allocate alpha across other reported populations/comparisons.
    No independence between threshold indicators is assumed.
    """
    if type(expected_n) is not int or expected_n < 1:
        raise ValueError("Declare a positive complete-prefix denominator")
    samples = tuple(rational(x) for x in values)
    grid = tuple(rational(t) for t in thresholds)
    error = rational(alpha)
    if len(samples) != expected_n:
        raise ValueError("Sample count differs from the declared denominator")
    if any(not 0 <= x <= 1 for x in samples):
        raise ValueError("Expected certified nonnegative improvements in [0, 1]")
    if not grid or any(not 0 < t <= 1 for t in grid):
        raise ValueError("Supply nonempty thresholds in (0, 1]")
    if any(a >= b for a, b in pairwise(grid)):
        raise ValueError("Thresholds must be strictly increasing")
    if not 0 < error < 1:
        raise ValueError("Error probability must lie strictly between 0 and 1")
    previous, lower, rows = Fraction(0), Fraction(0), []
    for threshold in grid:
        count = sum(x >= threshold for x in samples)
        probability = binomial_lower(expected_n, count, error / len(grid), bits)
        lower += (threshold - previous) * probability
        rows.append(ThresholdBound(threshold, count, probability))
        previous = threshold
    return GainBound(
        expected_n, error, sum(samples, Fraction(0)) / expected_n, lower, tuple(rows)
    )
