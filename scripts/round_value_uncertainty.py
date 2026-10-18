"""Bootstrap whole shared-deal columns, retaining context/seat/model coupling."""

import numpy as np


def deal_weights(samples, replicates, rng):
    if (
        type(samples) is not int
        or type(replicates) is not int
        or min(samples, replicates) < 2
    ):
        raise ValueError("Use at least two deals and bootstrap replicates")
    return rng.multinomial(samples, np.full(samples, 1 / samples), size=(8, replicates))


def bootstrap_values(points0, dealers, terminal, weights):
    points0, dealers, terminal, weights = map(
        np.asarray, (points0, dealers, terminal, weights)
    )
    if (
        points0.ndim != 6
        or points0.shape[:-1] != (8, 2, 2, 59, 2)
        or points0.shape[-1] < 2
    ):
        raise ValueError("Expected complete sampled public transition kernels")
    if dealers.shape != points0.shape or terminal.shape != points0.shape:
        raise ValueError("Endpoint shapes differ")
    if (
        points0.dtype.kind not in "iu"
        or dealers.dtype.kind not in "iu"
        or terminal.dtype != bool
    ):
        raise ValueError("Invalid endpoint types")
    n = points0.shape[-1]
    if (
        weights.ndim != 3
        or weights.shape[0] != 8
        or weights.shape[2] != n
        or weights.shape[1] < 2
    ):
        raise ValueError("Expected round x replicate x shared-deal weights")
    if (
        weights.dtype.kind not in "iu"
        or (weights < 0).any()
        or not (weights.sum(-1) == n).all()
    ):
        raise ValueError(
            "Bootstrap weights must be nonnegative counts summing to sample size"
        )
    if not terminal[7].all() or not np.isin(dealers, (0, 1)).all():
        raise ValueError("Invalid final round or dealer")
    if not ((points0[~terminal] >= 1) & (points0[~terminal] <= 59)).all():
        raise ValueError("Invalid live point totals")
    b = weights.shape[1]
    result = np.zeros((b, 8, 2, 2, 59, 2), dtype=np.float64)
    for r in range(7, -1, -1):
        for seat in (0, 1):
            for model in (0, 1):
                key = (r, seat, model)
                p = points0[key].reshape(118, n)
                d = dealers[key].reshape(118, n)
                done = terminal[key].reshape(118, n)
                utility = np.sign(p.astype(np.int64) - 30) * (1 - 2 * seat)
                after = (
                    result[:, r + 1, seat, model].reshape(b, 118)
                    if r < 7
                    else np.zeros((b, 118))
                )
                indices = (p.clip(1, 59) - 1) * 2 + d
                targets = np.where(done[None], utility[None], after[:, indices])
                result[:, r, seat, model] = (
                    np.einsum("bjn,bn->bj", targets, weights[r], optimize=True) / n
                ).reshape(b, 59, 2)
    return result
