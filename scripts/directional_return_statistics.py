"""Whole-deal simultaneous bootstrap bands for the fixed directional probe."""

import numpy as np

from scripts.directional_return_probe import PROBES

ESTIMATORS = ("mc", "gae95_full", "gae95_window", "gae1_window")


def coordinates(contributions):
    """Input: parent,deal,estimator,seat,probe; preserve shared deals."""
    x = np.asarray(contributions, dtype=float)
    if x.ndim != 5 or x.shape[0] != 2 or x.shape[2:] != (4, 2, 5):
        raise ValueError(
            "Expected two parents, four estimators, two seats, five probes"
        )
    if not np.isfinite(x).all() or x.shape[1] < 2:
        raise ValueError("Need at least two complete finite deal blocks")
    labels, columns = [], []
    for parent in range(2):
        seat_mean = x[parent].mean(axis=2)
        for probe, name in enumerate(PROBES):
            for estimator, label in enumerate(ESTIMATORS):
                labels.append(f"parent{parent + 1}/{name}/{label}")
                columns.append(seat_mean[:, estimator, probe])
            for estimator, label in enumerate(ESTIMATORS[1:], 1):
                labels.append(f"parent{parent + 1}/{name}/{label}-mc")
                columns.append(seat_mean[:, estimator, probe] - seat_mean[:, 0, probe])
    return labels, np.stack(columns, axis=1)


def simultaneous_bands(data, replicates=10000, seed=98410001):
    """Fixed-SE max-deviation bootstrap, all columns in one 95% family.

    A degenerate empirical coordinate gets no sign claim: its interval is null.
    These are asymptotic resampling bands, not finite-sample guarantees.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or len(x) < 2:
        raise ValueError("Expected at least two deal rows")
    if not np.isfinite(x).all() or type(replicates) is not int or replicates < 2:
        raise ValueError("Invalid bootstrap input")
    n = len(x)
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(n)
    active = se > 0
    centered = x[:, active] - mean[active]
    rng = np.random.default_rng(seed)
    maxima = []
    for first in range(0, replicates, 100):
        count = rng.multinomial(n, np.full(n, 1 / n), size=min(100, replicates - first))
        if active.any():
            dev = count @ centered / n / se[active]
            maxima.extend(np.max(np.abs(dev), axis=1).tolist())
        else:
            maxima.extend([0.0] * len(count))
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    rows = []
    for mu, error, enabled in zip(mean, se, active):
        lower, upper = (float(mu - critical * error), float(mu + critical * error))
        rows.append(
            {
                "mean": float(mu),
                "standard_error": float(error),
                "simultaneous95": [lower, upper] if enabled else [None, None],
                "sign": (1 if lower > 0 else -1 if upper < 0 else 0) if enabled else 0,
            }
        )
    return {
        "deals": n,
        "replicates": replicates,
        "seed": seed,
        "family_size": x.shape[1],
        "nondegenerate": int(active.sum()),
        "critical": critical,
        "rows": rows,
    }


def summarize(contributions):
    labels, data = coordinates(contributions)
    result = simultaneous_bands(data)
    result["rows"] = dict(zip(labels, result["rows"]))
    reversal = []
    for parent in (1, 2):
        prefix = f"parent{parent}/koi_continue/"
        left = result["rows"][prefix + "mc"]["sign"]
        right = result["rows"][prefix + "gae95_window"]["sign"]
        reversal.append(left * right == -1)
    result["primary_reversal_by_parent"] = reversal
    result["primary_replicated_reversal"] = all(reversal)
    return result
