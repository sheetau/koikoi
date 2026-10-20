"""Separate scalar reconstruction of all score-auxiliary bootstrap coordinates."""

import numpy as np


def check_statistics(points, summary, seed, replicates):
    """Explicit seat/opponent sums, separate quantiles and bootstrap chunking."""
    columns = {}
    for parent in (1, 2):
        for scope, opponents in (("mixture", (0, 1)), ("wp", (0,)), ("point", (1,))):
            for policy, name in enumerate(("parent", "none", "win", "score")):
                wins, margins = [], []
                for deal in range(points.shape[3]):
                    w, p = 0.0, 0.0
                    for opponent in opponents:
                        for seat in (0, 1):
                            value = int(
                                points[parent - 1, policy, opponent, deal, seat]
                            )
                            w += 1 if value > 0 else 0.5 if value == 0 else 0
                            p += value
                    wins.append(w / (2 * len(opponents)))
                    margins.append(p / (2 * len(opponents)))
                prefix = f"seed{parent}/{scope}/{name}/"
                columns[prefix + "win"] = np.array(wins)
                columns[prefix + "points"] = np.array(margins)
            for left, right in (
                ("score", "none"),
                ("score", "win"),
                ("score", "parent"),
                ("win", "none"),
                ("win", "parent"),
                ("none", "parent"),
            ):
                for metric in ("win", "points"):
                    prefix = f"seed{parent}/{scope}/"
                    columns[prefix + left + "-" + right + "/" + metric] = (
                        columns[prefix + left + "/" + metric]
                        - columns[prefix + right + "/" + metric]
                    )
    assert set(columns) == set(summary["rows"])
    labels = list(columns)
    values = np.stack(list(columns.values()), axis=1)
    n = len(values)
    primary_labels = [
        f"seed{p}/mixture/score-{other}/win"
        for p in (1, 2)
        for other in ("none", "win", "parent")
    ]
    assert set(primary_labels) == set(summary["primary"]["rows"])
    assert summary["primary"]["deals"] == points.shape[3]
    assert summary["primary"]["seed"] == seed
    assert summary["primary"]["replicates"] == replicates
    indices = [labels.index(label) for label in primary_labels]
    mean = values.sum(0) / n
    se = np.sqrt(np.square(values[:, indices] - mean[indices]).sum(0) / (n - 1) / n)
    rng = np.random.default_rng(seed)
    samples, maxdev = [], []
    active = se > 0
    for first in range(0, replicates, 37):
        count = rng.multinomial(n, [1 / n] * n, size=min(37, replicates - first))
        sample = np.einsum("bi,ij->bj", count, values) / n
        samples.extend(sample)
        if active.any():
            deviation = abs(
                (sample[:, indices][:, active] - mean[indices][active]) / se[active]
            )
            maxdev.extend(deviation.max(1))
        else:
            maxdev.extend([0.0] * len(sample))
    critical = sorted(maxdev)[int(np.ceil(0.95 * (replicates - 1)))]
    np.testing.assert_allclose(
        critical, summary["primary"]["critical"], atol=2e-12, rtol=0
    )
    ordered = np.sort(np.asarray(samples), axis=0)
    intervals = []
    for quantile in (0.025, 0.975):
        index = quantile * (replicates - 1)
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        intervals.append(
            ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
            if hi != lo
            else ordered[lo]
        )
    for col, label in enumerate(labels):
        row = summary["rows"][label]
        np.testing.assert_allclose(row["mean"], mean[col], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            row["marginal95"], [x[col] for x in intervals], atol=1e-12, rtol=0
        )
    signs = []
    for col, label in enumerate(primary_labels):
        row = summary["primary"]["rows"][label]
        np.testing.assert_allclose(row["mean"], mean[indices[col]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["standard_error"], se[col], atol=1e-12, rtol=0)
        expected = mean[indices[col]] + np.array([-1, 1]) * critical * se[col]
        if not active[col]:
            assert row["simultaneous95"] == [None, None]
            sign = 0
        else:
            np.testing.assert_allclose(
                row["simultaneous95"], expected, atol=2e-12, rtol=0
            )
            sign = 1 if expected[0] > 0 else -1 if expected[1] < 0 else 0
        assert row["sign"] == sign
        signs.append(sign)
    assert summary["qualified"] == all(sign == 1 for sign in signs)
    assert summary["full_matches"] == points.size
    assert summary["compound_blocks"] == points.shape[3]
    assert summary["distinct_initial_deals"] == 2 * points.shape[3]
    assert (
        summary["bootstrap_seed"] == seed
        and summary["bootstrap_replicates"] == replicates
    )
    assert summary["primary"]["family_size"] == 6
    assert summary["primary"]["nondegenerate"] == int(active.sum())
