"""Fixed four-contrast duration criterion and complete secondary reporting."""

import numpy as np

from scripts.directional_return_statistics import simultaneous_bands

POLICIES = ("parent", "short", "long")
OPPONENTS = ("wp", "point")


def analyse(points, seed=99410001, replicates=10000):
    points = np.asarray(points)
    if (
        points.ndim != 5
        or points.shape[:3] != (2, 3, 2)
        or points.shape[-1] != 2
        or points.shape[3] < 2
        or points.dtype.kind != "i"
    ):
        raise ValueError(
            "Expected signed integer parent,policy,opponent,pair,seat margins"
        )
    points = points.astype(np.int64)
    win = ((np.sign(points) + 1) / 2).mean(-1)
    margins = points.mean(-1)
    columns, labels, primary = [], [], []
    for parent in range(2):
        for scope in ("mixture", *OPPONENTS):
            scores = (
                win[parent].mean(1)
                if scope == "mixture"
                else win[parent, :, OPPONENTS.index(scope)]
            )
            values = (
                margins[parent].mean(1)
                if scope == "mixture"
                else margins[parent, :, OPPONENTS.index(scope)]
            )
            for policy, name in enumerate(POLICIES):
                for metric, data in (("win", scores), ("points", values)):
                    labels.append(f"parent{parent + 1}/{scope}/{name}/{metric}")
                    columns.append(data[policy])
            for left, right in ((2, 1), (2, 0), (1, 0)):
                for metric, data in (("win", scores), ("points", values)):
                    if scope == "mixture" and left == 2 and metric == "win":
                        primary.append(len(columns))
                    labels.append(
                        f"parent{parent + 1}/{scope}/{POLICIES[left]}-{POLICIES[right]}/{metric}"
                    )
                    columns.append(data[left] - data[right])
    data = np.stack(columns, axis=1)
    bands = simultaneous_bands(data[:, primary], replicates=replicates, seed=seed)
    bands["rows"] = {labels[i]: row for i, row in zip(primary, bands["rows"])}
    assert len(primary) == 4 and data.shape[1] == 72
    # Every secondary marginal interval uses the same whole compound blocks.
    rng = np.random.default_rng(seed)
    means = []
    n = len(data)
    for start in range(0, replicates, 100):
        counts = rng.multinomial(
            n, np.full(n, 1 / n), size=min(100, replicates - start)
        )
        means.append(counts @ data / n)
    lower, upper = np.quantile(np.concatenate(means), [0.025, 0.975], axis=0)
    rows = {
        label: {"mean": float(mean), "marginal95": [float(lo), float(hi)]}
        for label, mean, lo, hi in zip(labels, data.mean(0), lower, upper)
    }
    return {
        "rows": rows,
        "primary": bands,
        "qualified": all(row["sign"] == 1 for row in bands["rows"].values()),
        "compound_blocks": n,
        "distinct_initial_deals": 2 * n,
        "full_matches": int(points.size),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "secondary_scope": "72 descriptive absolute/contrast win/point means with marginal percentile intervals; only the four simultaneous primary bands determine qualification.",
    }
