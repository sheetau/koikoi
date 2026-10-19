"""Paired study analysis with fixed criteria for koi-temperature training."""

import numpy as np

from koikoi.evaluate import paired_interval

POLICIES = ("parent", "fixed", "calibrated")
OPPONENTS = ("wp", "point")


def cases(parent, opponent, pairs=2048, seed_base=97110001):
    if parent not in (1, 2) or opponent not in OPPONENTS or pairs < 1:
        raise ValueError("Invalid comparison cell")
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "replicate": rep,
            "seat": seat,
            "seed": seed_base + 10000 * OPPONENTS.index(opponent) + rep,
        }
        for rep in range(pairs)
        for seat in (0, 1)
    ]


def analyse(points, bootstrap_seed=97310001, bootstrap_replicates=10000):
    points = np.asarray(points)
    if (
        points.ndim != 5
        or points.shape[:3] != (2, 3, 2)
        or points.shape[-1] != 2
        or points.shape[3] < 2
    ):
        raise ValueError("Expected parent,policy,opponent,pair,seat axes")
    if points.dtype.kind not in "iu":
        raise ValueError("Integer candidate-relative point margins required")
    win = ((np.sign(points) + 1) / 2).mean(axis=-1)
    margin = points.mean(axis=-1)
    absolute, contrasts = [], []
    for parent in range(2):
        for scope in ("mixture", *OPPONENTS):
            scores = (
                win[parent].mean(1)
                if scope == "mixture"
                else win[parent, :, OPPONENTS.index(scope)]
            )
            values = (
                margin[parent].mean(1)
                if scope == "mixture"
                else margin[parent, :, OPPONENTS.index(scope)]
            )
            for index, policy in enumerate(POLICIES):
                absolute.append(
                    {
                        "parent": parent + 1,
                        "policy": policy,
                        "scope": scope,
                        "win": float(scores[index].mean()),
                        "points": float(values[index].mean()),
                        "win_interval": paired_interval(
                            scores[index], bootstrap_seed, bootstrap_replicates
                        ),
                        "point_interval": paired_interval(
                            values[index], bootstrap_seed, bootstrap_replicates
                        ),
                    }
                )
            for left, right in (
                ("calibrated", "fixed"),
                ("calibrated", "parent"),
                ("fixed", "parent"),
            ):
                i, j = POLICIES.index(left), POLICIES.index(right)
                delta_win, delta_points = scores[i] - scores[j], values[i] - values[j]
                contrasts.append(
                    {
                        "parent": parent + 1,
                        "left": left,
                        "right": right,
                        "scope": scope,
                        "win_change": float(delta_win.mean()),
                        "point_change": float(delta_points.mean()),
                        "win_interval": paired_interval(
                            delta_win, bootstrap_seed, bootstrap_replicates
                        ),
                        "point_interval": paired_interval(
                            delta_points, bootstrap_seed, bootstrap_replicates
                        ),
                    }
                )
    primary = [
        r for r in contrasts if r["scope"] == "mixture" and r["left"] == "calibrated"
    ]
    assert len(primary) == 4
    return {
        "absolute": absolute,
        "contrasts": contrasts,
        "replicated_strength_gain": all(r["win_interval"][0] > 0 for r in primary),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "scope": "Development comparison; paired blocks retain both seats and both fixed opponent strata.",
    }
