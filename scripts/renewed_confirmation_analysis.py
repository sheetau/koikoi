"""Complete-grid paired analysis for the prospective three-policy confirmation.

Pure record/statistics helpers. No policies are loaded and no deals are played.
"""

import json
from pathlib import Path

import numpy as np

from scripts.renewed_continuation_analysis import bootstrap_intervals

POLICIES = ("parent", "fixed_public", "renewed_public")
PAIRS = 4096
DEAL_START = 101100001
BOOTSTRAP_SEED = 101300001
BOOTSTRAP_DRAWS = 10000
SCOPES = (("mixture", (0, 1)), ("wp", (0,)), ("point", (1,)))
COMPARISONS = (
    ("renewed_public", "parent"),
    ("renewed_public", "fixed_public"),
    ("fixed_public", "parent"),
)
NAMES = tuple(f"{a}-minus-{b}" for a, b in COMPARISONS)


def cases(parent, opponent):
    if (
        type(parent) is not int
        or parent not in (1, 2)
        or opponent not in ("wp", "point")
    ):
        raise ValueError("Expected parent1/2 and wp/point")
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "seat": seat,
            "replicate": replicate,
            "seed": DEAL_START + 10000 * (opponent == "point") + replicate,
        }
        for seat in (0, 1)
        for replicate in range(PAIRS)
    ]


def row_identity(row, parent, opponent):
    if (
        type(parent) is not int
        or parent not in (1, 2)
        or opponent not in ("wp", "point")
    ):
        raise ValueError("Unexpected evaluation group")
    rep, seat = row["replicate"], row["seat"]
    if (
        row["policy"] not in POLICIES
        or type(row["parent"]) is not int
        or row["parent"] != parent
        or row["opponent"] != opponent
        or type(rep) is not int
        or not 0 <= rep < PAIRS
        or type(seat) is not int
        or seat not in (0, 1)
        or type(row["seed"]) is not int
        or row["seed"] != DEAL_START + 10000 * (opponent == "point") + rep
    ):
        raise ValueError("Unexpected confirmation cell or deal key")
    scores = row["terminal_points"]
    if (
        not isinstance(scores, list)
        or len(scores) != 2
        or any(type(p) is not int for p in scores)
        or sum(scores) != 60
    ):
        raise ValueError("Invalid terminal point totals")
    return POLICIES.index(row["policy"]), rep, seat


def load_points(root):
    points = np.empty((2, 3, 2, PAIRS, 2), dtype=np.float64)
    for parent in (1, 2):
        for group, opponent in enumerate(("wp", "point")):
            seen = set()
            path = Path(root) / f"eval_parent{parent}_{opponent}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    identity = row_identity(row, parent, opponent)
                    if identity in seen:
                        raise ValueError("Duplicate confirmation cell")
                    seen.add(identity)
                    policy, rep, seat = identity
                    scores = row["terminal_points"]
                    points[parent - 1, policy, group, rep, seat] = (
                        scores[seat] - scores[1 - seat]
                    )
            if len(seen) != 3 * PAIRS * 2:
                raise ValueError("Incomplete confirmation grid")
    return points


def confirmed_strength(results):
    expected = {
        (p, scope, name) for p in (1, 2) for scope, _ in SCOPES for name in NAMES
    }
    found = {}
    for row in results:
        key = row["parent"], row["scope"], row["comparison"]
        if type(row["parent"]) is not int or key not in expected or key in found:
            raise ValueError("Unexpected or duplicate confirmation contrast")
        for name in ("win", "point"):
            metric = row[name]
            interval = metric["interval95"]
            if (
                len(interval) != 2
                or not np.isfinite([metric["change"], *interval]).all()
                or interval[0] > interval[1]
            ):
                raise ValueError("Invalid confirmation statistic")
        found[key] = row
    if set(found) != expected:
        raise ValueError("Incomplete confirmation contrasts")
    return all(
        found[p, "mixture", name]["win"]["interval95"][0] > 0
        for p in (1, 2)
        for name in NAMES[:2]
    )


def analyse(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (2, 3, 2, PAIRS, 2) or not np.isfinite(points).all():
        raise ValueError("Expected complete finite two-parent three-policy points")
    contrasts, absolute = [], []
    for parent in (1, 2):
        for scope, groups in SCOPES:
            delta_rows = [
                {"parent": parent, "scope": scope, "comparison": name} for name in NAMES
            ]
            absolute_rows = [
                {"parent": parent, "scope": scope, "policy": name} for name in POLICIES
            ]
            for metric, outcomes in (
                ("win", (np.sign(points[parent - 1]) + 1) / 2),
                ("point", points[parent - 1]),
            ):
                # Convert each match to a win score before averaging its two seats.
                paired = outcomes[:, groups].mean(axis=-1)
                changes = np.stack(
                    [
                        paired[POLICIES.index(a)] - paired[POLICIES.index(b)]
                        for a, b in COMPARISONS
                    ]
                )
                estimates = np.concatenate((changes, paired))
                intervals = bootstrap_intervals(
                    estimates, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS
                )
                for i, row in enumerate(delta_rows):
                    row[metric] = {
                        "change": float(changes[i].mean()),
                        "interval95": intervals[i].tolist(),
                    }
                for i, row in enumerate(absolute_rows):
                    row[metric] = {
                        "mean": float(paired[i].mean()),
                        "interval95": intervals[3 + i].tolist(),
                    }
            contrasts.extend(delta_rows)
            absolute.extend(absolute_rows)
    return {
        "contrasts": contrasts,
        "absolute": absolute,
        "confirmed_strength_gain": confirmed_strength(contrasts),
    }
