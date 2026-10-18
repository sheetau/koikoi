"""Paired five-policy analysis for the prospective chance-renewal comparison.

This module does not run games or load a partially completed study implicitly.
The future runner and independent auditor own completion/provenance checks.
"""

import json
from pathlib import Path

import numpy as np

POLICIES = (
    "parent",
    "fixed_public",
    "fixed_masked",
    "renewed_public",
    "renewed_masked",
)
PAIRS = 1024
BOOTSTRAP_SEED = 95310001
BOOTSTRAP_DRAWS = 10000
COMPARISONS = (
    ("renewed_public", "parent"),
    ("renewed_public", "fixed_public"),
    ("renewed_public", "fixed_masked"),
    ("renewed_public", "renewed_masked"),
    ("renewed_masked", "parent"),
    ("renewed_masked", "fixed_public"),
    ("renewed_masked", "fixed_masked"),
    ("fixed_public", "parent"),
    ("fixed_public", "fixed_masked"),
    ("fixed_masked", "parent"),
)
INTERACTION = "chance_by_memory"
NAMES = tuple(f"{a}-minus-{b}" for a, b in COMPARISONS) + (INTERACTION,)
SCOPES = (("mixture", (0, 1)), ("wp", (0,)), ("point", (1,)))


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
            "seed": 95110001 + 10000 * (opponent == "point") + replicate,
        }
        for seat in (0, 1)
        for replicate in range(PAIRS)
    ]


def load_points(root):
    """Require the whole40,960-match grid;this is not a trajectory audit."""
    points = np.empty((2, 5, 2, PAIRS, 2), dtype=np.float64)
    for parent in (1, 2):
        for group, opponent in enumerate(("wp", "point")):
            seen = set()
            path = Path(root) / f"eval_parent{parent}_{opponent}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    policy = row["policy"]
                    rep, seat = row["replicate"], row["seat"]
                    if (
                        policy not in POLICIES
                        or type(rep) is not int
                        or not 0 <= rep < PAIRS
                        or type(seat) is not int
                        or seat not in (0, 1)
                        or type(row["parent"]) is not int
                        or row["parent"] != parent
                        or row["opponent"] != opponent
                        or type(row["seed"]) is not int
                        or row["seed"] != 95110001 + 10000 * group + rep
                    ):
                        raise ValueError("Unexpected evaluation cell or deal key")
                    key = (policy, rep, seat)
                    if key in seen:
                        raise ValueError("Duplicate evaluation cell")
                    seen.add(key)
                    scores = row["terminal_points"]
                    if (
                        not isinstance(scores, list)
                        or len(scores) != 2
                        or any(type(v) is not int for v in scores)
                        or sum(scores) != 60
                    ):
                        raise ValueError("Invalid terminal point totals")
                    points[parent - 1, POLICIES.index(policy), group, rep, seat] = (
                        scores[seat] - scores[1 - seat]
                    )
            if len(seen) != 5 * PAIRS * 2:
                raise ValueError("Incomplete evaluation grid")
    return points


def bootstrap_intervals(changes, *, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """Bootstrap shared paired deals independently within each opponent stratum."""
    changes = np.asarray(changes, dtype=np.float64)
    if changes.ndim != 3 or min(changes.shape) < 1 or not np.isfinite(changes).all():
        raise ValueError("Expected finite contrast by stratum by paired-deal changes")
    if type(draws) is not int or draws < 1:
        raise ValueError("Expected positive bootstrap draw count")
    # Constant within-stratum changes have a degenerate exact bootstrap distribution.
    if np.all(changes == changes[:, :, :1]):
        return np.repeat(changes.mean(axis=(1, 2))[:, None], 2, axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((len(changes), draws))
    for start in range(0, draws, 128):
        count = min(128, draws - start)
        indices = rng.integers(
            0, changes.shape[2], (count, changes.shape[1], changes.shape[2])
        )
        selected = changes[:, np.arange(changes.shape[1])[None, :, None], indices]
        samples[:, start : start + count] = selected.mean(axis=(2, 3))
    return np.quantile(samples, [0.025, 0.975], axis=1).T


def analyse(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (2, 5, 2, PAIRS, 2) or not np.isfinite(points).all():
        raise ValueError("Expected complete finite two-parent five-policy point array")
    results = []
    for parent in (1, 2):
        for scope, groups in SCOPES:
            records = [
                {"parent": parent, "comparison": name, "scope": scope} for name in NAMES
            ]
            for metric, outcomes in (
                ("win", (np.sign(points[parent - 1]) + 1) / 2),
                ("point", points[parent - 1]),
            ):
                # Average seats after computing each match's win outcome.
                paired = outcomes[:, groups].mean(axis=-1)
                changes = [
                    paired[POLICIES.index(a)] - paired[POLICIES.index(b)]
                    for a, b in COMPARISONS
                ]
                changes.append((paired[3] - paired[4]) - (paired[1] - paired[2]))
                changes = np.stack(changes)
                intervals = bootstrap_intervals(changes)
                for index, record in enumerate(records):
                    record[metric] = {
                        "change": float(changes[index].mean()),
                        "interval95": intervals[index].tolist(),
                    }
            results.extend(records)
    return results


def decisions(results):
    """Keep strength,public-memory benefit and their interaction separate."""
    expected = {(p, s, n) for p in (1, 2) for s, _ in SCOPES for n in NAMES}
    records = {}
    for row in results:
        key = (row["parent"], row["scope"], row["comparison"])
        if key not in expected or key in records:
            raise ValueError("Unexpected or duplicate analysis cell")
        for metric in ("win", "point"):
            value = row[metric]
            interval = value["interval95"]
            if (
                len(interval) != 2
                or not np.isfinite([value["change"], *interval]).all()
            ):
                raise ValueError("Invalid analysis statistic")
            if interval[0] > interval[1]:
                raise ValueError("Reversed confidence interval")
        records[key] = row
    if set(records) != expected:
        raise ValueError("Incomplete analysis cells")

    def passed(names):
        return all(
            records[parent, "mixture", name]["win"]["interval95"][0] > 0
            for parent in (1, 2)
            for name in names
        )

    return {
        "replicated_strength_gain": passed(
            ("renewed_public-minus-parent", "renewed_public-minus-fixed_public")
        ),
        "replicated_public_memory_gain": passed(
            ("renewed_public-minus-renewed_masked",)
        ),
        "replicated_positive_interaction": passed((INTERACTION,)),
    }
