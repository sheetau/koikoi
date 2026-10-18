"""Synthetic full grids and independent paired-bootstrap arithmetic."""

import copy
import json

import numpy as np
import pytest

import scripts.renewed_confirmation_analysis as confirmation
from scripts.renewed_confirmation_analysis import (
    BOOTSTRAP_SEED,
    DEAL_START,
    PAIRS,
    POLICIES,
    analyse,
    cases,
    confirmed_strength,
    load_points,
    row_identity,
)
from scripts.renewed_continuation_analysis import bootstrap_intervals


def fixture_points():
    points = np.empty((2, 3, 2, PAIRS, 2))
    points[..., 0], points[..., 1] = 6, -4
    points[:, 2, ..., 1] = 4
    return points


def test_sign_before_pair_average_all_contrasts_and_absolute_scores():
    report = analyse(fixture_points())
    assert len(report["contrasts"]) == len(report["absolute"]) == 18
    assert report["confirmed_strength_gain"]
    for row in report["absolute"]:
        assert row["win"]["mean"] == (1 if row["policy"] == "renewed_public" else 0.5)
        assert row["point"]["mean"] == (5 if row["policy"] == "renewed_public" else 1)
        for metric in ("win", "point"):
            assert row[metric]["interval95"] == [row[metric]["mean"]] * 2
    for row in report["contrasts"]:
        expected = 0 if row["comparison"] == "fixed_public-minus-parent" else 0.5
        assert row["win"]["change"] == expected
        assert row["win"]["interval95"] == [expected, expected]


def test_four_bounds_required_even_with_positive_points_and_other_cells():
    rows = analyse(fixture_points())["contrasts"]
    selected = next(
        r
        for r in rows
        if r["parent"] == 2
        and r["scope"] == "mixture"
        and r["comparison"] == "renewed_public-minus-fixed_public"
    )
    selected["win"]["interval95"][0] = 0
    assert not confirmed_strength(rows)
    for bad in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
        with pytest.raises(ValueError):
            confirmed_strength(bad)
    selected["point"]["change"] = float("nan")
    with pytest.raises(ValueError, match="statistic"):
        confirmed_strength(rows)


def test_bootstrap_4096_pair_arithmetic_matches_independent_scalar_loop():
    values = np.empty((2, 2, PAIRS))
    i = np.arange(PAIRS)
    values[0, 0], values[0, 1] = (i % 7 - 3) / 4, (i % 11 - 5) / 4
    values[1, 0], values[1, 1] = i % 5 - 2, i % 13 - 6
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scalar = []
    for _ in range(257):  # Cross the producer's chunk boundary twice.
        sampled = rng.integers(0, PAIRS, size=(2, PAIRS))
        scalar.append(
            [
                sum(float(values[k, g, sampled[g]].sum()) for g in range(2))
                / (2 * PAIRS)
                for k in range(2)
            ]
        )
    expected = np.percentile(np.asarray(scalar), [2.5, 97.5], axis=0).T
    np.testing.assert_array_equal(
        bootstrap_intervals(values, seed=BOOTSTRAP_SEED, draws=257), expected
    )


def test_nonconstant_analysis_connects_absolute_and_difference_intervals(monkeypatch):
    monkeypatch.setattr(confirmation, "BOOTSTRAP_DRAWS", 257)
    points = fixture_points()
    indices = np.arange(PAIRS)
    for policy in range(3):
        for group in range(2):
            shift = 2 * policy * (1 if group == 0 else -1)
            points[:, policy, group, :, 0] = 2 * (indices % 5 - 2) + shift
            points[:, policy, group, :, 1] = 2 * (indices % 7 - 3) + shift
    report = analyse(points)
    absolute = next(
        r
        for r in report["absolute"]
        if r["parent"] == 1
        and r["scope"] == "mixture"
        and r["policy"] == "renewed_public"
    )
    contrast = next(
        r
        for r in report["contrasts"]
        if r["parent"] == 1
        and r["scope"] == "mixture"
        and r["comparison"] == "renewed_public-minus-parent"
    )
    for metric in ("win", "point"):
        values = (np.sign(points[0]) + 1) / 2 if metric == "win" else points[0]
        paired = values.mean(axis=-1)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means, differences = [], []
        for _ in range(257):
            selected = rng.integers(0, PAIRS, size=(2, PAIRS))
            candidate = sum(
                float(paired[2, g, selected[g]].sum()) for g in range(2)
            ) / (2 * PAIRS)
            baseline = sum(float(paired[0, g, selected[g]].sum()) for g in range(2)) / (
                2 * PAIRS
            )
            means.append(candidate)
            differences.append(candidate - baseline)
        np.testing.assert_array_equal(
            absolute[metric]["interval95"], np.percentile(means, [2.5, 97.5])
        )
        np.testing.assert_array_equal(
            contrast[metric]["interval95"], np.percentile(differences, [2.5, 97.5])
        )


def test_full_synthetic_grid_and_strict_identity_checks(tmp_path):
    def record(parent, opponent, policy, rep, seat):
        return {
            "parent": parent,
            "opponent": opponent,
            "policy": policy,
            "replicate": rep,
            "seat": seat,
            "seed": DEAL_START + 10000 * (opponent == "point") + rep,
            "terminal_points": [31, 29],
        }

    for parent in (1, 2):
        for opponent in ("wp", "point"):
            with (tmp_path / f"eval_parent{parent}_{opponent}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                for policy in POLICIES:
                    for rep in range(PAIRS):
                        for seat in (0, 1):
                            stream.write(
                                json.dumps(record(parent, opponent, policy, rep, seat))
                                + "\n"
                            )
    points = load_points(tmp_path)
    assert points.shape == (2, 3, 2, PAIRS, 2)
    np.testing.assert_array_equal(points[..., 0], 2)
    np.testing.assert_array_equal(points[..., 1], -2)
    path = tmp_path / "eval_parent2_point.jsonl"
    first = record(2, "point", "parent", 0, 0)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(first) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_points(tmp_path)
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        load_points(tmp_path)
    for change in (
        {"seed": 95120001},
        {"seat": True},
        {"parent": True},
        {"policy": "renewed_masked"},
        {"terminal_points": [30, 29]},
    ):
        with pytest.raises(ValueError):
            row_identity({**first, **change}, 2, "point")


def test_case_allocation_is_shared_and_does_not_construct_games():
    wp, point = cases(1, "wp"), cases(2, "point")
    assert len(wp) == len(point) == 8192
    assert len({r["seed"] for r in wp + point}) == 8192
    assert not {r["seed"] for r in wp} & {r["seed"] for r in point}
    assert {r["seed"] for r in wp} == {r["seed"] for r in cases(2, "wp")}
    with pytest.raises(ValueError):
        cases(True, "wp")
