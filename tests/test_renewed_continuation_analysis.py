"""Independent paired-outcome fixtures for the five-policy renewal design."""

import copy
import json

import numpy as np
import pytest

from scripts.renewed_continuation_analysis import (
    analyse,
    bootstrap_intervals,
    cases,
    decisions,
    load_points,
)


def points_fixture():
    points = np.empty((2, 5, 2, 1024, 2))
    points[..., 0], points[..., 1] = 2, -2
    points[:, 3] = 4
    return points


def test_seat_averaging_primary_memory_and_interaction_are_distinct():
    points = points_fixture()
    records = analyse(points)
    assert len(records) == 66
    assert decisions(records) == {
        "replicated_strength_gain": True,
        "replicated_public_memory_gain": True,
        "replicated_positive_interaction": True,
    }
    for row in records:
        if row["comparison"] in (
            "renewed_public-minus-parent",
            "renewed_public-minus-renewed_masked",
            "chance_by_memory",
        ):
            assert row["win"] == {"change": 0.5, "interval95": [0.5, 0.5]}
            assert row["point"] == {"change": 4.0, "interval95": [4.0, 4.0]}

    # Both renewed policies improve identically:strength passes,signal does not.
    points[:, 4] = 4
    assert decisions(analyse(points)) == {
        "replicated_strength_gain": True,
        "replicated_public_memory_gain": False,
        "replicated_positive_interaction": False,
    }
    # Public identity helps equally with fixed and fresh chance:no renewal gain.
    points = points_fixture()
    points[:, 1] = 4
    assert decisions(analyse(points)) == {
        "replicated_strength_gain": False,
        "replicated_public_memory_gain": True,
        "replicated_positive_interaction": False,
    }


def test_one_parent_or_opponent_and_point_margin_cannot_replace_primary():
    points = points_fixture()
    points[1, 3, :, :, 0], points[1, 3, :, :, 1] = 60, -2
    results = analyse(points)
    assert not decisions(results)["replicated_strength_gain"]
    record = next(
        r
        for r in results
        if (r["parent"], r["scope"], r["comparison"])
        == (2, "mixture", "renewed_public-minus-parent")
    )
    assert record["point"]["change"] == 29
    assert record["win"]["change"] == 0
    # A benefit only against WP cancels an equal loss against Point.
    points = points_fixture()
    points[:, 3, 1] = -4
    assert not decisions(analyse(points))["replicated_strength_gain"]


def test_vector_bootstrap_matches_scalar_paired_stratified_calculation():
    # Opposite effects across strata and correlated contrasts expose pooling or
    # independent policy/seat resampling. Include more than one chunk.
    changes = np.array(
        [
            [[0, 1, -1, 0.5, 0.5], [2, 0, 0, -2, 1]],
            [[0, -2, 2, -1, -1], [-4, 0, 0, 4, -2]],
        ]
    )
    rng = np.random.default_rng(95310001)
    sampled = [[], []]
    for _ in range(257):
        first, second = rng.integers(0, 5, size=(2, 5))
        for index in range(2):
            value = (
                sum(changes[index, 0, j] for j in first) / 5
                + sum(changes[index, 1, j] for j in second) / 5
            ) / 2
            sampled[index].append(value)
    expected = np.array([np.percentile(values, [2.5, 97.5]) for values in sampled])
    np.testing.assert_allclose(bootstrap_intervals(changes, draws=257), expected)
    constant = np.array([[[1, 1], [-3, -3]]])
    np.testing.assert_array_equal(bootstrap_intervals(constant), [[-1, -1]])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_intervals([[[float("nan")]]])


def test_complete_grid_deal_identity_and_malformed_records(tmp_path):
    policies = (
        "parent",
        "fixed_public",
        "fixed_masked",
        "renewed_public",
        "renewed_masked",
    )
    count, seeds = 0, set()
    for parent in (1, 2):
        for index, opponent in enumerate(("wp", "point")):
            rows = []
            generated = cases(parent, opponent)
            assert len(generated) == 2048
            for policy_index, policy in enumerate(policies):
                for seat in (0, 1):
                    for replicate in range(1024):
                        case = {
                            "parent": parent,
                            "opponent": opponent,
                            "seat": seat,
                            "replicate": replicate,
                            "seed": 95110001 + index * 10000 + replicate,
                        }
                        assert generated[seat * 1024 + replicate] == case
                        scores = [0, 0]
                        scores[seat] = 28 + policy_index
                        scores[1 - seat] = 32 - policy_index
                        rows.append(
                            json.dumps(
                                {**case, "policy": policy, "terminal_points": scores}
                            )
                        )
                        seeds.add(case["seed"])
                        count += 1
            (tmp_path / f"eval_parent{parent}_{opponent}.jsonl").write_text(
                "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
            )
    assert count == 40960 and len(seeds) == 2048
    points = load_points(tmp_path)
    for policy in range(5):
        assert (points[:, policy] == 2 * policy - 4).all()

    path = tmp_path / "eval_parent1_wp.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    for field, value in (
        ("seed", 93110001),
        ("parent", 2),
        ("replicate", True),
        ("replicate", 1024),
        ("seat", -1),
        ("policy", "public"),
        ("terminal_points", [30.0, 30]),
        ("terminal_points", [31, 30]),
    ):
        bad = json.loads(lines[0])
        bad[field] = value
        path.write_text(
            "\n".join([json.dumps(bad), *lines[1:]]) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_points(tmp_path)
    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_points(tmp_path)
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        load_points(tmp_path)


def test_decision_requires_all_cells_and_strictly_positive_bound():
    records = analyse(points_fixture())
    for invalid in (records[:-1], [*records, records[0]]):
        with pytest.raises(ValueError):
            decisions(invalid)
    copied = copy.deepcopy(records)
    primary = next(
        r
        for r in copied
        if (r["parent"], r["scope"], r["comparison"])
        == (2, "mixture", "renewed_public-minus-fixed_public")
    )
    primary["win"]["interval95"][0] = 0
    assert not decisions(copied)["replicated_strength_gain"]
    primary["win"]["interval95"] = [1, -1]
    with pytest.raises(ValueError, match="Reversed"):
        decisions(copied)
    with pytest.raises(ValueError, match="complete finite"):
        analyse(np.zeros((2, 4, 2, 1024, 2)))
