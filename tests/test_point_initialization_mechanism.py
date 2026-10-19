import json
import math

import numpy as np
import pytest

from scripts.audit_koi_temperature_mechanism import equal_scalars
from scripts.audit_point_initialization_mechanism import scalar_description
from scripts.point_initialization_mechanism import (
    describe,
    disagreements,
    greedy,
    validation_indices,
    write_new,
)


def fixture():
    values = np.full((9, 50), -np.inf, dtype=np.float32)
    phases = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    for i in range(9):
        first = 48 if phases[i] == 3 else 0
        values[i, first : first + 2] = [0.0, 0.05 * math.log(3)]
    mask = np.isfinite(values)
    mask[8, 49] = False
    return values, mask, phases


def test_every_phase_uses_its_declared_divisor_and_own_initialization():
    values, mask, phases = fixture()
    current = greedy(values, mask)
    wp, point = current.copy(), current.copy()
    wp[0] = 0
    point[6] = 48
    temperatures = [0.1, 0.2, 0.3, 0.4]
    actual = describe(values, mask, phases, current, wp, point, "point", temperatures)
    expected = scalar_description(
        values, mask, phases, current, wp, point, "point", temperatures
    )
    equal_scalars(expected, actual)
    assert actual["0"]["changed_vs_wp"] == 1
    assert actual["0"]["changed_vs_initial"] == 0
    assert actual["3"]["changed_vs_initial"] == 1
    assert actual["3"]["stop_to_continue_vs_initial"] == 1
    assert actual["3"]["all_rows"] == 3 and actual["3"]["unforced_rows"] == 2
    declared = [
        actual[str(p)]["declared_temperature"]["mean_normalized_entropy"]
        for p in range(4)
    ]
    assert declared == sorted(declared) and len(set(declared)) == 4
    for p in range(4):
        assert (
            actual[str(p)]["common_temperature"]["mean_normalized_entropy"]
            < declared[p]
        )


def test_score_rescaling_preserves_probabilities_but_not_raw_gap():
    values, mask, phases = fixture()
    current = greedy(values, mask)
    base = describe(values, mask, phases, current, current, current, "wp", [0.05] * 4)
    scaled = describe(
        values * 4, mask, phases, current, current, current, "wp", [0.2] * 4
    )
    for phase in range(4):
        a, b = base[str(phase)], scaled[str(phase)]
        equal_scalars(a["declared_temperature"], b["declared_temperature"])
        assert b["mean_top_two_raw_score_gap"] == 4 * a["mean_top_two_raw_score_gap"]
        assert a["changed_vs_initial"] == b["changed_vs_initial"] == 0


def test_negative_groups_and_forced_choices_cannot_enter_comparisons(tmp_path):
    assert validation_indices(np.array([-5, -1, 0, 3, 4, 7])).tolist() == [3, 5]
    values, mask, phases = fixture()
    actions = greedy(values, mask)
    assert actions[-1] == -1
    raw = {"mask": mask, "phases": phases}
    for seed in (1, 2):
        for mode in ("wp_fixed", "point_fixed", "point_matched"):
            raw[f"seed{seed}_{mode}_actions"] = actions.copy()
    raw["seed1_point_matched_actions"][-1] = 49
    raw["seed1_point_matched_actions"][6] = 48
    result = disagreements(raw)
    assert len(result) == 6
    assert result["seed1_point_fixed__seed1_point_matched"]["3"] == 1
    assert all(
        v == 0 for v in result["seed2_point_fixed__seed2_point_matched"].values()
    )
    target = tmp_path / "record.json"
    write_new(target, {"old": True})
    with pytest.raises(FileExistsError):
        write_new(target, {"old": False})
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
