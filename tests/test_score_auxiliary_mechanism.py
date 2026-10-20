import math

import numpy as np
import pytest

from scripts.audit_koi_temperature_mechanism import equal_scalars
from scripts.audit_score_auxiliary_mechanism import scalar_description
from scripts.point_initialization_mechanism import greedy, validation_indices
from scripts.run_arbitration_study import read
from scripts.score_auxiliary_mechanism import (
    describe,
    disagreements,
    learning_diagnostics,
)


def fixture():
    scores = np.full((9, 50), -np.inf, dtype=np.float32)
    phases = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    for i in range(9):
        first = 48 if phases[i] == 3 else 0
        scores[i, first : first + 2] = [0, 0.05 * math.log(3)]
    mask = np.isfinite(scores)
    mask[-1, 49] = False
    return scores, mask, phases


def test_known_entropy_and_both_koi_directions_exclude_forced_rows():
    scores, mask, phases = fixture()
    scores[7, 48:50] = scores[7, [49, 48]]
    actions = greedy(scores, mask)
    original = actions.copy()
    original[6], original[7] = 48, 49
    scores[-1] = np.nan  # No neural output is needed for a forced action.
    actual = describe(scores, mask, phases, actions, original)
    equal_scalars(scalar_description(scores, mask, phases, actions, original), actual)
    entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2)
    for phase in range(4):
        assert actual[str(phase)]["common_temperature"][
            "mean_normalized_entropy"
        ] == pytest.approx(entropy, abs=1e-7)
    assert actual["3"]["changed_vs_point"] == 2
    assert actual["3"]["stop_to_continue_vs_point"] == 1
    assert actual["3"]["continue_to_stop_vs_point"] == 1
    assert actual["3"]["forced_rows"] == 1 and actual["3"]["unforced_rows"] == 2


def test_all_matched_contrasts_keep_seeds_and_forced_choices_separate():
    scores, mask, phases = fixture()
    actions = greedy(scores, mask)
    raw = {"mask": mask, "phases": phases}
    for seed in (1, 2):
        for kind in ("none", "win", "score"):
            raw[f"seed{seed}_{kind}_actions"] = actions.copy()
    raw["seed1_score_actions"][6] = 48
    raw["seed1_score_actions"][-1] = 49
    pairs = disagreements(raw)
    assert len(pairs) == 6
    assert pairs["seed1_win__seed1_score"]["3"] == 1
    assert pairs["seed1_none__seed1_score"]["3"] == 1
    assert sum(sum(row.values()) for row in pairs.values()) == 2
    assert validation_indices(np.array([-5, -1, 0, 3, 4, 7])).tolist() == [3, 5]


def test_all_training_cycles_retained_and_missing_or_duplicate_cycle_rejected():
    rows = [
        {
            "auxiliary_phase": {"cycle": (i - 20) // 5, "mean_training_losses": None}
            if i > 20 and (i - 20) % 5 == 0
            else None
        }
        for i in range(1, 521)
    ]
    phases = learning_diagnostics(rows)
    assert len(phases) == 100 and phases[0]["mean_training_losses"] is None
    with pytest.raises(AssertionError):
        learning_diagnostics(rows[:-1])
    rows[-1]["auxiliary_phase"]["cycle"] = 99
    with pytest.raises(AssertionError):
        learning_diagnostics(rows)


def test_reused_point_population_matches_previously_verified_statistics():
    with np.load(
        "runs/point_initialization_mechanism/scores.npz", allow_pickle=False
    ) as raw:
        scores, mask, phases = raw["point_scores"], raw["mask"], raw["phases"]
        assert len(scores) == 3258 and int((mask.sum(1) > 1).sum()) == 3045
        actions = greedy(scores, mask)
        result = describe(scores, mask, phases, actions, actions)
        equal_scalars(
            scalar_description(scores, mask, phases, actions, actions), result
        )
        old = read("runs/point_initialization_mechanism/report.json")["results"][
            "initial_point_fixed"
        ]
        for phase in range(4):
            item = result[str(phase)]
            equal_scalars(
                old[str(phase)]["common_temperature"], item["common_temperature"]
            )
            assert item["changed_vs_point"] == 0
        assert sum(item["forced_rows"] for item in result.values()) == 213
