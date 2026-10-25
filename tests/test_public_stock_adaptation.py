"""V5A fixed-family, paired-comparison and endpoint-identity checks."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.public_stock_adaptation import (
    ALL_THREE,
    CONTROL,
    EVALUATE,
    PREVIEW_CHOICE,
    PROFILES,
    compare,
    launch_gate,
    sha,
    validate_evaluation,
)


def v4_evidence():
    comparisons = {
        f"{profile}_parent{p}_{key}": {"lower": 0.1}
        for profile in (PREVIEW_CHOICE, ALL_THREE)
        for p in range(2) for key in ("gain", "excess", "high_minus_low")
    }
    comparisons.update({f"preview_minus_all_parent{p}": {"lower": -0.1} for p in range(2)})
    return {
        "method": "public_stock_robustness_v4", "comparisons": comparisons,
        "evaluation_totals": {"matches": 4608},
        "score_tests_passed": {PREVIEW_CHOICE: True, ALL_THREE: True},
        "all_work_targets_passed": False,
    }


def test_score_launch_gate_does_not_convert_failed_work_matching_to_success():
    evidence = v4_evidence()
    assert all(launch_gate(evidence).values())
    assert evidence["all_work_targets_passed"] is False


@pytest.mark.parametrize("failure", ["incomplete", "missing_comparison", "false_pass", "no_finalist"])
def test_incomplete_or_failed_v4_cannot_launch_adaptation(failure):
    evidence = v4_evidence()
    if failure == "incomplete":
        evidence["evaluation_totals"]["matches"] -= 1
    elif failure == "missing_comparison":
        evidence["comparisons"].pop("preview_minus_all_parent0")
    elif failure == "false_pass":
        evidence["comparisons"][f"{ALL_THREE}_parent1_gain"]["lower"] = 0
    else:
        for profile in (PREVIEW_CHOICE, ALL_THREE):
            evidence["comparisons"][f"{profile}_parent0_gain"]["lower"] = -0.01
            evidence["score_tests_passed"][profile] = False
    with pytest.raises(ValueError):
        launch_gate(evidence)


def paired_scores():
    old_other = np.array([0.25, 0.5, 0.5, 0.75])
    return {
        f"{profile}_parent{p}": np.stack([np.full(4, .75), old_other + .125, old_other])
        for profile in PROFILES for p in range(2)
    }


def test_shared_stream_variation_cancels_in_opponent_matched_difference():
    indices = np.random.default_rng(51).integers(0, 4, (500, 4))
    contrasts, passed, eligible = compare(paired_scores(), indices)
    assert len(contrasts) == 12
    assert contrasts[f"{PREVIEW_CHOICE}_parent1_other"] == {"mean": .125, "lower": .125}
    assert contrasts[f"{PREVIEW_CHOICE}_parent1_own"] == {"mean": .25, "lower": .25}
    assert all(passed.values())
    assert eligible == [PREVIEW_CHOICE, ALL_THREE]


def test_one_failed_parent_or_control_prevents_stronger_background_claim():
    data = paired_scores()
    indices = np.tile(np.arange(4), (100, 1))
    data[f"{PREVIEW_CHOICE}_parent1"][0] = .5
    _, passed, eligible = compare(data, indices)
    assert not passed[PREVIEW_CHOICE] and eligible == [ALL_THREE]
    data[f"{CONTROL}_parent0"][0] = .25
    _, passed, eligible = compare(data, indices)
    assert not passed[CONTROL] and eligible == []


@pytest.fixture
def endpoint(tmp_path):
    checkpoint, opponent, folder = tmp_path / "new.pt", tmp_path / "old.pt", tmp_path / "result"
    checkpoint.write_bytes(b"new")
    opponent.write_bytes(b"old")
    sources = {
        p: "source" for p in (
            "scripts/evaluate_public_stock_residual.py", "koikoi/public_stock_residual.py",
            "koikoi/public_stock_residual_search.py", "koikoi/public_stock_search.py",
            "koikoi/public_stock.py", "koikoi/public_stock_native.py",
            "koikoi/native/public_stock.h", "koikoi/native/core.h",
        )
    }
    result = {
        "config": {**deepcopy(EVALUATE), "checkpoint": str(checkpoint), "opponent": str(opponent), "out": str(folder)},
        "profile": PREVIEW_CHOICE, "checkpoint_sha256": sha(checkpoint), "opponent_sha256": sha(opponent),
        "matches": 2048, "rollouts": 0, "transitions": 0, "roots": 0,
        "source_hashes": sources,
    }
    return result, {"sources": sources.copy()}, checkpoint, opponent, PREVIEW_CHOICE, folder


def test_fixed_qualification_receipt_is_accepted(endpoint):
    validate_evaluation(*endpoint)


@pytest.mark.parametrize("failure", ["opponent", "seed", "search", "source", "matches"])
def test_wrong_opponent_cohort_or_search_cannot_enter_qualification(endpoint, failure):
    result = endpoint[0]
    if failure == "opponent":
        result["opponent_sha256"] = result["checkpoint_sha256"]
    elif failure == "seed":
        result["config"]["deal_seed"] -= 1
    elif failure == "search":
        result["transitions"] = 1
    elif failure == "source":
        result["source_hashes"].pop("koikoi/public_stock.py")
    else:
        result["matches"] -= 1
    with pytest.raises(ValueError):
        validate_evaluation(*endpoint)
