import numpy as np
import pytest
import torch

from scripts.diagnose_paired_targets import cluster_interval, selected_point, summarize


def test_actual_action_resolves_zero_delta_ties_by_action_id():
    public = torch.zeros(3, 562)
    public[:, 560:] = torch.tensor([[49, 48], [2, 12], [4, 9]])
    # Equal scores use the lower native action ID, not the expert's position.
    actions = torch.tensor([48, 2, 9])
    assert selected_point(public, actions).tolist() == [True, False, True]
    with pytest.raises(ValueError, match="outside"):
        selected_point(public, torch.tensor([1, 2, 9]))


def test_single_action_gain_sign_scaling_and_root_weighting():
    # Three roots from the same match; one from another match.
    groups = np.array([[10, 0], [10, 0], [10, 0], [11, 0]])
    outcomes = np.array([[-1, 1], [-1, 1], [-1, 1], [1, -1]])
    result = summarize(np.ones(4), np.ones(4, dtype=bool), outcomes, groups, 0)
    assert result["source_matches"] == 2
    assert result["single_action_comparisons"]["parent"]["win_score_gain"] == 0.5
    assert result["single_action_comparisons"]["point"]["win_score_gain"] == 0
    assert result["single_action_comparisons"]["parent"][
        "source_match_bootstrap_ci95"
    ] == [-1, 1]
    assert result["validation_mse"] == 3
    assert result["zero_baseline_mse"] == 4
    assert sum(row["roots"] for row in result["calibration_bins"]) == 4


def test_bootstrap_counts_source_matches_not_roots():
    values = np.array([-1, 0, 1, 1])
    assert cluster_interval(values, np.zeros((4, 2), dtype=int)) is None
    with pytest.raises(ValueError, match="nonempty"):
        cluster_interval([], np.zeros((0, 2)))
    with pytest.raises(ValueError, match="Nonfinite"):
        cluster_interval([np.nan], [[0, 0]])


def test_perfect_root_predictions_and_parent_policy():
    outcomes = np.array([[-1, 1], [1, 0], [0, -1], [-1, -1]])
    delta = outcomes[:, 1] - outcomes[:, 0]
    groups = np.column_stack((np.arange(4), np.zeros(4, dtype=int)))
    best = summarize(delta, delta > 0, outcomes, groups, 0.25)
    assert best["validation_mse"] == 0
    assert best["single_action_comparisons"]["parent"]["win_score_gain"] == 0.25
    assert best["single_action_comparisons"]["point"]["win_score_gain"] == 0.25
    parent = summarize(np.zeros(4), np.zeros(4, dtype=bool), outcomes, groups, 0.25)
    assert parent["single_action_comparisons"]["parent"][
        "source_match_bootstrap_ci95"
    ] == [0, 0]
