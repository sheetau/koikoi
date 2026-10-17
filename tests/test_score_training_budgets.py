import numpy as np
import pytest
import torch

from scripts.collect_reusable_paired_targets import collect
from scripts.compare_score_training_budgets import disjoint, evaluate, paired_change
from scripts.paired_rollout_targets import match_split
from scripts.replay_reusable_paired_features import replay
from scripts.run_arbitration_study import PARENTS, sha
from scripts.train_score_distribution import fit


def test_shared_cohort_evaluates_raw_and_feature_transfer(tmp_path):
    training_path, evaluation_path = (
        tmp_path / "training.pt",
        tmp_path / "evaluation.pt",
    )
    collect(
        PARENTS[0][0], training_path, seed=18966001, roots=32, batch=16, device="cpu"
    )
    evaluation = collect(
        PARENTS[0][0], evaluation_path, seed=18967001, roots=32, batch=16, device="cpu"
    )
    train_cache, evaluation_cache = (
        tmp_path / "train_features.pt",
        tmp_path / "eval_features.pt",
    )
    replay(training_path, train_cache)
    cache = replay(evaluation_path, evaluation_cache)
    valid = match_split(evaluation["match_groups"])
    for representation in ("raw", "features"):
        checkpoint = tmp_path / f"{representation}.pt"
        fit(
            training_path,
            checkpoint,
            kind="preference",
            epochs=2,
            feature_cache=train_cache if representation == "features" else None,
        )
        report, wins, points = evaluate(
            checkpoint, evaluation, sha(evaluation_path), cache, sha(evaluation_cache)
        )
        assert report["training_evaluation_groups_disjoint"]
        assert report["evaluation_roots"] == int(valid.sum())
        assert report["representation"] == representation
        assert len(wins) == len(points) == int(valid.sum())
        assert np.array_equal(wins, np.sign(points))
        parent = evaluation["outcomes"][valid, 0].numpy()
        assert report["single_action_comparisons"]["parent"][
            "win_score_gain"
        ] == pytest.approx(float(((wins - parent) / 2).mean()))
        zero = paired_change(
            (wins, points), (wins, points), evaluation["match_groups"][valid].numpy()
        )
        assert zero["win_score_change"] == zero["point_margin_change"] == 0
        assert zero["win_source_match_ci95"] == [0, 0]
    with pytest.raises(ValueError, match="cache mismatch"):
        evaluate(checkpoint, evaluation, "wrong", cache, sha(evaluation_cache))
    evaluation["parent"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="identities"):
        evaluate(
            checkpoint, evaluation, sha(evaluation_path), cache, sha(evaluation_cache)
        )


def test_group_leakage_and_paired_sign_units():
    with pytest.raises(ValueError, match="overlap"):
        disjoint(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[3, 4], [5, 6]]))
    disjoint(torch.tensor([[1, 2]]), torch.tensor([[1, 3]]))
    left = np.array([1.0, 0.0, -1.0]), np.array([4.0, 0.0, -8.0])
    right = np.array([-1.0, -1.0, 0.0]), np.array([-2.0, -6.0, 0.0])
    result = paired_change(left, right, np.array([[1, 0], [2, 0], [3, 0]]))
    assert result["win_score_change"] == pytest.approx(1 / 3)
    assert result["point_margin_change"] == pytest.approx(4 / 3)
