import numpy as np
import pytest
import torch

from scripts.diagnose_switch_budget import threshold_for_budget


def test_switch_budget_is_training_only_strict_and_positive():
    train = torch.arange(-50, 50, dtype=torch.float32)
    threshold = threshold_for_budget(train)
    assert threshold == 40
    # Nine strictly better values; a threshold tie keeps the parent action.
    validation = torch.tensor([-5.0, 40.0, 41.0, 500.0])
    assert (validation > threshold).tolist() == [False, False, True, True]
    assert (train > threshold).sum() == 9
    assert threshold_for_budget(torch.zeros(100)) == 0
    assert threshold_for_budget(-torch.arange(1, 101, dtype=torch.float32)) == 0
    for invalid in (0, 1, -0.1, float("nan")):
        with pytest.raises(ValueError):
            threshold_for_budget(train, invalid)


def test_actual_feature_fit_budget_diagnostic(tmp_path):
    from scripts.collect_opponent_paired_targets import collect
    from scripts.diagnose_switch_budget import diagnose
    from scripts.run_arbitration_study import PARENTS
    from scripts.train_score_distribution import fit

    base = tmp_path / "data"
    collect(PARENTS[0][0], base, 18972001, roots=64, batch=16, envs=16, device="cpu")
    folder = base / "opponent_preserving"
    dataset, cache = folder / "targets.pt", folder / "features.pt"
    checkpoint = tmp_path / "fitted.pt"
    fit(dataset, checkpoint, kind="paired_win", epochs=2, feature_cache=cache)
    report = diagnose(checkpoint, tmp_path / "diagnostic.json")
    assert 0 <= report["training_switch_fraction"] <= 0.1
    assert report["threshold"] >= 0
    assert report["validation_roots"] > 0
    # Pairwise gains must agree algebraically with the two expert-relative gains.
    difference = (
        report["restricted"]["parent"]["win_score_gain"]
        - report["unrestricted"]["parent"]["win_score_gain"]
    )
    assert report["restricted_minus_unrestricted"]["win_score_change"] == pytest.approx(
        difference, abs=1e-7
    )
    for key in (
        "restricted_minus_unrestricted",
        "restricted_minus_same_rate_random_expectation",
    ):
        assert np.isfinite(report[key]["win_score_change"])
    with pytest.raises(ValueError, match="fresh"):
        diagnose(checkpoint, tmp_path / "diagnostic.json")
