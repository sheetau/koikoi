import pytest

from koikoi.compare import compare_reports


def report(score, seeds=(1, 2, 3)):
    return {
        "profile": "paper8",
        "opponent": "rl-wp",
        "split": "development",
        "protocol_version": 2,
        "records": [
            {
                "seed": seed,
                "seat0_score": score,
                "seat1_score": score,
                "seat0_point_difference": 10 * score,
                "seat1_point_difference": 10 * score,
            }
            for seed in seeds
        ],
    }


def test_shared_pairs_and_direction():
    result = compare_reports(report(0.75), report(0.25, (2, 3, 4)))
    assert result["shared_pairs"] == 2
    assert result["score_difference_first_minus_second"] == 0.5
    assert result["point_difference_first_minus_second"] == 5


def test_refuse_incomparable_or_unpaired_results():
    other = report(0.5)
    other["opponent"] = "greedy"
    with pytest.raises(ValueError, match="opponent"):
        compare_reports(report(0.5), other)
    with pytest.raises(ValueError, match="At least"):
        compare_reports(report(0.5), report(0.5, (3, 4)))


def test_refuse_changed_opponent_weights():
    first, second = report(0.5), report(0.5)
    for item, digest in ((first, "old"), (second, "new")):
        item["opponent"] = "opponent.pt"
        item["checkpoints"] = {"opponent.pt": {"sha256": digest}}
    with pytest.raises(ValueError, match="checkpoint bytes"):
        compare_reports(first, second)
