import copy

import numpy as np
import pytest

from scripts.audit_score_auxiliary_statistics import check_statistics
from scripts.score_auxiliary_analysis import analyse


def test_all_coordinates_and_six_comparisons_match_separate_scalar_bootstrap():
    points = np.random.default_rng(8).integers(-8, 9, size=(2, 4, 2, 17, 2)) * 2
    report = analyse(points, seed=11, replicates=111)
    check_statistics(points, report, 11, 111)
    assert len(report["rows"]) == 120
    assert len(report["primary"]["rows"]) == 6
    assert report["full_matches"] == 544
    assert report["distinct_initial_deals"] == 34
    assert report["compound_blocks"] == 17
    for seed in (1, 2):
        for control in ("none", "win", "parent"):
            label = f"seed{seed}/mixture/score-{control}/win"
            assert label in report["primary"]["rows"]
    for field in ("rows", "qualified", "family_size", "primary_mean"):
        bad = copy.deepcopy(report)
        if field == "rows":
            bad["rows"]["seed1/wp/score/points"]["mean"] += 1
        elif field == "qualified":
            bad["qualified"] = not bad["qualified"]
        elif field == "family_size":
            bad["primary"]["family_size"] = 4
        else:
            bad["primary"]["rows"]["seed2/mixture/score-win/win"]["mean"] += 0.1
        with pytest.raises(AssertionError):
            check_statistics(points, bad, 11, 111)


def test_outcome_only_control_cannot_be_omitted_from_primary_success():
    points = np.full((2, 4, 2, 64, 2), -2, dtype=np.int64)
    points[:, 3, :, :48] = 2
    passed = analyse(points, seed=7, replicates=100)
    assert passed["qualified"]
    check_statistics(points, passed, 7, 100)
    points[1, 2] = points[1, 3]
    failed = analyse(points, seed=7, replicates=100)
    assert not failed["qualified"]
    label = "seed2/mixture/score-win/win"
    assert failed["primary"]["rows"][label]["simultaneous95"] == [None, None]
    assert sum(row["sign"] == 1 for row in failed["primary"]["rows"].values()) == 5
    check_statistics(points, failed, 7, 100)


def test_point_gains_do_not_replace_win_gains_and_draws_receive_half_credit():
    points = np.zeros((2, 4, 2, 8, 2), dtype=np.int64)
    points[:, :, :, 4:] = 2
    points[:, 3, :, 4:] = 40
    report = analyse(points, seed=5, replicates=50)
    assert not report["qualified"]
    assert report["rows"]["seed1/mixture/score/win"]["mean"] == 0.75
    assert report["rows"]["seed1/mixture/score-none/points"]["mean"] == 19
    assert report["rows"]["seed1/mixture/score-none/win"]["mean"] == 0
    check_statistics(points, report, 5, 50)


def test_equal_mixture_and_seat_pairing_have_expected_units():
    points = np.zeros((2, 4, 2, 5, 2), dtype=np.int64)
    points[0, 3, 0] = [8, 0]  # WP win score .75, point margin4.
    points[0, 3, 1] = [-2, -6]  # Point win score0, point margin-4.
    report = analyse(points, seed=2, replicates=20)
    assert report["rows"]["seed1/mixture/score/win"]["mean"] == 0.375
    assert report["rows"]["seed1/mixture/score/points"]["mean"] == 0
    assert report["rows"]["seed1/mixture/score-none/win"]["mean"] == -0.125
    assert not report["qualified"]


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 4, 2, 5, 2), dtype=float),
        np.zeros((2, 4, 2, 1, 2), dtype=int),
        np.zeros((1, 4, 2, 5, 2), dtype=int),
    ],
)
def test_incomplete_or_noninteger_results_are_rejected(points):
    with pytest.raises(ValueError):
        analyse(points, seed=2, replicates=20)
