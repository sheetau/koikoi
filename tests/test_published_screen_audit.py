"""Independent auditor fixtures; no model inference or CUDA work."""

import copy

import numpy as np
import pytest

from scripts.audit_published_screen import interval, raw_pairs, statistics


def report():
    return {
        "pairs": 2,
        "records": [
            {
                "seed": 1000001,
                "seat0_score": 0,
                "seat1_score": 0,
                "seat0_point_difference": -4,
                "seat1_point_difference": -2,
            },
            {
                "seed": 1000000,
                "seat0_score": 1,
                "seat1_score": 1,
                "seat0_point_difference": 2,
                "seat1_point_difference": 6,
            },
        ],
    }


def test_bootstrap_known_two_pair_endpoints_and_zero_difference():
    score, points = raw_pairs(report(), 2)
    np.testing.assert_array_equal(score, [1, 0])
    np.testing.assert_array_equal(points, [4, -3])
    assert statistics(score, points) == {
        "score_mean": 0.5,
        "score_ci95": [0, 1],
        "point_mean": 0.5,
        "point_ci95": [-3, 4],
    }
    assert interval(score - score) == [0, 0]
    assert interval(np.array([0.25] * 5)) == [0.25, 0.25]


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "outcome", "odd_points", "nan"]
)
def test_invalid_raw_match_records_rejected(fault):
    value = copy.deepcopy(report())
    if fault == "duplicate":
        value["records"][0]["seed"] = 1000000
    elif fault == "missing":
        value["records"].pop()
    elif fault == "outcome":
        value["records"][0]["seat0_score"] = 1
    elif fault == "odd_points":
        value["records"][0]["seat0_point_difference"] = -3
    else:
        value["records"][0]["seat0_point_difference"] = float("nan")
    with pytest.raises(AssertionError):
        raw_pairs(value, 2)
