import math

import numpy as np
import pytest

from scripts.audit_koi_temperature_mechanism import (
    equal_scalars,
    independent_phase_report,
    scalar_statistics,
)


def test_independent_scalar_entropy_and_forced_exclusion():
    scores = np.full((3, 50), -np.inf)
    scores[:2, 48] = 0
    scores[:2, 49] = 0.05 * math.log(3)
    scores[2, 48] = 1
    mask = np.isfinite(scores)
    result = independent_phase_report(
        scores,
        mask,
        np.array([3, 3, 3]),
        np.array([49, 49, -1]),
        np.array([48, 49, -1]),
        0.05,
    )["3"]
    entropy = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    assert result["all_rows"] == 3 and result["unforced_rows"] == 2
    assert result["changed_greedy_actions"] == result["stop_to_continue"] == 1
    equal_scalars(
        result["common_temperature"],
        {
            "rows": 2,
            "mean_entropy": entropy,
            "mean_normalized_entropy": entropy / math.log(2),
            "mean_greedy_probability": 0.75,
            "median_greedy_probability": 0.75,
        },
    )
    with pytest.raises(AssertionError):
        equal_scalars({"a": 0.5}, {"a": 0.6})
    with pytest.raises(ValueError):
        scalar_statistics(scores, mask, 0.05)
    with pytest.raises(ValueError):
        scalar_statistics(scores[:2], mask[:2], float("nan"))


def test_multiple_legal_actions_extreme_gaps_and_masked_nan():
    scores = np.array(
        [[math.log(2), 0, 0, float("nan")], [10000, -10000, -10000, float("nan")]]
    )
    mask = np.array([[True, True, True, False]] * 2)
    result = scalar_statistics(scores, mask, 1)
    entropy = -0.5 * math.log(0.5) - 0.5 * math.log(0.25)
    assert result["rows"] == 2
    assert result["mean_entropy"] == pytest.approx(entropy / 2)
    assert result["mean_normalized_entropy"] == pytest.approx(
        entropy / (2 * math.log(3))
    )
    assert (
        result["mean_greedy_probability"] == result["median_greedy_probability"] == 0.75
    )
