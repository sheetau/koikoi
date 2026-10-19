import math

import numpy as np
import pytest

from scripts.phase_exploration_calibration import (
    calibrate_temperature,
    choice_statistics,
)


def test_calibration_matches_independent_binary_entropy():
    scores, mask = (
        np.array([[0.0, math.log(3), 999.0]]),
        np.array([[True, True, False]]),
    )
    target = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(2)
    t = calibrate_temperature(scores, mask, target)
    assert t == pytest.approx(1, abs=1e-12)
    assert choice_statistics(scores, mask, t)[
        "mean_greedy_probability"
    ] == pytest.approx(0.75)
    assert calibrate_temperature(scores * 2 + 17, mask, target) == pytest.approx(
        2, abs=1e-12
    )


def test_forced_or_unattainable_entropy_is_rejected():
    with pytest.raises(ValueError, match="unforced"):
        choice_statistics([[0, 1]], np.array([[True, False]]), 0.05)
    with pytest.raises(ValueError, match="attainable"):
        calibrate_temperature([[0, 0]], np.array([[True, True]]), 0.5)
