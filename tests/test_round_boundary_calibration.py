"""Independent fixtures for public continuation-value diagnostics."""

import numpy as np

from scripts.calibrate_round_boundaries import (
    RIDGE,
    cluster_interval,
    extract,
    features,
    fit_logistic,
    sigmoid,
)


def test_boundary_extraction_tracks_points_and_player_signs():
    # One two-round game and one game ending immediately; terminal padding ignored.
    data = {
        "valid": np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
        "round": np.array([[0, 0], [0, 0], [1, 0]]),
        "phase": np.array([[0, 0], [3, 4], [0, 4]]),
        "player": np.array([[0, 1], [0, 1], [1, 1]]),
        "value": np.array([[0.4, -0.2], [0.3, 0.0], [0.6, 0.0]]),
        "result": np.zeros((3, 2, 4), dtype=int),
        "final": np.zeros((2, 208), dtype=int),
    }
    data["result"][0, 1] = [-1, 1, 1, -30]
    data["result"][1, 0] = [0, 1, 0, -6]
    data["result"][2, 0] = [1, 1, 1, 12]
    data["final"][:, 106:108] = [[36, 24], [0, 60]]
    expected = np.array(
        [
            [100, 0, 30, 1, 1, 0.7, 2, 0, 0],
            [101, 0, 30, 0, 0, 0.6, 2, 0, 1],
            [100, 1, 24, 0, 1, 0.2, 2, 2, 0],
        ]
    )
    np.testing.assert_allclose(extract(data, 100, 2), expected, atol=1e-15)
    data["final"][0, 106:108] = 30
    assert np.all(extract(data, 100, 2)[[0, 2], 4] == 0.5)


def test_features_exclude_labels_values_and_identifiers_and_swap_seats():
    rows = np.array([[17, 4, 18, 1, 0, 0.8, 0, 63, 7]], dtype=float)
    before = features(rows)
    rows[:, [0, 1, 4, 5, 6, 7, 8]] = -123
    np.testing.assert_array_equal(features(rows), before)
    rows[:, 2] = 60 - rows[:, 2]
    rows[:, 3] = 1 - rows[:, 3]
    np.testing.assert_array_equal(features(rows), -before)
    weights = np.array([2.7, -0.3])
    np.testing.assert_allclose(
        sigmoid(before @ weights) + sigmoid(features(rows) @ weights), 1
    )


def test_fractional_labels_have_known_regularized_optimum():
    # Choose labels so the independent first-order optimum is exactly beta.
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    beta = np.array([0.4, -0.7])
    y = 1 / (1 + np.exp(-beta)) + 2 * RIDGE * beta
    actual, info = fit_logistic(x, y)
    np.testing.assert_allclose(actual, beta, atol=1e-8, rtol=0)
    assert info["gradient_max"] < 1e-10


def test_round_zero_unidentified_score_slope_stays_zero():
    x = np.array([[0, 1], [0, -1], [0, 1], [0, -1]], dtype=float)
    weights, _ = fit_logistic(x, np.full(4, 0.5))
    np.testing.assert_array_equal(weights, [0, 0])


def test_bootstrap_keeps_whole_matches_and_weights_boundaries():
    rows = np.zeros((5, 9))
    rows[:, 0] = [10, 10, 10, 20, 20]
    # Each complete match has zero sum; any cluster resampling must stay zero.
    assert cluster_interval(rows, np.array([0.5, -0.2, -0.3, 0.4, -0.4])) == [0, 0]
