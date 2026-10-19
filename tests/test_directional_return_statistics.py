import numpy as np
import pytest

from scripts.directional_return_statistics import coordinates, simultaneous_bands


def test_simultaneous_bands_match_independent_resamples():
    x = np.array([[1.0, 2.0, 7.0], [2.0, 5.0, 7.0], [8.0, -2.0, 7.0], [-3.0, 1.0, 7.0]])
    result = simultaneous_bands(x, replicates=317, seed=19)
    rng = np.random.default_rng(19)
    se = x[:, :2].std(0, ddof=1) / 2
    maxima = []
    for _ in range(317):
        frequency = rng.multinomial(4, [0.25] * 4)
        sampled = np.repeat(x[:, :2], frequency, axis=0)
        maxima.append(max(abs((sampled.mean(0) - x[:, :2].mean(0)) / se)))
    critical = sorted(maxima)[int(np.ceil(0.95 * 316))]
    assert result["critical"] == pytest.approx(critical)
    for i in range(2):
        np.testing.assert_allclose(
            result["rows"][i]["simultaneous95"],
            x[:, i].mean() + np.array([-1, 1]) * critical * se[i],
        )
    assert result["rows"][2]["simultaneous95"] == [None, None]
    assert result["rows"][2]["sign"] == 0


def test_coordinates_keep_both_seats_and_parents_in_deal_block():
    x = np.arange(2 * 3 * 4 * 2 * 5).reshape(2, 3, 4, 2, 5)
    labels, y = coordinates(x)
    assert y.shape == (3, 70)
    np.testing.assert_array_equal(
        y[:, labels.index("parent2/koi_continue/mc")],
        (x[1, :, 0, 0, 0] + x[1, :, 0, 1, 0]) / 2,
    )
    np.testing.assert_array_equal(
        y[:, labels.index("parent1/koi_preference/gae1_window-mc")],
        x[0, :, 3, :, 4].mean(1) - x[0, :, 0, :, 4].mean(1),
    )
    with pytest.raises(ValueError):
        coordinates(x[:, :1])
