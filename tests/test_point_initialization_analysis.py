import numpy as np
import pytest

from scripts.audit_point_initialization_evaluation import check_statistics
from scripts.point_initialization_analysis import analyse


def test_primary_resampling_preserves_every_seat_opponent_and_parent():
    p = np.random.default_rng(44).integers(-5, 6, size=(2, 4, 2, 7, 2))
    result = analyse(p, seed=37, replicates=213)
    primary = []
    for parent in range(2):
        for other in (1, 0):
            values = []
            for deal in range(7):
                delta = 0.0
                for opponent in range(2):
                    for seat in range(2):

                        def score(x):
                            return 1.0 if x > 0 else 0.0 if x < 0 else 0.5

                        delta += (
                            score(p[parent, 3, opponent, deal, seat])
                            - score(p[parent, other, opponent, deal, seat])
                        ) / 4
                values.append(delta)
            primary.append(values)
    matrix = np.asarray(primary).T
    rng = np.random.default_rng(37)
    se = matrix.std(0, ddof=1) / np.sqrt(7)
    maxima = []
    sampled = []
    for _ in range(213):
        count = rng.multinomial(7, np.full(7, 1 / 7))
        sample = np.repeat(matrix, count, axis=0).mean(0)
        sampled.append(sample)
        maxima.append(max(abs((sample - matrix.mean(0)) / se)))
    critical = sorted(maxima)[int(np.ceil(0.95 * 212))]
    assert result["primary"]["critical"] == pytest.approx(critical)
    for col, (label, row) in enumerate(result["primary"]["rows"].items()):
        np.testing.assert_allclose(
            row["simultaneous95"],
            matrix[:, col].mean() + np.array([-1, 1]) * se[col] * critical,
        )
        np.testing.assert_allclose(
            result["rows"][label]["marginal95"],
            np.quantile(np.array(sampled)[:, col], [0.025, 0.975]),
        )
    check_statistics(p, result, 37, 213)
    assert len(result["rows"]) == 120 and result["full_matches"] == 224


def test_ties_constant_contrasts_and_single_seed_gain_do_not_qualify():
    p = np.zeros((2, 4, 2, 8, 2), dtype=int)
    result = analyse(p, replicates=101)
    assert not result["qualified"]
    assert result["rows"]["parent1/mixture/point_matched/win"]["mean"] == 0.5
    assert all(
        row["simultaneous95"] == [None, None]
        for row in result["primary"]["rows"].values()
    )
    p[0, 3, :, :7] = 1
    assert not analyse(p, replicates=101)["qualified"]
    p[1, 3, :, :7] = 1
    assert analyse(p, replicates=101)["qualified"]
    with pytest.raises(ValueError):
        analyse(p.astype(float))
