import numpy as np
import pytest

from scripts.boundary_allocation_diagnostic import optimal_independent_control


@pytest.mark.parametrize("covariance", [-0.9, 0.1, 0.9])
def test_allocation_closed_form_matches_independent_dense_grid(covariance):
    result = optimal_independent_control(1, 1, covariance, 0.05, 1)
    grid = np.linspace(-2, 2, 100001)
    objective = (
        np.abs(grid) * np.sqrt(0.05) + np.sqrt(1 - 2 * grid * covariance + grid**2)
    ) ** 2
    assert result["variance_cost_ratio"] == pytest.approx(
        float(objective.min()), abs=1e-8
    )
    if abs(covariance) == 0.1:
        assert result["alpha"] == 0
    else:
        assert np.sign(result["alpha"]) == np.sign(covariance)


def test_perfect_and_invalid_covariance():
    perfect = optimal_independent_control(1, 1, 1, 0.1, 1)
    assert perfect["alpha"] == 1 and perfect["cheap_per_high"] is None
    assert perfect["variance_cost_ratio"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        optimal_independent_control(1, 1, 2, 0.1, 1)
