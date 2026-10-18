import numpy as np
import pytest

from scripts.audit_fixed_midround_terminal import verify_uniform_model_posterior


def test_many_normalized_terms_need_not_sum_to_integer_count_ratio():
    compatible = np.zeros((4844, 2), dtype=bool)
    compatible[:2324, 1] = True
    saved = (compatible.astype(np.float64) / 2324).sum(axis=0)
    assert abs(saved[1] - 1.0) > 2e-14
    verify_uniform_model_posterior(saved.tolist(), compatible)
    with pytest.raises(AssertionError):
        verify_uniform_model_posterior([0.0, 1.0], compatible)
    corrupted = saved.copy()
    corrupted[1] = np.nextafter(corrupted[1], np.inf)
    with pytest.raises(AssertionError):
        verify_uniform_model_posterior(corrupted, compatible)


def test_both_model_columns_are_verified_and_empty_support_rejected():
    compatible = np.array([[1, 1], [1, 0], [0, 1], [0, 1]], dtype=bool)
    saved = (compatible.astype(np.float64) / 5).sum(axis=0)
    verify_uniform_model_posterior(saved, compatible)
    with pytest.raises(AssertionError):
        verify_uniform_model_posterior(saved[::-1], compatible)
    with pytest.raises(AssertionError):
        verify_uniform_model_posterior([0.0, 0.0], np.zeros((3, 2), dtype=bool))
