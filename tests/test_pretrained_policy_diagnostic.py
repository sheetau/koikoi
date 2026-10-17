"""Hand-computed probability and mask fixtures for the saved-state diagnostic."""

import math

import numpy as np
import pytest

from scripts.audit_pretrained_policy import recompute
from scripts.audit_pretrained_ties import scalar_probabilities, vector_probabilities
from scripts.diagnose_pretrained_policy import describe


@pytest.mark.parametrize("summarize", [describe, recompute])
def test_two_action_probabilities_and_policy_change(summarize):
    scores = np.array([[math.log(3), 0, 100], [0, math.log(3), 100]])
    mask = np.array([[True, True, False]] * 2)
    original = np.array([0, 0])
    result = summarize(scores, mask, original, 1)
    assert result["rows"] == 2 and result["changed_greedy_actions"] == 1
    assert result["mean_greedy_probability"] == pytest.approx(0.75)
    assert result["mean_initial_greedy_probability"] == pytest.approx(0.5)
    assert result["mean_entropy"] == pytest.approx(
        -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    )
    assert result["mean_top_two_raw_score_gap"] == pytest.approx(math.log(3))
    colder = summarize(scores, mask, original, 0.5)
    assert colder["mean_greedy_probability"] == pytest.approx(0.9)
    assert colder["changed_greedy_actions"] == 1


@pytest.mark.parametrize("summarize", [describe, recompute])
def test_koi_direction_counts_and_score_offset_invariance(summarize):
    scores = np.full((3, 50), -np.inf)
    scores[:, 48:] = [[0, 1], [1, 0], [0, 0]]
    mask = np.isfinite(scores)
    result = summarize(scores, mask, np.array([48, 49, 48]), 0.05)
    assert result["initial_stop_to_continue"] == 1
    assert result["initial_continue_to_stop"] == 1
    assert result["changed_greedy_actions"] == 2
    shifted = summarize(scores + 100, mask, np.array([48, 49, 48]), 0.05)
    for key in result:
        np.testing.assert_allclose(result[key], shifted[key], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "fault", ["forced", "nonfinite", "temperature", "illegal_reference"]
)
@pytest.mark.parametrize("summarize", [describe, recompute])
def test_rejects_invalid_statistical_rows(fault, summarize):
    scores, mask, reference, temperature = (
        np.array([[1.0, 0.0]]),
        np.array([[True, True]]),
        np.array([0]),
        0.05,
    )
    if fault == "forced":
        mask[0, 1] = False
    elif fault == "nonfinite":
        scores[0, 1] = np.nan
    elif fault == "temperature":
        temperature = 0
    else:
        scores = np.array([[1.0, 0.0, 5.0]])
        mask = np.array([[True, True, False]])
        reference = np.array([2])
    with pytest.raises(AssertionError):
        summarize(scores, mask, reference, temperature)


@pytest.mark.parametrize("calculate", [scalar_probabilities, vector_probabilities])
def test_exact_tie_mass_excludes_arbitrary_argmax_choice(calculate):
    scores = np.array([[0, 0, -math.log(2), 100], [math.log(2), 0, -np.inf, 100]])
    mask = np.array([[True, True, True, False], [True, True, False, False]])
    single, whole_set, count = calculate(scores, mask, 1)
    np.testing.assert_allclose(single, [0.4, 2 / 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(whole_set, [0.8, 2 / 3], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(count, [2, 1])
