"""Hand-calculated signed win boundaries for the independent search auditor."""

import copy

import pytest

from scripts.audit_prepared_belief_actions import check_choice, check_estimate


def estimate(n, points, alternative=2):
    return {
        "paired_points": [points] * n,
        "win_change": 1.0,
        "lower_margin": 1.0,
        "point_change": points[1] - points[0],
        "alternative": alternative,
        "work": {
            "decisions": 2 * n,
            "round_endings": 2 * n,
            "completed_branches": 2 * n,
        },
    }


def test_independent_win_and_point_arithmetic():
    row = estimate(2, [-3, 8])
    assert check_estimate(row, 2) == (1, 1)
    row["win_change"] = 11
    with pytest.raises(AssertionError, match="win_change"):
        check_estimate(row, 2)
    row = estimate(2, [0, 8])
    row["win_change"] = row["lower_margin"] = 0.5
    assert check_estimate(row, 2) == (0.5, 0.5)


def test_confirmation_selects_only_proposed_alternative():
    proposal, confirmation = estimate(128, [-3, 8]), estimate(512, [-3, 8])
    decision = {
        "base": 1,
        "alternative": 2,
        "alternatives": [proposal],
        "proposal": proposal,
        "confirmation": confirmation,
        "changed": True,
    }
    row = {"action": 2, "decision": decision}
    check_choice(row, [1, 2], 1)
    bad = copy.deepcopy(row)
    bad["decision"]["confirmation"]["alternative"] = 3
    with pytest.raises(AssertionError):
        check_choice(bad, [1, 2], 1)
    bad = copy.deepcopy(row)
    bad["action"] = 1
    with pytest.raises(AssertionError):
        check_choice(bad, [1, 2], 1)
