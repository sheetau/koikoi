import copy

import pytest

from scripts.audit_prepared_belief_confirmation import verify_condition
from tests.test_audit_prepared_belief_actions import estimate


def test_fresh_sample_accepts_unchanged_candidate_only():
    previous = {
        "decision": {"alternative": 2, "proposal": {"win_change": 0.1}},
        "sampling": {"proposal": {"worlds_sha256": "old"}},
    }
    condition = {
        "confirmed": True,
        "confirmation": estimate(512, [-2, 3]),
        "action": 2,
        "sampling": {"model_indices": [0] * 512, "worlds_sha256": "fresh"},
    }
    verify_condition(condition, previous, 1)
    changed = copy.deepcopy(condition)
    changed["confirmation"]["alternative"] = 3
    with pytest.raises(AssertionError):
        verify_condition(changed, previous, 1)
    changed = copy.deepcopy(condition)
    changed["sampling"]["worlds_sha256"] = "old"
    with pytest.raises(AssertionError):
        verify_condition(changed, previous, 1)


def test_nonpositive_proposal_skips_rollout_and_keeps_parent():
    previous = {"decision": {"alternative": 2, "proposal": {"win_change": 0.0}}}
    condition = {
        "confirmed": False,
        "confirmation": None,
        "sampling": None,
        "action": 1,
    }
    verify_condition(condition, previous, 1)
    condition["action"] = 2
    with pytest.raises(AssertionError):
        verify_condition(condition, previous, 1)
