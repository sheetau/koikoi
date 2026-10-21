from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.proposal_interval_tail import compile_proposal, propose_then_compare
from scripts.verify_interval_tail import verify_receipt
from tests.test_symbolic_final_turn_tail import dynamics, setup


def test_contingent_proposal_is_adopted_with_an_incomplete_true_parent():
    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    result = propose_then_compare(
        worlds,
        dyn,
        "parent",
        {"fixture": "fixture"},
        proposal_decisions=10000,
        reference_decisions=500,
    )
    selection = result["selection"]
    assert selection.candidate is not None
    assert selection.parent.lower < selection.parent.upper
    assert selection.candidate.lower == Fraction(97, 110)
    assert selection.gain_lower == Fraction(14, 275)
    assert len(selection.candidate.program.overrides) > 1
    assert result["decisions"] <= 10500
    for receipt in (selection.parent, selection.candidate):
        verify_receipt(worlds, receipt, actor, {"fixture": actor}, "parent")


def test_no_partial_proposal_is_misreported_as_a_complete_policy_value():
    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    result = propose_then_compare(
        worlds,
        dyn,
        "parent",
        {"fixture": "fixture"},
        proposal_decisions=1,
        reference_decisions=100,
    )
    assert result["proposal_phase"].result.plan is None
    assert result["candidate_certificate"] is None
    assert result["selection"].candidate is None
    assert result["parent_receipt"].work["decisions"] == 0
    assert result["decisions"] <= 1


def test_compiler_rejects_a_different_belief_mode_or_leaf_rule():
    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    result = propose_then_compare(
        worlds,
        dyn,
        "parent",
        {"fixture": "fixture"},
        proposal_decisions=10000,
        reference_decisions=1,
    )
    phase = deepcopy(result["proposal_phase"])
    phase.mode = "fixed"
    with pytest.raises(ValueError, match="optimized proposal"):
        compile_proposal(phase, worlds, dyn, "parent", {"fixture": "fixture"})
    phase = deepcopy(result["proposal_phase"])
    phase.result.plan.leaf_rule = "unknown"
    with pytest.raises(ValueError, match="default-domain"):
        compile_proposal(phase, worlds, dyn, "parent", {"fixture": "fixture"})
    with pytest.raises(ValueError, match="same complete belief"):
        compile_proposal(
            result["proposal_phase"], worlds, dyn, "another", {"fixture": "fixture"}
        )
