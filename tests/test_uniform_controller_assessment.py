from dataclasses import replace
from fractions import Fraction

import pytest

from scripts.uniform_controller_assessment import assess_pair, verify_pair
from tests.test_symbolic_tail_assessment import prepared


@pytest.mark.parametrize("swap", [False, True])
def test_true_parent_and_symbolic_program_values_and_saved_execution(swap):
    actor, worlds, source, _, selection, _ = prepared(swap)
    result = verify_pair(worlds, selection, actor, {"fixture": actor})
    assert result["parent"]["value"] == "204/275"
    assert result["selected"]["value"] == "461/550"
    for checked in result.values():
        assert checked["represented_work"]["completions"] == sum(worlds.weights)
    assert result["parent"]["new_actor_calls"]["parent"] > 0
    # This three-draw fixture has only forced opponent decisions.
    assert result["parent"]["new_actor_calls"]["opponent"] == 0

    class CounterActor:
        def __init__(self):
            self.calls = 0

        def act(self, env):
            self.calls += 1
            return actor.act(env)

    parent, opponent = CounterActor(), CounterActor()
    execution = assess_pair(source, selection, parent, opponent, "parent")
    assert execution["actual_work"] == execution["replay_work"]
    assert execution["actual_work"]["completions"] == 2
    assert execution["calls"]["parent_adapter_calls"] == parent.calls
    assert execution["calls"]["opponent_adapter_calls"] == opponent.calls
    assert execution["calls"]["symbolic_parent_adapter_calls"] > 0
    with pytest.raises(ValueError, match="parent value"):
        verify_pair(
            worlds, replace(selection, parent_q=Fraction(1)), actor, {"fixture": actor}
        )
    with pytest.raises(ValueError, match="selected value"):
        verify_pair(
            worlds,
            replace(selection, selected_q=Fraction(0)),
            actor,
            {"fixture": actor},
        )


def test_fallback_execution_and_mismatched_root_identity_rejected():
    actor, worlds, source, _, selection, _ = prepared(cap=1)
    assert verify_pair(worlds, selection, actor, {"fixture": actor}) == {}
    result = assess_pair(source, selection, actor, actor, "parent")
    assert result["outcomes"]["parent"] == result["outcomes"]["controller"]
    with pytest.raises(ValueError, match="identity"):
        assess_pair(source, selection, actor, actor, "wrong parent")
    changed = replace(
        selection, binding=replace(selection.binding, root_key=b"other root")
    )
    with pytest.raises(ValueError, match="another root"):
        assess_pair(source, changed, actor, actor, "parent")
    with pytest.raises(ValueError, match="recorded source"):
        assess_pair({**source, "actions": []}, selection, actor, actor, "parent")


def test_candidate_missing_noncertified_key_is_not_silent_parent_fallback():
    actor, worlds, _, _, selection, _ = prepared()
    altered = replace(selection, plan=replace(selection.plan, policy={}))
    with pytest.raises(KeyError, match="Missing candidate"):
        verify_pair(worlds, altered, actor, {"fixture": actor})
    with pytest.raises(ValueError, match="declared opponent"):
        verify_pair(worlds, selection, actor, {})
