from dataclasses import replace
from fractions import Fraction

import pytest

from koikoi.layout import PHASE, TERMINAL
from scripts.incumbent_tail_controller import evaluate_controller, select_improvement
from scripts.symbolic_final_turn_tail import SymbolicFinalTurnPlanner
from scripts.symbolic_tail_assessment import assess_three, verify_adoption
from tests.test_symbolic_final_turn_tail import dynamics, setup


def prepared(swap=False, cap=1258):
    actor, root, worlds = setup(swap)
    dyn = dynamics(actor, worlds)
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dyn,
        "parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=cap,
        planner_type=SymbolicFinalTurnPlanner,
    )
    env, actions = root.clone(), []
    for _ in range(32):
        if env.state[PHASE] == TERMINAL:
            break
        action = actor.act(env)
        actions.append(action)
        env.step(action)
    assert env.state[PHASE] == TERMINAL
    source = {
        "seat": worlds.viewer,
        "selected_root": {"state": root.state.tolist(), "decision": 0},
        "actions": actions,
        "final": env.state.tolist(),
    }
    phases = {"fixed": fixed, "optimized": optimized}
    if optimized is None:
        phases.pop("optimized")
    return actor, worlds, source, phases, selection, dyn.policy_cache


@pytest.mark.parametrize("swap", [False, True])
def test_adopted_program_pair_is_replayed_and_executes_complete_tails(swap):
    actor, worlds, source, phases, selection, cache = prepared(swap)
    assert selection.parent_q == Fraction(204, 275)
    assert selection.gain == Fraction(53, 550)
    checked = verify_adoption(
        worlds, phases, selection, actor, {"fixture": actor}, cache
    )
    for label, value in (("parent", "204/275"), ("selected", "461/550")):
        assert list(checked[label]["values"].values()) == [value]
        assert checked[label]["represented_work"]["completions"] == sum(worlds.weights)

    class CountingActor:
        def __init__(self):
            self.calls = 0

        def act(self, env):
            self.calls += 1
            return actor.act(env)

    parent_counter, opponent_counter = CountingActor(), CountingActor()
    root_only = select_improvement(phases["fixed"])
    assert root_only.plan.action != selection.plan.action
    result = assess_three(
        source,
        {"root_only": root_only, "controller": selection},
        parent_counter,
        opponent_counter,
        "parent",
    )
    assert result["actual_work"] == result["replay_work"]
    assert result["actual_work"]["completions"] == 3
    assert (
        result["outcomes"]["controller"]["actions"][0]
        != result["outcomes"]["root_only"]["actions"][0]
    )
    assert result["calls"]["symbolic_parent_adapter_calls"] > 0
    assert result["calls"]["parent_adapter_calls"] == parent_counter.calls
    assert result["calls"]["opponent_adapter_calls"] == opponent_counter.calls
    assert (
        result["calls"]["parent_adapter_calls"]
        > result["calls"]["symbolic_parent_adapter_calls"]
    )
    with pytest.raises(ValueError, match="Selection"):
        verify_adoption(
            worlds,
            phases,
            replace(selection, selected_q=Fraction(1)),
            actor,
            {"fixture": actor},
            cache,
        )


def test_unvalued_parent_retains_the_identical_source_tail():
    actor, worlds, source, phases, selection, _ = prepared(cap=1)
    assert selection.plan is None and selection.reason == "parent_unvalued"
    assert verify_adoption(worlds, phases, selection, actor, {"fixture": actor}) == {}
    result = assess_three(
        source,
        {"root_only": selection, "controller": selection},
        actor,
        actor,
        "parent",
    )
    assert all(v == result["outcomes"]["parent"] for v in result["outcomes"].values())
    assert result["calls"].get("symbolic_parent_adapter_calls", 0) == 0


def test_conditional_assessment_rejects_wrong_arm_set_and_root_binding():
    actor, _, source, _, selection, _ = prepared()
    with pytest.raises(ValueError, match="two planned"):
        assess_three(source, {"controller": selection}, actor, actor, "parent")
    changed = replace(
        selection, binding=replace(selection.binding, digest="different belief")
    )
    with pytest.raises(ValueError, match="same root"):
        assess_three(
            source,
            {"controller": selection, "root_only": changed},
            actor,
            actor,
            "parent",
        )
