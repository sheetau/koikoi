from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.frontier_hybrid_controller import (
    FrontierHybridExecutor,
    evaluate_frontier_hybrid,
    validate_hybrid,
)
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.symbolic_final_turn_tail import SymbolicExecutor, SymbolicFinalTurnPlanner
from scripts.verify_frontier_patch import verify_frontier_patch
from tests.test_symbolic_final_turn_tail import dynamics, setup


def test_actual_first_phase_refusal_recovers_a_proved_global_gain():
    actor, _, worlds = setup()
    result = evaluate_frontier_hybrid(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        phase_decisions=500,
        patch_draws=2,
        prefix_decisions=400,
        local_phase_decisions=200,
    )
    assert result.original.reason == "parent_unvalued"
    assert result.fixed.result.plan is None and result.optimized is None
    assert result.fixed.result.work["decisions"] == 465
    assert result.patch.decisions == 495 and result.decisions == 960
    assert result.gain_lower == Fraction(7, 550)
    report = verify_frontier_patch(
        worlds, result.patch, actor, {"fixture": actor}, "parent", full_values=True
    )
    assert Fraction(report["exact_increment"]) >= result.gain_lower


@pytest.mark.parametrize("margin", [-40, 6, 40])
def test_completed_original_decisions_and_execution_are_identical(margin):
    actor, root, worlds = setup(margin=margin)
    old = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=10000,
        planner_type=SymbolicFinalTurnPlanner,
    )
    new = evaluate_frontier_hybrid(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        phase_decisions=10000,
    )
    assert new.patch is None and (new.fixed, new.optimized, new.original) == old
    assert new.decisions == sum(
        p.result.work["decisions"] for p in old[:2] if p is not None
    )
    assert new.gain_lower == (new.original.gain or 0)
    original = SymbolicExecutor(old[2], actor, "parent")
    hybrid = FrontierHybridExecutor(new, actor, "parent")
    env = root.clone()
    while env.state[PHASE] != TERMINAL:
        if env.state[PLAYER] == worlds.viewer:
            action = original.act(env)
            assert action == hybrid.act(env)
        else:
            action = actor.act(env)
        env.step(action)


def test_prefix_refusal_keeps_parent_and_both_spent_phases_count():
    actor, root, worlds = setup()
    result = evaluate_frontier_hybrid(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        phase_decisions=500,
        patch_draws=2,
        prefix_decisions=1,
    )
    assert result.patch is not None and not result.patch.frontier.complete
    assert result.gain_lower == 0 and result.decisions <= 1000
    executor = FrontierHybridExecutor(result, actor, "parent")
    env = root.clone()
    while env.state[PHASE] != TERMINAL:
        action = actor.act(env)
        if env.state[PLAYER] == worlds.viewer:
            assert executor.act(env) == action
        env.step(action)


def test_hybrid_rejects_extra_work_different_belief_and_replaced_original():
    actor, _, worlds = setup()
    result = evaluate_frontier_hybrid(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        phase_decisions=500,
        patch_draws=2,
        prefix_decisions=400,
    )
    with pytest.raises(ValueError, match="Integrated work"):
        validate_hybrid(replace(result, decisions=result.decisions + 1))
    broken = deepcopy(result)
    broken.patch.frontier.parent_program = replace(
        broken.patch.frontier.parent_program,
        binding=replace(broken.patch.frontier.parent_program.binding, digest="wrong"),
    )
    with pytest.raises(ValueError, match="same parent refusal"):
        validate_hybrid(broken)
    broken = deepcopy(result)
    broken.original.reason = "no_strict_gain"
    with pytest.raises(ValueError, match="Original controller selection"):
        validate_hybrid(broken)
