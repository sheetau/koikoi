from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.deferred_belief_tail import DeferredWorldSet, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_tail_controller import (
    ContingentExecutor,
    TailSelection,
    bind_worlds,
    evaluate_controller,
    saved_phase,
    select_improvement,
)
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root
from tests.test_incumbent_belief_tail import ImprovingGame, IncumbentToy


def toy_setup():
    return ImprovingGame(False), DeferredWorldSet(
        np.array([[0, w, 0] for w in range(4)], dtype=np.int32),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.int64),
        ("unused",),
        0,
        0,
    )


def test_controller_adopts_exact_feasible_gain_after_caps():
    dynamics, worlds = toy_setup()
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dynamics,
        "parent",
        {"unused": "opponent"},
        max_decisions_per_phase=13,
        planner_type=IncumbentToy,
    )
    assert fixed.result.refusal and optimized.result.refusal
    assert selection.mode == "fixed"  # Equal exact candidates prefer fixed.
    assert selection.parent_q == Fraction(1, 4)
    assert selection.selected_q == Fraction(3, 4) and selection.gain == Fraction(1, 2)
    assert selection.plan.optimality_gap == Fraction(1, 4)
    assert sum(p.result.work["decisions"] for p in (fixed, optimized)) == 26
    assert selection.plan is fixed.result.plan
    # A future-optimized parent action must never serve as the parent reference.
    with pytest.raises(ValueError, match="unchanged parent"):
        select_improvement(optimized)
    with pytest.raises(ValueError, match="mode"):
        select_improvement(replace(fixed, result=optimized.result))


def test_controller_rejects_context_mismatch_and_unvalued_parent():
    dynamics, worlds = toy_setup()
    fixed, optimized, _ = evaluate_controller(
        worlds,
        dynamics,
        "parent",
        {"unused": "opponent"},
        max_decisions_per_phase=8,
        planner_type=IncumbentToy,
    )
    changed_worlds = replace(worlds, weights=worlds.weights * 2)
    variants = (
        bind_worlds(changed_worlds, dynamics, "parent", {"unused": "opponent"}),
        bind_worlds(worlds, dynamics, "different-parent", {"unused": "opponent"}),
        bind_worlds(worlds, dynamics, "parent", {"unused": "different-opponent"}),
    )
    for binding in variants:
        with pytest.raises(ValueError, match="exact evaluation context"):
            select_improvement(fixed, replace(optimized, binding=binding))
    empty, skipped, selection = evaluate_controller(
        worlds,
        dynamics,
        "parent",
        {"unused": "opponent"},
        max_decisions_per_phase=1,
        planner_type=IncumbentToy,
    )
    assert empty.result.plan is None and skipped is None
    assert selection.plan is None and selection.reason == "parent_unvalued"
    assert selection.parent_q is selection.gain is None


@pytest.mark.parametrize("swap", [False, True])
def test_native_schedule_fallback_and_contingent_execution(swap):
    torch.set_num_threads(2)
    root, actor = late_root(), Extremal()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer] = 200
    root.state[POINTS + 1 - viewer] = 30
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support)
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dynamics,
        "parent",
        {"first": "opponent"},
    )
    assert selection.parent_q == 1 and selection.gain == 0
    assert optimized is None  # Ceiling in fixed phase avoids unnecessary work.
    assert selection.plan is None and selection.reason == "no_strict_gain"
    fallback = ContingentExecutor(selection, actor, "parent")
    assert fallback.act(root) == actor.act(root)
    # Exercise execution of a real, complete native policy independently of the
    # adoption test above. The synthetic reference value is only a routing fixture.
    adopted = TailSelection(
        selection.binding,
        "fixed",
        Fraction(0),
        Fraction(1),
        fixed.result.plan,
        "strict_exact_gain",
    )
    executor = ContingentExecutor(adopted, actor, "parent")
    env = root.clone()
    while env.state[PHASE] != TERMINAL:
        if env.state[PLAYER] == viewer:
            action = executor.act(env)
            key = visible_keys(env.state[None], viewer)[0]
            assert action == adopted.plan.policy[key]
        else:
            action = actor.act(env)
        env.step(action)
    with pytest.raises(ValueError, match="live"):
        executor.act(env)
    with pytest.raises(ValueError, match="identity"):
        ContingentExecutor(adopted, actor, "wrong")
    broken = ContingentExecutor(adopted, actor, "parent")
    broken.policy.clear()
    with pytest.raises(KeyError, match="no action"):
        broken.act(root)
    other = root.clone()
    other.state[POINTS + viewer] += 1
    with pytest.raises(ValueError, match="root"):
        ContingentExecutor(adopted, actor, "parent").act(other)
    restored = saved_phase(selection.binding, "fixed", fixed.result.plan)
    assert select_improvement(restored).reason == "no_strict_gain"
