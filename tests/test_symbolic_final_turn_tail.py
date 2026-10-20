from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.deferred_belief_tail import deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner
from scripts.incumbent_tail_controller import (
    EvaluatedPhase,
    bind_worlds,
    evaluate_controller,
    select_improvement,
)
from scripts.symbolic_final_turn_tail import (
    RULE,
    SymbolicExecutor,
    SymbolicFinalTurnPlanner,
    final_own_turn_win,
)
from scripts.verify_symbolic_tail import verify_program
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


def setup(swap=False, margin=6):
    torch.set_num_threads(2)
    actor, root = Extremal(), late_root()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer] = 30 + margin // 2
    root.state[POINTS + 1 - viewer] = 30 - margin // 2
    support = enumerate_support(root, {"fixture": actor}, max_hands=1000)
    worlds = deferred_worlds(root, support)
    return actor, root, worlds


def dynamics(actor, worlds):
    return KoiTailDynamics(4096, worlds.model_names, {"fixture": actor}, actor)


def test_win_region_is_public_strict_and_confined_to_own_last_turn():
    _, root, worlds = setup()
    viewer = worlds.viewer
    states = np.repeat(root.state[None], 7, axis=0)
    states[:, TURN] = 15
    states[1, POINTS + viewer] = states[1, POINTS + 1 - viewer] + 2
    states[2, TURN] = 14
    states[3, ROUND] = 6
    states[4, PLAYER] = 1 - viewer
    states[5, PHASE] = TERMINAL
    states[6, POINTS + viewer] = states[6, POINTS + 1 - viewer] - 6
    expected = [True, False, False, False, False, False, False]
    np.testing.assert_array_equal(final_own_turn_win(states, viewer), expected)
    states[:, :96] = states[:, :96][:, ::-1]
    np.testing.assert_array_equal(final_own_turn_win(states, viewer), expected)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-6, 2, 6])
def test_symbolic_values_match_full_python_replay_and_baseline_planner(swap, margin):
    actor, _, worlds = setup(swap, margin)
    worlds = replace(
        worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
    )
    for optimize in (False, True):
        old = IncumbentBeliefPlanner(dynamics(actor, worlds)).solve(worlds, optimize)
        new = SymbolicFinalTurnPlanner(dynamics(actor, worlds)).solve(worlds, optimize)
        assert old.refusal is new.refusal is None
        assert new.plan.leaf_rule == RULE
        assert old.plan.values == new.plan.values
        assert old.plan.action == new.plan.action
        assert old.plan.action_upper_bounds == new.plan.action_upper_bounds
        verified = verify_program(worlds, new.plan, actor, {"fixture": actor})
        assert verified["values"] == {
            str(a): str(q) for a, q in old.plan.values.items()
        }
        if margin > 2:
            assert new.work.get("certified_own_win_mass", 0) > 0
            assert new.work["decisions"] < old.work["decisions"]
        else:
            assert new.plan.root_policies == old.plan.root_policies
            assert new.work == old.work


def test_executor_checks_root_rule_and_missing_key_domain():
    actor, root, worlds = setup()
    dyn = dynamics(actor, worlds)
    result = SymbolicFinalTurnPlanner(dyn).solve(worlds, False)
    binding = bind_worlds(worlds, dyn, "fixture-parent", {"fixture": "fixture"})
    selection = select_improvement(EvaluatedPhase(binding, "fixed", result))
    assert selection.parent_q == Fraction(204, 275)
    assert selection.selected_q == Fraction(228, 275)
    executor = SymbolicExecutor(selection, actor, "fixture-parent")
    env = root.clone()
    for _ in range(10):
        if env.state[PHASE] == TERMINAL:
            break
        env.step(
            executor.act(env) if env.state[PLAYER] == worlds.viewer else actor.act(env)
        )
    assert env.state[PHASE] == TERMINAL
    assert env.state[POINTS + worlds.viewer] > env.state[POINTS + 1 - worlds.viewer]
    assert executor.symbolic_parent_calls > 0
    bad = deepcopy(selection)
    bad.plan.leaf_rule = "unknown"
    with pytest.raises(ValueError):
        SymbolicExecutor(bad, actor, "fixture-parent")
    fresh = SymbolicExecutor(selection, actor, "fixture-parent")
    changed = root.clone()
    changed.state[POINTS + worlds.viewer] += 2
    with pytest.raises(ValueError):
        fresh.act(changed)
    # Started executors must not silently cover arbitrary absent histories.
    changed = root.clone()
    changed.state[TURN] = 14
    assert visible_keys(changed.state[None], worlds.viewer)[0] not in executor.policy
    with pytest.raises(KeyError):
        executor.act(changed)


def test_oracle_rejects_missing_actions_outside_declared_region():
    actor, _, worlds = setup(margin=2)
    result = SymbolicFinalTurnPlanner(dynamics(actor, worlds)).solve(worlds)
    broken = deepcopy(result.plan)
    action = next(iter(broken.root_policies))
    broken.root_policies[action].clear()
    with pytest.raises(KeyError):
        verify_program(worlds, broken, actor, {"fixture": actor})


def test_nonforced_symbolic_actions_are_fully_replayed():
    from koikoi.reference import ReferenceEnv

    actor, root = Extremal(), ReferenceEnv(31010004, "paper8")
    for _ in range(33):
        root.step(actor.act(root))
    assert root.state[TURN] == 15 and root.state[PHASE] == 0
    root.state[ROUND] = 7
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer], root.state[POINTS + 1 - viewer] = 33, 27
    support = enumerate_support(root, {"fixture": actor})
    worlds = deferred_worlds(root, support)
    old = IncumbentBeliefPlanner(dynamics(actor, worlds)).solve(worlds)
    new = SymbolicFinalTurnPlanner(dynamics(actor, worlds)).solve(worlds)
    assert new.plan.values == old.plan.values
    assert new.work["decisions"] < old.work["decisions"]
    for parent in (actor, Extremal(largest=True)):
        # Both legal capture choices in the certified region preserve outcome.
        checked = verify_program(worlds, new.plan, parent, {"fixture": actor})
        assert checked["new_parent_adapter_calls"] > 0
        assert all(Fraction(v) == 1 for v in checked["values"].values())


def test_budget_recovers_a_strict_complete_program_and_parent_reference():
    actor, _, worlds = setup()
    cap = 1258
    old_fixed, _, old_selection = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=cap,
    )
    assert old_fixed.result.plan is None and old_selection.reason == "parent_unvalued"
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=cap,
        planner_type=SymbolicFinalTurnPlanner,
    )
    assert selection.parent_q == Fraction(204, 275)
    assert selection.gain > 0 and selection.plan is not None
    for phase in (fixed, optimized):
        assert phase.result.work["decisions"] <= cap
        if phase.result.plan is not None:
            verify_program(worlds, phase.result.plan, actor, {"fixture": actor})


def test_opponent_only_certificate_is_preserved_and_program_serializes():
    import pickle

    from scripts.opponent_last_turn_tail import OpponentLastTurnPlanner
    from tests.test_opponent_last_turn_tail import setup as opponent_setup

    actor, _, worlds = opponent_setup()
    old = OpponentLastTurnPlanner(dynamics(actor, worlds)).solve(worlds)
    new = SymbolicFinalTurnPlanner(dynamics(actor, worlds)).solve(worlds)
    restored = pickle.loads(pickle.dumps(new.plan))
    assert restored == new.plan
    assert new.plan.values == old.plan.values
    assert new.plan.root_policies == old.plan.root_policies
    assert new.work["decisions"] == old.work["decisions"]
    assert (
        new.work["certified_opponent_loss_mass"] == old.work["certified_opponent_mass"]
    )
    assert new.work["certified_own_win_mass"] == 0
    verify_program(worlds, restored, actor, {"fixture": actor})
