from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from scripts.belief_tail import KoiTailDynamics
from scripts.deferred_belief_tail import deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.opponent_last_turn_tail import (
    OpponentLastTurnPlanner,
    opponent_last_turn_loss,
    verify_policy_values,
)
from scripts.reference_tail_policy import verify_deferred_policy_values
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


def setup(swap=False, margin=-6):
    torch.set_num_threads(2)
    actor, root = Extremal(), late_root()
    while root.state[TURN] < 14:
        root.step(actor.act(root))
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


def test_certificate_is_public_strict_and_only_after_last_learner_move():
    _, root, worlds = setup()
    viewer = worlds.viewer
    state = root.state.copy()
    state[TURN], state[PLAYER] = 15, 1 - viewer
    rows = np.repeat(state[None], 8, axis=0)
    rows[1, POINTS + viewer] = rows[1, POINTS + 1 - viewer] - 2
    rows[2, POINTS + viewer] = rows[2, POINTS + 1 - viewer]
    rows[3, TURN] = 14
    rows[4, ROUND] = 6
    rows[5, PLAYER] = viewer
    rows[6, PHASE] = TERMINAL
    rows[7, POINTS + viewer] = rows[7, POINTS + 1 - viewer] + 6
    expected = [True, False, False, False, False, False, False, False]
    np.testing.assert_array_equal(opponent_last_turn_loss(rows, viewer), expected)
    changed = rows.copy()
    changed[:, :48] = changed[:, :48][:, ::-1]
    changed[:, 48:96] = changed[:, 48:96][:, ::-1]
    np.testing.assert_array_equal(opponent_last_turn_loss(changed, viewer), expected)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-6, -2, 6])
def test_values_and_complete_learner_maps_match_without_certified_replay(swap, margin):
    actor, _, worlds = setup(swap, margin)
    # Unequal integer world weights exercise weighted aggregation, not row counts.
    worlds = replace(
        worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
    )
    for optimize in (False, True):
        control_dynamics = dynamics(actor, worlds)
        control = IncumbentBeliefPlanner(control_dynamics).solve(worlds, optimize)
        candidate_dynamics = dynamics(actor, worlds)
        candidate = OpponentLastTurnPlanner(candidate_dynamics).solve(worlds, optimize)
        assert control.refusal is candidate.refusal is None
        assert candidate.plan.values == control.plan.values
        assert candidate.plan.policy == control.plan.policy
        assert candidate.plan.root_policies == control.plan.root_policies
        assert candidate.plan.action == control.plan.action
        checked = verify_policy_values(worlds, candidate.plan, {"fixture": actor})
        original_oracle = verify_deferred_policy_values(
            worlds, control.plan, control_dynamics.policy_cache
        )
        for name in ("values", "physical_work", "represented_work"):
            assert checked[name] == original_oracle[name]
        if margin < -2:
            assert candidate.work["decisions"] < control.work["decisions"]
            assert candidate.work["certified_opponent_mass"] > 0
            assert (
                candidate.work["represented_completions"]
                + candidate.work["certified_opponent_mass"]
                == control.work["represented_completions"]
            )
        else:
            assert candidate.work == control.work


def test_same_cap_recovers_exact_parent_reference_without_fake_completions():
    actor, _, worlds = setup()
    control = IncumbentBeliefPlanner(dynamics(actor, worlds)).solve(worlds)
    full = OpponentLastTurnPlanner(dynamics(actor, worlds)).solve(worlds)
    cap = full.work["decisions"]
    assert cap == 50 < control.work["decisions"] == 374
    old, skipped, old_selection = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "fixture-parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=cap,
    )
    assert old.result.plan is None and skipped is None
    assert old_selection.reason == "parent_unvalued"
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "fixture-parent",
        {"fixture": "fixture"},
        max_decisions_per_phase=cap,
        planner_type=OpponentLastTurnPlanner,
    )
    assert fixed.result.refusal is optimized.result.refusal is None
    assert selection.parent_q == Fraction(7, 25)
    assert selection.reason == "no_strict_gain"  # Capability, not a strength result.
    assert fixed.result.work["completions"] == 14
    assert fixed.result.work["certified_opponent_mass"] == 324
    assert fixed.result.work["represented_completions"] == 126


def test_four_draw_nested_choices_and_partial_action_cutoffs():
    torch.set_num_threads(2)
    actor, root = Extremal(), ReferenceEnv(31010001, "paper8")
    for _ in range(2000):
        if root.state[TURN] == 12 and root.state[PHASE] == 0:
            break
        root.step(actor.act(root))
    else:
        raise AssertionError("Four-draw fixture missing")
    root.state[ROUND] = 7
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer], root.state[POINTS + 1 - viewer] = 20, 40
    support = enumerate_support(root, {"fixture": actor}, max_hands=1000)
    worlds = deferred_worlds(root, support, max_draws=4, max_hands=1000)
    old_dynamics, new_dynamics = dynamics(actor, worlds), dynamics(actor, worlds)
    old = IncumbentBeliefPlanner(old_dynamics).solve(worlds)
    new = OpponentLastTurnPlanner(new_dynamics).solve(worlds)
    assert old.refusal is new.refusal is None
    assert old.plan.action == new.plan.action
    assert old.plan.values == new.plan.values == {36: Fraction(1411, 10560)}
    assert old.plan.action_upper_bounds == new.plan.action_upper_bounds
    assert old.plan.root_policies == new.plan.root_policies
    assert old.work["continuation_cutoffs"] > 0
    assert new.work["continuation_cutoffs"] > 0
    assert new.work["certified_opponent_mass"] > 0
    assert new.work["decisions"] == 152994 < old.work["decisions"] == 698133
    assert new.work["own_decision_nodes"] > 0
    checked = verify_policy_values(worlds, new.plan, {"fixture": actor})
    original = verify_deferred_policy_values(
        worlds, old.plan, old_dynamics.policy_cache
    )
    for name in ("values", "physical_work", "represented_work"):
        assert checked[name] == original[name]
