from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.layout import EPISODE, PHASE, ROUND, SEED, STOCK, TURN
from koikoi.reference import ReferenceEnv
from scripts.belief_tail import (
    FiniteBeliefPlanner,
    KoiTailDynamics,
    TailBudgetExceeded,
    enumerate_tail_worlds,
)
from scripts.deferred_belief_tail import DeferredBeliefPlanner, deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_worlds import enumerate_support
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


def checked_dynamics(worlds, actor, counts):
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    native = dynamics.advance

    def checked(states, actions):
        after, work = native(states, actions)
        expected = {"decisions": len(states), "round_endings": 0, "completions": 0}
        for before, action, actual in zip(states, actions, after, strict=True):
            env = reference(before)
            result = env.step(int(action))
            np.testing.assert_array_equal(actual, env.state)
            expected["round_endings"] += int(result[1])
            expected["completions"] += int(result[2])
            counts["phases"].add(int(before[PHASE]))
        assert expected == work
        counts["transitions"] += len(states)
        return after, work

    dynamics.advance = checked
    return dynamics


@pytest.mark.parametrize("swap", [False, True])
def test_deferred_equals_eager_all_root_policies_and_weighted_work(swap):
    torch.set_num_threads(2)
    root = late_root()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    actor = Extremal()
    support = enumerate_support(root, {"first": actor})
    prior = {"first": 3}
    eager = enumerate_tail_worlds(root, support, prior)
    deferred = deferred_worlds(root, support, prior)
    assert len(eager.states) == 4950 and len(deferred.states) == 5
    assert deferred.weights.sum() == eager.weights.sum()
    original = deferred.states.copy()
    counts = {"transitions": 0, "phases": set()}
    dynamics = checked_dynamics(deferred, actor, counts)
    eager_dynamics = KoiTailDynamics(
        len(eager.states), eager.model_names, {"first": actor}, actor
    )
    steps = 0
    for optimize in (False, True):
        expected = FiniteBeliefPlanner(eager_dynamics).solve(eager, optimize)
        actual = DeferredBeliefPlanner(dynamics).solve(deferred, optimize)
        assert (actual.action, actual.values, actual.root_policies) == (
            expected.action,
            expected.values,
            expected.root_policies,
        )
        for key in ("decisions", "round_endings", "completions"):
            assert actual.work[f"represented_{key}"] == 3 * expected.work[key]
        assert actual.work["decisions"] < expected.work["decisions"]
        assert actual.work["observation_nodes"] == expected.work["observation_nodes"]
        steps += actual.work["decisions"]
    assert counts["transitions"] == steps
    np.testing.assert_array_equal(deferred.states, original)
    alternative = reference(eager.states[-1])
    alternative.state[SEED], alternative.state[EPISODE] = 12345, 12
    end = 49 + int(alternative.state[STOCK])
    alternative.state[72:end] = alternative.state[72:end][::-1]
    invariant = deferred_worlds(alternative, support, prior)
    np.testing.assert_array_equal(invariant.states, deferred.states)
    np.testing.assert_array_equal(invariant.weights, deferred.weights)
    with pytest.raises(TailBudgetExceeded, match="node"):
        DeferredBeliefPlanner(dynamics, max_nodes=1).solve(deferred)
    dynamics.capacity = 1
    with pytest.raises(TailBudgetExceeded, match="frontier"):
        DeferredBeliefPlanner(dynamics).solve(deferred)
    with pytest.raises(TypeError, match="unordered"):
        DeferredBeliefPlanner(dynamics).solve(eager)
    with pytest.raises(ValueError, match="incomplete"):
        deferred_worlds(root, replace(support, hands=support.hands[:-1]), prior)
    with pytest.raises(TailBudgetExceeded, match="draw"):
        deferred_worlds(root, support, max_draws=2)


def test_four_draw_tail_matches_complete_prefix_solution():
    torch.set_num_threads(2)
    root, actor = ReferenceEnv(31010001, "paper8"), Extremal()
    for _ in range(2000):
        if root.state[TURN] == 12 and root.state[PHASE] == 0:
            break
        root.step(actor.act(root))
    assert root.state[TURN] == 12 and root.state[PHASE] == 0
    root.state[ROUND] = 7
    support = enumerate_support(root, {"first": actor}, max_hands=1000)
    deferred = deferred_worlds(root, support, max_draws=4)
    eager = enumerate_tail_worlds(root, support, max_draws=4, max_worlds=1000000)
    assert deferred.draws == 4
    assert deferred.weights.sum() == len(eager.states)
    counts = {"transitions": 0, "phases": set()}
    dynamics = checked_dynamics(deferred, actor, counts)
    eager_dynamics = KoiTailDynamics(
        len(eager.states), eager.model_names, {"first": actor}, actor
    )
    for optimize in (False, True):
        expected = FiniteBeliefPlanner(eager_dynamics, max_nodes=1000000).solve(
            eager, optimize
        )
        actual = DeferredBeliefPlanner(dynamics, max_nodes=1000000).solve(
            deferred, optimize
        )
        assert (actual.action, actual.values, actual.root_policies) == (
            expected.action,
            expected.values,
            expected.root_policies,
        )
        for key in ("decisions", "round_endings", "completions"):
            assert actual.work[f"represented_{key}"] == expected.work[key]
        assert actual.work["decisions"] < expected.work["decisions"]
    assert counts["phases"] == {0, 1, 2, 3}
