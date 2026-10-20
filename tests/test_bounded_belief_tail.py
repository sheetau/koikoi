import numpy as np
import pytest
import torch

from koikoi.layout import ROUND
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.bounded_belief_tail import BoundedBeliefPlanner
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.deferred_belief_tail import DeferredWorldSet, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.terminal_reduced_belief_tail import TerminalReducedPlanner
from tests.test_belief_tail import GuessingGame, swap_seats
from tests.test_ceiling_belief_tail import ToyPlanner, toy_worlds
from tests.test_history_conditioned_worlds import Extremal, late_root


class BoundedToy(BoundedBeliefPlanner):
    transition = ToyPlanner.transition


def compare(plan, expected):
    assert (
        plan.action,
        plan.baseline_action,
        plan.values[plan.action],
        plan.policy,
    ) == (
        expected.action,
        expected.baseline_action,
        expected.values[expected.action],
        expected.policy,
    )
    for action, value in plan.values.items():
        assert value == expected.values[action]
        assert plan.root_policies[action] == expected.root_policies[action]
    for action, upper in plan.action_upper_bounds.items():
        assert expected.values[action] <= upper <= plan.values[plan.action]
    assert set(plan.values).isdisjoint(plan.action_upper_bounds)
    assert set(plan.values) | set(plan.action_upper_bounds) == set(expected.values)
    assert plan.work["decisions"] <= expected.work["decisions"]


@pytest.mark.parametrize("reveal", [False, True])
@pytest.mark.parametrize("masses", [(1, 1), (3, 1)])
def test_hidden_information_and_nonunit_masses(reveal, masses):
    dynamics, worlds = GuessingGame(reveal), toy_worlds(masses)
    for optimize in (False, True):
        # The unpruned finite solver supplies every exact root action.
        from scripts.belief_tail import FiniteBeliefPlanner

        expected = FiniteBeliefPlanner(dynamics).solve(worlds, optimize)
        compare(BoundedToy(dynamics).solve(worlds, optimize), expected)


class DelayedLossGame:
    def __init__(self, shared):
        self.shared = shared

    def keys(self, states, viewer):
        return [
            (int(s[0]), -1 if s[0] == 0 or (s[0] == 1 and self.shared) else int(s[1]))
            for s in states
        ]

    @staticmethod
    def legal(state):
        return (0, 1) if state[0] == 0 else (0,)

    @staticmethod
    def baseline_action(state):
        return 0

    @staticmethod
    def opponent_actions(states, models):
        return np.zeros(len(states), dtype=np.int32)

    @staticmethod
    def players(states):
        return (states[:, 0] != 0).astype(np.int32)

    @staticmethod
    def terminal(states):
        return states[:, 0] == 3

    @staticmethod
    def outcomes(states, viewer):
        return states[:, 2]

    def advance(self, states, actions):
        after = states.copy()
        for i, (s, a) in enumerate(zip(states, actions, strict=True)):
            if s[0] == 0:
                if a == 0:
                    after[i, 0], after[i, 2] = 3, 1 if s[1] != 3 else -1
                else:
                    after[i, 0] = 1
            elif self.shared and s[0] == 1 and s[1] != 0:
                after[i, 0] = 2
            else:
                after[i, 0], after[i, 2] = 3, -1 if s[1] < 2 else 1
        terminal = int(self.terminal(after).sum())
        return after, {
            "decisions": len(states),
            "round_endings": terminal,
            "completions": terminal,
        }


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("masses", [(1, 1, 1, 1), (3, 1, 1, 2)])
def test_partial_action_upper_bound_discards_unsolved_continuation(shared, masses):
    dynamics = DelayedLossGame(shared)
    worlds = DeferredWorldSet(
        np.array([[0, w, 0] for w in range(4)], dtype=np.int32),
        np.zeros(4, dtype=np.int64),
        np.array(masses, dtype=np.int64),
        ("unused",),
        0,
        0,
    )
    expected = ToyPlanner(dynamics).solve(worlds)
    plan = BoundedToy(dynamics).solve(worlds)
    compare(plan, expected)
    assert 0 < plan.values[plan.action] < 1
    assert set(plan.values) == {0}
    assert set(plan.action_upper_bounds) == {1}
    assert plan.work["bounded_actions"] == 1
    assert plan.work["continuation_cutoffs"] == 1
    assert plan.work["decisions"] < expected.work["decisions"]


@pytest.mark.parametrize("swap", [False, True])
def test_native_values_policies_oracle_and_caps(swap):
    torch.set_num_threads(2)
    root = late_root()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    actor = Extremal()
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support, {"first": 3})
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    for optimize in (False, True):
        expected = TerminalReducedPlanner(dynamics).solve(worlds, optimize)
        plan = BoundedBeliefPlanner(dynamics).solve(worlds, optimize)
        compare(plan, expected)
        ceiling = CeilingBeliefPlanner(dynamics).solve(worlds, optimize)
        assert plan.work["decisions"] <= ceiling.work["decisions"]
        verify_deferred_policy_values(worlds, plan, dynamics.policy_cache)
    with pytest.raises(TailBudgetExceeded, match="Physical"):
        BoundedBeliefPlanner(dynamics, max_decisions=1).solve(worlds)
    with pytest.raises(TailBudgetExceeded, match="node"):
        BoundedBeliefPlanner(dynamics, max_nodes=1).solve(worlds)
