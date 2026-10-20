import numpy as np
import pytest
import torch

from koikoi.layout import ROUND
from scripts.belief_tail import FiniteBeliefPlanner, KoiTailDynamics, TailBudgetExceeded
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.deferred_belief_tail import DeferredWorldSet, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.terminal_reduced_belief_tail import TerminalReducedPlanner
from tests.test_belief_tail import GuessingGame, swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


class ToyPlanner(CeilingBeliefPlanner):
    def transition(self, states, models, weights, actions):
        after = self.advance(states, actions)
        self.work["represented_decisions"] += sum(map(int, weights))
        ended = sum(map(int, weights[self.dynamics.terminal(after)]))
        self.work["represented_round_endings"] += ended
        self.work["represented_completions"] += ended
        return after, models, weights


def toy_worlds(masses):
    return DeferredWorldSet(
        np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32),
        np.zeros(2, dtype=np.int64),
        np.array(masses, dtype=np.int64),
        ("unused",),
        0,
        0,
    )


@pytest.mark.parametrize("reveal", [False, True])
@pytest.mark.parametrize("masses", [(1, 1), (3, 1)])
def test_ceiling_respects_hidden_information_and_nonunit_masses(reveal, masses):
    dynamics, worlds = GuessingGame(reveal), toy_worlds(masses)
    for optimize in (False, True):
        expected = FiniteBeliefPlanner(dynamics).solve(worlds, optimize)
        plan = ToyPlanner(dynamics).solve(worlds, optimize)
        assert (plan.action, plan.policy, plan.values[plan.action]) == (
            expected.action,
            expected.policy,
            expected.values[expected.action],
        )
        assert all(q == expected.values[a] for a, q in plan.values.items())
        assert set(plan.unevaluated_actions) == set(expected.values) - set(plan.values)
        assert not plan.unevaluated_actions or plan.values[plan.action] == 1
        if not reveal:
            assert plan.values[plan.action] < 1


class WinningGame(GuessingGame):
    @staticmethod
    def legal(state):
        return (0, 1, 2)

    @staticmethod
    def baseline_action(state):
        return 1

    @staticmethod
    def advance(states, actions):
        after = states.copy()
        after[:, 0], after[:, 2] = 2, 1
        return after, dict.fromkeys(
            ("decisions", "round_endings", "completions"), len(states)
        )


def test_root_ceiling_omits_values_and_preserves_nonfirst_parent_tie():
    dynamics, worlds = WinningGame(False), toy_worlds((1, 3))
    expected = FiniteBeliefPlanner(dynamics).solve(worlds)
    plan = ToyPlanner(dynamics).solve(worlds)
    assert plan.action == expected.action == 1
    assert plan.policy == expected.policy
    assert plan.values == {1: 1}
    assert plan.unevaluated_actions == (0, 2)
    assert plan.root_ceiling_attained
    assert plan.work["pruned_actions"] == 2
    assert plan.work["decisions"] == 2 < expected.work["decisions"]


@pytest.mark.parametrize("swap", [False, True])
def test_native_ceiling_matches_complete_plans_and_independent_replay(swap):
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
        plan = CeilingBeliefPlanner(dynamics).solve(worlds, optimize)
        assert (plan.action, plan.policy, plan.values[plan.action]) == (
            expected.action,
            expected.policy,
            expected.values[expected.action],
        )
        for action, value in plan.values.items():
            assert value == expected.values[action]
            assert plan.root_policies[action] == expected.root_policies[action]
        assert set(plan.unevaluated_actions) == set(expected.values) - set(plan.values)
        assert plan.work["decisions"] <= expected.work["decisions"]
        verify_deferred_policy_values(worlds, plan, dynamics.policy_cache)
    with pytest.raises(TailBudgetExceeded, match="Physical transition"):
        CeilingBeliefPlanner(dynamics, max_decisions=1).solve(worlds)
    with pytest.raises(TailBudgetExceeded, match="node"):
        CeilingBeliefPlanner(dynamics, max_nodes=1).solve(worlds)
