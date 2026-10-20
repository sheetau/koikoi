from fractions import Fraction

import numpy as np
import pytest
import torch

from koikoi.layout import ROUND
from scripts.belief_tail import FiniteBeliefPlanner, KoiTailDynamics, TailBudgetExceeded
from scripts.bounded_belief_tail import BoundedBeliefPlanner
from scripts.deferred_belief_tail import DeferredWorldSet, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner
from scripts.reference_tail_policy import verify_deferred_policy_values
from tests.test_bounded_belief_tail import DelayedLossGame
from tests.test_ceiling_belief_tail import ToyPlanner
from tests.test_history_conditioned_worlds import Extremal, late_root


class ImprovingGame(DelayedLossGame):
    @staticmethod
    def legal(state):
        return (0, 1, 2) if state[0] == 0 else (0,)

    def advance(self, states, actions):
        after, work = super().advance(states, np.where(actions == 2, 1, actions))
        for i, (state, action) in enumerate(zip(states, actions, strict=True)):
            if state[0] == 0 and action != 2:
                after[i, 0] = 3
                after[i, 2] = (
                    (1 if state[1] == 0 else -1)
                    if action == 0
                    else (1 if state[1] != 3 else -1)
                )
            elif state[0] != 0 and state[1] == 0:
                # The first unresolved observation is a win, so an interrupted
                # candidate can have actual partial work before it is discarded.
                after[i, 2] = 1
        terminal = int(self.terminal(after).sum())
        work.update(round_endings=terminal, completions=terminal)
        return after, work


class IncumbentToy(IncumbentBeliefPlanner):
    _transition = ToyPlanner.transition


@pytest.mark.parametrize("cap", [8, 13])
def test_cap_retains_improvement_without_claiming_optimality(cap):
    dynamics = ImprovingGame(False)
    worlds = DeferredWorldSet(
        np.array([[0, w, 0] for w in range(4)], dtype=np.int32),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.int64),
        ("unused",),
        0,
        0,
    )
    expected = FiniteBeliefPlanner(dynamics).solve(worlds)
    result = IncumbentToy(dynamics, max_decisions=cap).solve(worlds)
    plan = result.plan
    assert result.refusal == "Physical transition cap exceeded"
    assert not plan.search_complete
    assert plan.action == 1 and plan.baseline_action == 0
    assert plan.values == {0: Fraction(1, 4), 1: Fraction(3, 4)}
    assert plan.action_upper_bounds == {2: 1}
    assert plan.optimality_gap == Fraction(1, 4)
    assert plan.policy == expected.policy
    for action, policy in plan.root_policies.items():
        assert policy == expected.root_policies[action]
        assert plan.values[action] == expected.values[action]
    assert expected.values[2] <= plan.action_upper_bounds[2]
    assert result.work["decisions"] == cap
    empty = IncumbentToy(dynamics, max_decisions=1).solve(worlds)
    assert empty.plan is None and empty.work["decisions"] == 0
    full = IncumbentToy(dynamics).solve(worlds)
    assert full.plan.search_complete and full.plan.optimality_gap == 0
    assert full.plan.policy == expected.policy


def test_native_full_plan_and_refusal_accounting():
    torch.set_num_threads(2)
    root, actor = late_root(), Extremal()
    root.state[ROUND] = 7
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support)
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    for optimize in (False, True):
        expected = BoundedBeliefPlanner(dynamics).solve(worlds, optimize)
        result = IncumbentBeliefPlanner(dynamics).solve(worlds, optimize)
        assert result.refusal is None and result.plan.search_complete
        for name in (
            "values",
            "action",
            "policy",
            "root_policies",
            "work",
            "action_upper_bounds",
        ):
            assert getattr(result.plan, name) == getattr(expected, name)
        verify_deferred_policy_values(worlds, result.plan, dynamics.policy_cache)
    control = BoundedBeliefPlanner(dynamics, max_decisions=1)
    with pytest.raises(TailBudgetExceeded):
        control.solve(worlds)
    result = IncumbentBeliefPlanner(dynamics, max_decisions=1).solve(worlds)
    assert result.plan is None
    assert result.work == control.work
