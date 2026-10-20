from fractions import Fraction

import pytest
import torch

from koikoi.layout import ROUND
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.deferred_belief_tail import deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.profile_refused_tails import ProfiledPlanner
from tests.test_history_conditioned_worlds import Extremal, late_root


def test_profile_preserves_complete_and_refused_plans():
    torch.set_num_threads(2)
    root, actor = late_root(), Extremal()
    root.state[ROUND] = 7
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support)
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    for optimize in (False, True):
        expected = CeilingBeliefPlanner(dynamics).solve(worlds, optimize)
        planner = ProfiledPlanner(dynamics)
        actual = planner.solve(worlds, optimize)
        assert actual == expected
        assert sum(planner.phase_turn.values()) == actual.work["decisions"]
        assert {
            r["action"]: Fraction(r["q"]) for r in planner.root_actions
        } == actual.values
        assert all(r["status"] == "complete" for r in planner.root_actions)
        assert (
            sum(r["work"]["decisions"] for r in planner.root_actions)
            == actual.work["decisions"]
        )
    control = CeilingBeliefPlanner(dynamics, max_decisions=100)
    planner = ProfiledPlanner(dynamics, max_decisions=100)
    for candidate in (control, planner):
        with pytest.raises(TailBudgetExceeded, match="Physical"):
            candidate.solve(worlds)
    planner.finish_root("budget_refused")
    assert planner.work == control.work
    assert planner.root_actions[-1]["status"] == "budget_refused"
    assert "q" not in planner.root_actions[-1]
    assert sum(planner.phase_turn.values()) == planner.work["decisions"]
    assert (
        sum(r["work"]["decisions"] for r in planner.root_actions)
        == planner.work["decisions"]
    )
