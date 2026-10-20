import pytest
import torch

from koikoi.layout import ROUND
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.deferred_belief_tail import DeferredBeliefPlanner, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.terminal_reduced_belief_tail import TerminalReducedPlanner
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


@pytest.mark.parametrize("swap", [False, True])
def test_terminal_aggregation_preserves_values_policies_and_physical_work(swap):
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
        expected = DeferredBeliefPlanner(dynamics).solve(worlds, optimize)
        plan = TerminalReducedPlanner(dynamics).solve(worlds, optimize)
        assert (plan.action, plan.baseline_action, plan.values, plan.root_policies) == (
            expected.action,
            expected.baseline_action,
            expected.values,
            expected.root_policies,
        )
        for key in (
            "decisions",
            "round_endings",
            "completions",
            "represented_decisions",
            "represented_round_endings",
            "represented_completions",
            "own_decision_nodes",
            "chance_rows_added",
            "maximum_frontier_rows",
        ):
            assert plan.work[key] == expected.work[key]
        assert "observation_nodes" not in plan.work
        assert plan.work["live_observation_nodes"] < expected.work["observation_nodes"]
        verify_deferred_policy_values(worlds, plan, dynamics.policy_cache)
    with pytest.raises(TailBudgetExceeded, match="Physical transition"):
        TerminalReducedPlanner(dynamics, max_decisions=1).solve(worlds)
    with pytest.raises(TailBudgetExceeded, match="node"):
        TerminalReducedPlanner(dynamics, max_nodes=1).solve(worlds)
