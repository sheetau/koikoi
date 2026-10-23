from fractions import Fraction

import pytest

from scripts.resumable_frontier import compare_root_programs, solve_resumable_frontier
from scripts.verify_frontier_patch import verify_frontier_patch
from scripts.verify_interval_tail import verify_receipt
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("budget", [1, 100, 10000])
def test_complete_program_intervals_under_shared_budget(swap, budget):
    actor, _, worlds = setup(swap)
    result = compare_root_programs(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        max_decisions=budget,
    )
    assert result["decisions"] <= budget
    for receipt in (result["parent_receipt"], *result["candidate_receipts"]):
        oracle = verify_receipt(worlds, receipt, actor, {"fixture": actor}, "parent")
        assert receipt.lower <= Fraction(oracle["value"]) <= receipt.upper


@pytest.mark.parametrize("swap", [False, True])
def test_resumable_frontier_independent_value_decomposition(swap):
    actor, _, worlds = setup(swap)
    patch = solve_resumable_frontier(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=2,
        max_decisions=10000,
        prefix_decisions=4000,
        local_decisions=4000,
    )
    audit = verify_frontier_patch(
        worlds, patch, actor, {"fixture": actor}, "parent", full_values=True
    )
    assert Fraction(audit["exact_increment"]) >= patch.gain_lower > 0
    assert patch.decisions == patch.frontier.work["decisions"] + sum(
        r["decisions"] for _, r in patch.comparisons
    )
    assert patch.decisions <= 10000
