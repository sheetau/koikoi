from fractions import Fraction

import pytest

from scripts.frontier_policy_patch import solve_frontier_patch
from scripts.pruned_optimized_frontier import solve_pruned_optimized_frontier
from scripts.verify_frontier_patch import verify_frontier_patch
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-50, 6, 50])
def test_pruned_optimized_values_and_saved_physical_work(swap, margin):
    parent, _, worlds = setup(swap, margin)
    kwargs = {
        "draws": 2,
        "max_decisions": 10000,
        "prefix_decisions": 4000,
        "per_phase_decisions": 2000,
    }
    old = solve_frontier_patch(
        worlds, dynamics(parent, worlds), "parent", {"fixture": "fixture"}, **kwargs
    )
    result = solve_pruned_optimized_frontier(
        worlds, dynamics(parent, worlds), "parent", {"fixture": "fixture"}, **kwargs
    )
    new = result.patch
    assert new.decisions == new.frontier.work["decisions"] + sum(
        r["decisions"] for _, r in new.comparisons
    )
    assert new.decisions <= old.decisions <= 10000
    assert new.gain_lower == old.gain_lower
    if abs(margin) == 50:
        assert result.skipped
        assert new.decisions < old.decisions
    if margin == -50:
        assert len(result.skipped) == len(new.frontier.nodes)
        assert not new.comparisons
    omitted = {key for key, _, _ in result.skipped}
    assert all(
        comparison["selection"].candidate is None
        for key, comparison in old.comparisons
        if key in omitted
    )
    old_values = verify_frontier_patch(
        worlds, old, parent, {"fixture": parent}, "parent", full_values=True
    )
    new_values = verify_frontier_patch(
        worlds, new, parent, {"fixture": parent}, "parent", full_values=True
    )
    assert (
        new_values["full_values"]["patched"]["value"]
        == old_values["full_values"]["patched"]["value"]
    )
    assert Fraction(new_values["exact_increment"]) >= new.gain_lower


@pytest.mark.parametrize("budget", [1, 400, 1000])
def test_pruned_optimized_budgets_and_incomplete_prefix(budget):
    parent, _, worlds = setup()
    result = solve_pruned_optimized_frontier(
        worlds,
        dynamics(parent, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=2,
        max_decisions=budget,
    )
    assert result.patch.decisions <= budget
    if not result.patch.frontier.complete:
        assert not result.skipped and not result.patch.comparisons
