"""Public outcome pruning must preserve every executable improvement."""

from fractions import Fraction

import pytest

from scripts.certified_frontier import solve_certified_frontier
from scripts.continuation_frontier import (
    ContinuationActors,
    solve_continuation_frontier,
    verify_continuation_patch,
)
from tests.test_symbolic_final_turn_tail import Extremal, dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-50, 6, 50])
@pytest.mark.parametrize("named", [False, True])
def test_public_skips_preserve_comparisons_and_exact_value(swap, margin, named):
    parent, _, worlds = setup(swap, margin)
    alternatives = {"max": Extremal(True)} if named else {}
    original_actors = ContinuationActors("parent", parent, alternatives)
    skipped_actors = ContinuationActors("parent", parent, alternatives)
    kwargs = {"draws": 1, "max_decisions": 10000, "local_decisions": 4000}
    old = solve_continuation_frontier(
        worlds,
        dynamics(parent, worlds),
        original_actors,
        {"fixture": "fixture"},
        **kwargs,
    )
    result = solve_certified_frontier(
        worlds,
        dynamics(parent, worlds),
        skipped_actors,
        {"fixture": "fixture"},
        **kwargs,
    )
    new = result.patch
    assert new.decisions == old.decisions
    assert new.gain_lower == old.gain_lower
    omitted = {key for key, _, _ in result.skipped}
    assert new.comparisons == tuple(
        (key, r) for key, r in old.comparisons if key not in omitted
    )
    for key, comparison in old.comparisons:
        if key in omitted:
            assert comparison["decisions"] == 0
            assert comparison["selection"].candidate is None
            receipts = [comparison["parent_receipt"], *comparison["candidate_receipts"]]
            assert all(r.lower == r.upper == receipts[0].lower for r in receipts)
    if abs(margin) == 50:
        assert result.skipped
        assert len(result.skipped) == len(new.frontier.nodes)
        assert not skipped_actors.calls
    audit = verify_continuation_patch(
        worlds, new, skipped_actors, {"fixture": parent}, full_values=True
    )
    assert Fraction(audit["exact_increment"]) >= new.gain_lower


def test_refused_prefix_never_skips_or_patches():
    parent, _, worlds = setup()
    actors = ContinuationActors("parent", parent, {})
    result = solve_certified_frontier(
        worlds,
        dynamics(parent, worlds),
        actors,
        {"fixture": "fixture"},
        draws=1,
        max_decisions=1,
    )
    assert not result.patch.frontier.complete
    assert not result.patch.comparisons
    assert not result.skipped
