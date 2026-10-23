from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from scripts.reusable_frontier import (
    CompletedChoices,
    ReusablePlanner,
    solve_reusable_frontier,
)
from scripts.score_bounded_tail import ScoreBoundPlanner
from scripts.verify_frontier_patch import verify_frontier_patch
from scripts.verify_score_bounded_tail import verify_program
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-6, 2, 6])
def test_completed_reuse_preserves_values_and_public_program(swap, margin):
    actor, _, worlds = setup(swap, margin)
    worlds = replace(
        worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
    )
    old = ScoreBoundPlanner(dynamics(actor, worlds)).solve(worlds)
    planner = ReusablePlanner(dynamics(actor, worlds))
    first = planner.solve(worlds)
    second = planner.solve(worlds)
    for result in (first, second):
        assert result.plan.values == old.plan.values
        assert result.plan.policy == old.plan.policy
        assert result.plan.action == old.plan.action
    assert second.work["decisions"] <= first.work["decisions"]
    report = verify_program(worlds, second.plan, actor, {"fixture": actor})
    assert report["values"] == {str(a): str(v) for a, v in old.plan.values.items()}
    if margin == 6:
        assert second.work["completed_choice_hits"] > 0
        assert second.work["decisions"] < first.work["decisions"]
    changed = replace(worlds, weights=worlds.weights * 2)
    with pytest.raises(ValueError, match="different belief"):
        planner.solve(changed)
    with pytest.raises(ValueError, match="different belief"):
        planner.solve(worlds, False)


def test_interrupted_search_reuses_only_completed_choices():
    actor, _, worlds = setup()
    planner = ReusablePlanner(dynamics(actor, worlds), max_decisions=500)
    partial = planner.solve(worlds)
    assert partial.refusal is not None
    planner.max_decisions = 100000
    complete = planner.solve(worlds)
    plain = ScoreBoundPlanner(dynamics(actor, worlds), max_decisions=100000).solve(
        worlds
    )
    assert complete.plan.values == plain.plan.values
    assert complete.plan.policy == plain.plan.policy
    assert complete.work.get("completed_choice_hits", 0) > 0
    assert complete.work["decisions"] < plain.work["decisions"]


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("cache_bytes", [0, 1048576])
def test_adaptive_patch_has_independent_full_value_and_budget_audit(swap, cache_bytes):
    actor, _, worlds = setup(swap)
    patch = solve_reusable_frontier(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=2,
        max_decisions=20000,
        prefix_decisions=4000,
        pilot_decisions=100,
        local_decisions=10000,
        cache_bytes=cache_bytes,
    )
    report = verify_frontier_patch(
        worlds, patch, actor, {"fixture": actor}, "parent", full_values=True
    )
    assert Fraction(report["exact_increment"]) >= patch.gain_lower > 0
    assert patch.decisions <= 20000
    assert patch.decisions == patch.frontier.work["decisions"] + sum(
        r["decisions"] for _, r in patch.comparisons
    )
    for _, r in patch.comparisons:
        assert (
            r["decisions"]
            == r["proposal_work"].get("decisions", 0)
            + r["parent_receipt"].work["decisions"]
        )
    assert patch.cache_peak_bytes <= cache_bytes


def test_cache_returns_detached_policies_and_respects_payload_limit():
    cache = CompletedChoices(1000)
    value = {b"visible": {48: 1}}
    cache.put(b"belief", value, 17)
    copy, _work = cache.get(b"belief")
    copy[b"visible"][48] = 9
    assert cache.get(b"belief") == (value, 17)
    cache.put(b"oversized", deepcopy(value) | {b"large": b"x" * 2000}, 20)
    assert cache.get(b"oversized") is None
    assert cache.peak_bytes <= 1000
