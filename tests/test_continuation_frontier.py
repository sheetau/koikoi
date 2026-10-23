from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.continuation_frontier import (
    ContinuationActors,
    ContinuationEvaluator,
    ContinuationFrontierExecutor,
    ContinuationProgram,
    compare_continuations,
    solve_continuation_frontier,
    verify_continuation_patch,
    verify_continuation_receipt,
)
from scripts.diagnose_history_support import reference
from scripts.frontier_policy_patch import (
    FrontierExecutor,
    FrontierPatch,
    parent_frontier,
)
from scripts.interval_tail import (
    IntervalEvaluator,
    IntervalExecutor,
    make_program,
    select_interval,
)
from scripts.resumable_frontier import compare_root_programs
from scripts.verify_interval_tail import verify_receipt
from tests.test_history_conditioned_worlds import Extremal
from tests.test_symbolic_final_turn_tail import dynamics, setup


def fixture(swap=False):
    parent, root, worlds = setup(swap)
    actors = ContinuationActors("parent", parent, {"alternative": Extremal(True)})
    dyn = dynamics(parent, worlds)
    original = make_program(worlds, dyn, "parent", {"fixture": "fixture"})
    named = ContinuationProgram(
        original.binding,
        original.parent_root_action,
        ((original.binding.root_key, actors.action("alternative", worlds.states[0])),),
        "alternative",
    )
    return parent, root, worlds, actors, dyn, original, named


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("budget", [1, 100, 10000])
def test_named_intervals_cover_independent_full_values_after_interruptions(
    swap, budget
):
    parent, _, worlds, actors, dyn, _, named = fixture(swap)
    evaluator = ContinuationEvaluator(worlds, dyn, named, actors, certificates=False)
    receipt = evaluator.run(budget)
    oracle, explicit = verify_continuation_receipt(
        worlds, receipt, actors, {"fixture": parent}
    )
    exact = Fraction(oracle["value"])
    assert receipt.lower <= exact <= receipt.upper
    assert receipt.work["decisions"] <= budget
    assert len(explicit.program.overrides) > 1
    completed = evaluator.run(100000)
    assert completed.lower == completed.upper == exact == Fraction(228, 275)
    assert completed.work["decisions"] == 4733


@pytest.mark.parametrize("swap", [False, True])
def test_named_default_execution_matches_full_explicit_projection(swap):
    parent, root, worlds, actors, dyn, original, named = fixture(swap)
    receipts = [
        ContinuationEvaluator(worlds, dyn, p, actors, certificates=False).run(100000)
        for p in (original, named)
    ]
    selection = select_interval(receipts[0], receipts[1:])
    assert selection.candidate is receipts[1]
    frontier = parent_frontier(
        worlds, dyn, "parent", {"fixture": "fixture"}, draws=worlds.draws
    )
    assert len(frontier.nodes) == 1
    result = {
        "selection": selection,
        "decisions": sum(r.work["decisions"] for r in receipts),
    }
    patch = FrontierPatch(
        frontier,
        ((original.binding.root_key, result),),
        selection.gain_lower,
        result["decisions"],
    )
    audit = verify_continuation_patch(
        worlds, patch, actors, {"fixture": parent}, full_values=True
    )
    assert Fraction(audit["exact_increment"]) == Fraction(24, 275)
    assert (
        Fraction(audit["full_values"]["patched"]["value"])
        - Fraction(audit["full_values"]["parent"]["value"])
        == selection.gain_lower
    )
    _, explicit = verify_continuation_receipt(
        worlds, receipts[1], actors, {"fixture": parent}
    )
    explicit_selection = replace(selection, candidate=explicit)
    ordinary_patch = replace(
        patch,
        comparisons=(
            (original.binding.root_key, {**result, "selection": explicit_selection}),
        ),
    )
    # Multiple complete counterfactual trajectories, both seats. Dispatch uses
    # own/public keys and observation-only actors, not per-world overrides.
    for state in [root.state, *worlds.states]:
        env = reference(state)
        new = ContinuationFrontierExecutor(patch, actors)
        ordinary = FrontierExecutor(ordinary_patch, parent, "parent")
        fallback_calls = actors.alternatives["alternative"].calls
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] == worlds.viewer:
                action = new.act(env)
                assert action == ordinary.act(env)
            else:
                action = parent.act(env)
            env.step(action)
        assert env.state[PHASE] == TERMINAL
        assert actors.alternatives["alternative"].calls > fallback_calls


@pytest.mark.parametrize("swap", [False, True])
def test_shared_budget_and_frontier_decomposition(swap):
    parent, _, worlds, actors, dyn, _, _ = fixture(swap)
    tiny = compare_continuations(
        worlds, dyn, actors, {"fixture": "fixture"}, max_decisions=1
    )
    assert tiny["decisions"] <= 1
    for receipt in [tiny["parent_receipt"], *tiny["candidate_receipts"]]:
        verify_continuation_receipt(worlds, receipt, actors, {"fixture": parent})
    patch = solve_continuation_frontier(
        worlds,
        dyn,
        actors,
        {"fixture": "fixture"},
        draws=2,
        max_decisions=10000,
        prefix_decisions=4000,
        local_decisions=4000,
    )
    audit = verify_continuation_patch(
        worlds, patch, actors, {"fixture": parent}, full_values=True
    )
    assert Fraction(audit["exact_increment"]) >= patch.gain_lower > 0
    assert patch.decisions == patch.frontier.work["decisions"] + sum(
        r["decisions"] for _, r in patch.comparisons
    )
    assert patch.decisions <= 10000


def test_identity_root_invariant_and_legacy_dispatch_rejections():
    parent, root, worlds, actors, dyn, original, named = fixture()
    with pytest.raises(FrozenInstanceError):
        named.continuation_identity = "changed"
    with pytest.raises(TypeError):
        actors.alternatives["alternative"] = parent
    with pytest.raises(ValueError, match="Unknown"):
        ContinuationEvaluator(
            worlds, dyn, replace(named, continuation_identity="missing"), actors
        )
    with pytest.raises(ValueError, match="reference"):
        ContinuationEvaluator(worlds, dynamics(Extremal(True), worlds), named, actors)
    wrong_root = next(
        a for a in original.binding.legal if a != named.parent_root_action
    )
    with pytest.raises(ValueError, match="parent root"):
        ContinuationEvaluator(
            worlds, dyn, replace(named, parent_root_action=wrong_root), actors
        )
    with pytest.raises(ValueError, match="explicit actor dispatch"):
        IntervalEvaluator(worlds, dyn, named, certificates=False).run(100)
    receipts = [
        ContinuationEvaluator(worlds, dyn, p, actors, certificates=False).run(100000)
        for p in (original, named)
    ]
    selection = select_interval(receipts[0], receipts[1:])
    with pytest.raises(ValueError, match="explicit actor dispatch"):
        IntervalExecutor(selection, parent, "parent").act(root)
    with pytest.raises(ValueError, match="explicit actor dispatch"):
        verify_receipt(worlds, receipts[1], parent, {"fixture": parent}, "parent")


def test_action_cache_is_public_and_separated_by_actor_identity():
    parent, _, worlds = setup()
    low, high = Extremal(), Extremal(True)
    actors = ContinuationActors("parent", parent, {"low": low, "high": high})
    a = actors.action("low", worlds.states[0])
    calls = low.calls
    assert actors.action("low", worlds.states[-1]) == a
    assert low.calls == calls
    assert actors.action("high", worlds.states[0]) != a
    assert actors.calls["low"] == actors.calls["high"] == 1


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("budget", [1, 100, 4096, 10000])
def test_receipt_reuse_preserves_the_original_candidate_family(swap, budget):
    parent, _, worlds = setup(swap)
    actors = ContinuationActors("parent", parent, {})
    old = compare_root_programs(
        worlds,
        dynamics(parent, worlds),
        "parent",
        {"fixture": "fixture"},
        max_decisions=budget,
    )
    new = compare_continuations(
        worlds,
        dynamics(parent, worlds),
        actors,
        {"fixture": "fixture"},
        max_decisions=budget,
    )
    assert new["selection"] == old["selection"]
    assert new["candidate_receipts"] == old["candidate_receipts"]
    assert new["decisions"] == old["decisions"]


class StopAtFirstAction:
    def act(self, env):
        return int(env.legal().nonzero()[0][0])


@pytest.mark.parametrize("swap", [False, True])
def test_same_root_action_can_have_a_different_certified_continuation(swap):
    parent, _, worlds = setup(swap)
    dyn = dynamics(parent, worlds)
    actors = ContinuationActors("parent", parent, {"stopper": StopAtFirstAction()})
    original = make_program(worlds, dyn, "parent", {"fixture": "fixture"})
    named = ContinuationProgram(
        original.binding,
        original.parent_root_action,
        ((original.binding.root_key, actors.action("stopper", worlds.states[0])),),
        "stopper",
    )
    assert named.overrides == original.overrides
    assert original.is_parent and not named.is_parent
    receipts = [
        ContinuationEvaluator(worlds, dyn, p, actors, certificates=False).run(100000)
        for p in (original, named)
    ]
    assert receipts[0].lower == receipts[0].upper == Fraction(204, 275)
    assert receipts[1].lower == receipts[1].upper == Fraction(461, 550)
    selection = select_interval(receipts[0], receipts[1:])
    with pytest.raises(ValueError, match="unchanged parent"):
        select_interval(receipts[1], [receipts[0]])
    frontier = parent_frontier(
        worlds, dyn, "parent", {"fixture": "fixture"}, draws=worlds.draws
    )
    work = sum(r.work["decisions"] for r in receipts)
    patch = FrontierPatch(
        frontier,
        ((original.binding.root_key, {"selection": selection, "decisions": work}),),
        selection.gain_lower,
        work,
    )
    audit = verify_continuation_patch(
        worlds, patch, actors, {"fixture": parent}, full_values=True
    )
    assert Fraction(audit["exact_increment"]) == Fraction(53, 550)
    assert (
        audit["local_checks"][0]["oracle"]["candidate"]["continuation_identity"]
        == "stopper"
    )
