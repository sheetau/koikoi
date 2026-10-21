from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.belief_tail import KoiTailDynamics
from scripts.interval_tail import (
    IntervalEvaluator,
    IntervalExecutor,
    compare_programs,
    make_program,
    select_interval,
)
from scripts.verify_interval_tail import verify_receipt
from tests.test_symbolic_final_turn_tail import dynamics, setup


def programs(worlds, dyn):
    parent = make_program(worlds, dyn, "parent", {"fixture": "fixture"})
    others = [
        make_program(
            worlds, dyn, "parent", {"fixture": "fixture"}, {parent.binding.root_key: a}
        )
        for a in parent.binding.legal
        if a != parent.parent_root_action
    ]
    return [parent, *others]


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-40, 2, 6, 40])
@pytest.mark.parametrize("certificates", [False, True])
def test_resumed_intervals_contain_full_python_value(swap, margin, certificates):
    actor, _, worlds = setup(swap, margin)
    worlds = replace(
        worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
    )
    dyn = dynamics(actor, worlds)
    parent = programs(worlds, dyn)[0]
    evaluator = IntervalEvaluator(worlds, dyn, parent, certificates=certificates)
    snapshots = []
    for cap in (0, 1, 100, 500, 1000, 100000):
        r = evaluator.run(cap)
        assert r.work["decisions"] <= cap
        if snapshots:
            assert snapshots[-1].lower <= r.lower <= r.upper <= snapshots[-1].upper
        snapshots.append(r)
    assert r.lower == r.upper and r.refusal is None
    oracle = verify_receipt(worlds, r, actor, {"fixture": actor}, "parent")
    q = Fraction(oracle["value"])
    assert all(x.lower <= q <= x.upper for x in snapshots)
    assert r.resolved_mass == sum(map(int, worlds.weights))
    if not certificates:
        assert r.work["represented_completions"] == r.total_mass


def test_certifies_a_gain_before_parent_value_is_complete_and_executes_it():
    actor, root, worlds = setup()
    dyn = dynamics(actor, worlds)
    parent, candidate = programs(worlds, dyn)
    parent_receipt = IntervalEvaluator(worlds, dyn, parent).run(525)
    candidate_receipt = IntervalEvaluator(worlds, dyn, candidate).run(10000)
    selection = select_interval(parent_receipt, [candidate_receipt])
    assert parent_receipt.lower < parent_receipt.upper
    assert selection.gain_lower == Fraction(9, 275)
    assert candidate_receipt.lower == candidate_receipt.upper == Fraction(228, 275)
    verified = [
        verify_receipt(worlds, r, actor, {"fixture": actor}, "parent")
        for r in (parent_receipt, candidate_receipt)
    ]
    assert Fraction(verified[0]["value"]) == Fraction(204, 275)
    assert (
        Fraction(verified[1]["value"]) - Fraction(verified[0]["value"])
        >= selection.gain_lower
    )
    executor = IntervalExecutor(selection, actor, "parent")
    for _ in range(32):
        if root.state[PHASE] == TERMINAL:
            break
        root.step(
            executor.act(root)
            if root.state[PLAYER] == worlds.viewer
            else actor.act(root)
        )
    assert root.state[PHASE] == TERMINAL


@pytest.mark.parametrize("cap", [0, 100, 1000, 1258, 10000])
def test_shared_budget_is_not_multiplied_by_number_of_policies(cap):
    actor, _, worlds = setup()
    dyn = KoiTailDynamics(256, worlds.model_names, {"fixture": actor}, actor)
    result = compare_programs(
        worlds, dyn, programs(worlds, dyn), max_decisions=cap, quantum=256
    )
    assert result["decisions"] <= cap
    assert result["decisions"] == sum(r.work["decisions"] for r in result["receipts"])
    assert result["expanded_tasks"] <= 500000
    for r in result["receipts"]:
        verify_receipt(worlds, r, actor, {"fixture": actor}, "parent")
    if result["selection"].candidate is not None:
        assert result["selection"].gain_lower > 0
    if cap == 10000:
        assert result["reason"] == "best_candidate_certified"
        assert result["selection"].candidate is not None


def test_frontier_and_depth_refusals_keep_unresolved_mass():
    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    parent = programs(worlds, dyn)[0]
    for kwargs, phrase in (
        ({"max_frontier_rows": len(worlds.states)}, "frontier"),
        ({"max_depth": 1}, "depth"),
    ):
        evaluator = IntervalEvaluator(worlds, dyn, parent, **kwargs)
        r = evaluator.run(10000)
        assert phrase in r.refusal
        assert r.lower < r.upper
        verify_receipt(worlds, r, actor, {"fixture": actor}, "parent")
        assert evaluator.run(20000) == r
    evaluator = IntervalEvaluator(worlds, dyn, parent)
    r = evaluator.run(10000, max_nodes=0)
    assert r.work["decisions"] == 0 and r.lower == 0 and r.upper == 1
    assert "task" in r.refusal


def test_chunking_preserves_complete_values_and_physical_work():
    actor, _, worlds = setup()
    receipts = []
    for capacity in (64, 4096):
        dyn = KoiTailDynamics(capacity, worlds.model_names, {"fixture": actor}, actor)
        parent = programs(worlds, dyn)[0]
        receipts.append(
            IntervalEvaluator(worlds, dyn, parent, certificates=False).run(10000)
        )
    assert receipts[0].lower == receipts[1].lower == Fraction(204, 275)
    assert receipts[0].work["decisions"] == receipts[1].work["decisions"]


def test_rejects_changed_reference_binding_and_impossible_mass():
    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    parent, candidate = programs(worlds, dyn)
    p = IntervalEvaluator(worlds, dyn, parent).run(0)
    c = IntervalEvaluator(worlds, dyn, candidate).run(10000)
    assert select_interval(p, [c]).candidate is None
    with pytest.raises(ValueError, match="unchanged parent"):
        select_interval(c, [p])
    with pytest.raises(ValueError, match="mass ledger"):
        select_interval(replace(p, resolved_mass=p.total_mass + 1), [c])
    changed = replace(candidate, binding=replace(candidate.binding, digest="changed"))
    with pytest.raises(ValueError, match="belief"):
        IntervalEvaluator(worlds, dyn, changed)
    with pytest.raises(ValueError, match="belief"):
        select_interval(p, [replace(c, program=changed)])
    with pytest.raises(ValueError, match="root"):
        IntervalEvaluator(worlds, dyn, replace(parent, parent_root_action=47))
    with pytest.raises(ValueError, match="program"):
        replace(parent, overrides=parent.overrides * 2).validate()
    with pytest.raises(ValueError, match="parent identity"):
        IntervalExecutor(select_interval(p, [c]), actor, "another-parent")


def test_contingent_override_program_is_valued_as_a_whole():
    from scripts.score_bounded_tail import ScoreBoundPlanner

    actor, _, worlds = setup()
    dyn = dynamics(actor, worlds)
    proposed = ScoreBoundPlanner(dyn).solve(worlds, True).plan
    program = make_program(
        worlds, dyn, "parent", {"fixture": "fixture"}, proposed.policy
    )
    assert len(program.overrides) > 1 and not program.is_parent
    receipt = IntervalEvaluator(worlds, dyn, program).run(100000)
    assert receipt.lower == receipt.upper == Fraction(97, 110)
    verify_receipt(worlds, receipt, actor, {"fixture": actor}, "parent")


def test_illegal_future_override_has_the_same_parent_guard_in_every_interpreter():
    from scripts.belief_tail import visible_keys

    actor, root, worlds = setup()
    dyn = dynamics(actor, worlds)
    original = programs(worlds, dyn)[0]
    # Find a later learner decision on the real reference continuation.
    root.step(original.parent_root_action)
    while root.state[PHASE] != TERMINAL and root.state[PLAYER] != worlds.viewer:
        root.step(actor.act(root))
    assert root.state[PHASE] != TERMINAL
    key = visible_keys(root.state[None], worlds.viewer)[0]
    illegal = int(np.flatnonzero(~root.legal())[0])
    guarded = make_program(
        worlds, dyn, "parent", {"fixture": "fixture"}, {key: illegal}
    )
    evaluator = IntervalEvaluator(worlds, dyn, guarded, certificates=False)
    receipt = evaluator.run(10000)
    assert receipt.lower == receipt.upper == Fraction(204, 275)
    verify_receipt(worlds, receipt, actor, {"fixture": actor}, "parent")
    # Exercise the runtime semantics independently of the selection gate.
    parent = IntervalEvaluator(worlds, dyn, original).run(0)
    executor = IntervalExecutor(select_interval(parent, []), actor, "parent")
    executor.entries = guarded.validate()
    executor.started = True
    assert executor.act(root) == actor.act(root)
