"""Regression for forced skips and exact retention of already acquired sources."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import frontier_scope_controller as controller
from scripts.frontier_incremental_design import cases as original_cases
from scripts.frontier_incremental_design import specification as original_specification
from scripts.frontier_retained_sources import (
    audit_acquisition,
    bindings,
    collect_case,
    manifest,
)
from scripts.frontier_scope_assessment import (
    assess_pair,
    audit_plan,
    plan_source,
    validate_program,
)
from scripts.frontier_scope_design import cases, specification
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.run_arbitration_study import PARENTS, read, write
from scripts.symbolic_final_turn_tail import SymbolicFinalTurnPlanner
from scripts.uniform_controller_study import actors
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("margin", [-40, 6, 40])
def test_shorter_gate_preserves_original_phases_and_selection(margin):
    actor, _, worlds = setup(margin=margin)
    assert worlds.draws < 5
    hybrid = controller.evaluate_scoped_frontier(
        worlds, dynamics(actor, worlds), "parent", {"fixture": "fixture"}
    )
    old = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        max_nodes=500000,
        max_decisions_per_phase=2000000,
        planner_type=SymbolicFinalTurnPlanner,
    )
    assert (hybrid.fixed, hybrid.optimized, hybrid.original) == old
    assert hybrid.patch is None
    validate_program(hybrid, worlds, "parent", {"fixture": "fixture"}, specification())


def test_shorter_unvalued_parent_also_keeps_old_fallback(monkeypatch):
    actor, _, worlds = setup()
    config = {**controller.configuration(), "phase_decisions": 500}
    monkeypatch.setattr(controller, "configuration", lambda: config)
    hybrid = controller.evaluate_scoped_frontier(
        worlds, dynamics(actor, worlds), "parent", {"fixture": "fixture"}
    )
    assert hybrid.original.reason == "parent_unvalued" and hybrid.patch is None
    assert hybrid.decisions == 465
    validate_program(
        hybrid,
        worlds,
        "parent",
        {"fixture": "fixture"},
        {**specification(), "hybrid": config},
    )


def test_exactly_five_dispatches_to_unchanged_frozen_method(monkeypatch):
    calls = []
    sentinel = object()

    def selected(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(controller, "evaluate_selected_frontier", selected)
    worlds = SimpleNamespace(draws=5)
    assert controller.evaluate_scoped_frontier(worlds, None, "parent", {}) is sentinel
    assert calls == [(worlds, None, "parent", {})]
    with pytest.raises(ValueError, match="at-most-five"):
        controller.evaluate_scoped_frontier(
            SimpleNamespace(draws=6), None, "parent", {}
        )


def test_cohort_and_statistical_design_do_not_change():
    assert cases() == original_cases()
    old, new = original_specification(), specification()
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {"version", "boundary", "extension"}
    assert set(new) - set(old) == {"amendment"}


def test_all_retained_sources_are_bound_copied_and_never_replaced(tmp_path):
    assert len(manifest()["sources"]) == 15 and bindings()
    for row in manifest()["sources"]:
        folder = tmp_path / str(row["case"]["index"])
        source = collect_case(
            folder, row["case"], specification(), None, None, None, None
        )
        assert source == read(Path(row["folder"]) / "record.json")
        audit_acquisition(folder, row["case"])
        with pytest.raises(FileExistsError):
            collect_case(folder, row["case"], specification(), None, None, None, None)
    row = manifest()["sources"][0]
    folder = tmp_path / str(row["case"]["index"])
    seat = read(folder / "seat.json")
    seat["seat"] = 1 - seat["seat"]
    write(folder / "seat.json", seat)
    with pytest.raises(ValueError, match="replaced or modified"):
        audit_acquisition(folder, row["case"])


def test_first_observed_scope_failure_now_has_zero_increment_and_identical_execution(
    tmp_path,
):
    # Source 20 is the smallest-index observed scope exception; it is retained
    # in the original cohort. This is a disclosed repair check, not a new source.
    row = next(r for r in manifest()["sources"] if r["case"]["index"] == 20)
    source = collect_case(
        tmp_path / "source", row["case"], specification(), None, None, None, None
    )
    parents, opponents, _ = actors()
    protocol = {
        "design": specification(),
        "opponent_identities": read(
            "runs/frontier_incremental_evaluation/protocol.json"
        )["opponent_identities"],
    }
    planned, evidence = plan_source(source, parents, opponents, protocol)
    assert evidence["worlds"].draws == 4 and evidence["hybrid"].patch is None
    parent = parents[source["parent"]]
    checked, _ = audit_plan(planned, evidence, source, parent, opponents, protocol)
    assert checked == {
        "exact_increment": "0",
        "reason": "identical_programs",
        "oracle": None,
    }
    result = assess_pair(
        source,
        evidence["hybrid"],
        parent,
        opponents[source["opponent"]],
        PARENTS[source["parent"]][1],
    )
    assert result["outcomes"]["old"] == result["outcomes"]["hybrid"]
    assert result["actual_work"] == result["replay_work"]
