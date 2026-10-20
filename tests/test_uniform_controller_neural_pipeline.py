"""Eight pinned fixed-deck pipeline fixtures, not fresh evaluation samples."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from koikoi.env import BatchEnv
from scripts import uniform_controller_study as study
from scripts.run_arbitration_study import read, sha
from scripts.uniform_controller_design import specification
from scripts.uniform_deal_source import DeckSample
from tests.test_logged_source_posterior import INDICES, PROTOCOL, recover_decks


@pytest.fixture(scope="module")
def pinned_pipeline():
    base = Path("runs/prospective_incumbent")
    assert sha(base / "protocol.json") == PROTOCOL
    protocol = read(base / "protocol.json")
    actors = study.actors()
    for path, digest in study.PARENTS:
        assert sha(path) == digest == protocol["inputs"][path]
    assert study.opponent_identities() == protocol["opponent_identities"]
    metrics = defaultdict(Counter)
    rows = []
    yield base, protocol, actors, metrics, rows
    print(
        json.dumps(
            {
                "scope": "eight fixed-deck pipeline fixtures",
                "work": dict(metrics),
                "fixtures": rows,
            },
            sort_keys=True,
        )
    )


@pytest.mark.parametrize("index", INDICES)
def test_fixed_neural_source_plan_true_parent_oracle_and_audit(
    pinned_pipeline, tmp_path, index
):
    base, protocol, (parents, opponents, predictors), metrics, rows = pinned_pipeline
    folder = base / f"chunk_{index // 16:03}"
    summary = read(base / "summary.json")
    assert (
        sha(folder / "complete.json")
        == summary["chunk_bindings"][str(folder / "complete.json")]
    )
    assert (
        sha(folder / "records.json") == read(folder / "complete.json")["records_sha256"]
    )
    old = read(folder / "records.json")[index % 16]["source"]
    old_case = protocol["cases"][index]
    case = {
        "index": index,
        "population": f"{old_case['parent']}:{old_case['opponent']}",
        "replicate": 0,
    }
    # The historical seed preserves its public initial dealer for this fixed tape.
    # Production uses seed1 and independent entropy; this test samples neither.
    config = {**specification(), "environment_seed": old_case["seed"]}
    tape, recovery = recover_decks(old)
    decks = iter(tape)
    parent, opponent = parents[old_case["parent"]], opponents[old_case["opponent"]]
    source = study.collect_case(
        tmp_path / "source",
        case,
        config,
        parent,
        opponent,
        predictors,
        BatchEnv(1, debug=True),
        seat_draw=lambda n: old_case["seat"],
        deck_draw=lambda: DeckSample(next(decks), 1, 47),
    )
    for field in ("actions", "final", "selected_root", "work"):
        assert source[field] == old[field]
    study.source_files(tmp_path / "source", case, source, config)
    checked = study.audit_source(source, predictors, parent, opponent)
    local_protocol = {
        "design": config,
        "opponent_identities": protocol["opponent_identities"],
    }
    row, evidence = study.plan_source(source, parents, opponents, local_protocol)
    expected_base = Path("runs/symbolic_tail_replay")
    expected_folder = expected_base / f"chunk_{index // 16:03}"
    expected_summary = read(expected_base / "summary.json")
    assert (
        sha(expected_folder / "complete.json")
        == expected_summary["chunk_bindings"][str(expected_folder / "complete.json")]
    )
    assert (
        sha(expected_folder / "records.json")
        == read(expected_folder / "complete.json")["records_sha256"]
    )
    expected = read(expected_folder / "records.json")[index % 16]["arms"]["controller"][
        "selection"
    ]
    assert row["selection"] == expected
    selection, history = study.audit_plan(
        row, evidence, source, opponents, local_protocol
    )
    result = study.assess_pair(
        source, selection, parent, opponent, study.PARENTS[source["parent"]][1]
    )
    for label, counters in (
        ("seeded_tape_recovery", recovery),
        ("source", source["work"]),
        ("native_source", source["native_work"]),
        ("source_audit", checked["replay_work"]),
        ("source_actor_calls", checked["policy_verification_calls"]),
        ("audit_actor_calls", checked["policy_verification_calls"]),
        ("source_identification", source["identification_calls"]),
        ("audit_identification", checked["identification_calls"]),
        ("history_support", row["history_counts"]),
        ("audit_history", history),
        ("actual_tails", result["actual_work"]),
        ("actual_replay", result["replay_work"]),
        ("actual_inference", result["calls"]),
        (
            "planning_inference",
            {
                "parent": row["parent_adapter_calls"],
                "opponent": row["opponent_adapter_calls"],
            },
        ),
    ):
        metrics[label].update(counters)
    for phase in row["phases"].values():
        metrics["planning"].update(phase["work"])
    for oracle in row["oracle"].values():
        metrics["oracle_physical"].update(oracle["physical_work"])
        metrics["oracle_represented"].update(oracle["represented_work"])
        metrics["oracle_inference"].update(oracle["new_actor_calls"])
    rows.append(
        {
            "index": index,
            "gain": row["selection"]["gain"],
            "reason": row["selection"]["reason"],
        }
    )
