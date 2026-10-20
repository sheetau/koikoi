from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from scripts import certified_tail_replay as study
from scripts.belief_tail import KoiTailDynamics
from scripts.opponent_last_turn_tail import OpponentLastTurnPlanner
from scripts.prospective_incumbent import evaluate_arm
from scripts.run_arbitration_study import read, sha, write
from tests.test_opponent_last_turn_tail import setup


def test_equal_caps_and_complete_policy_equivalence_with_independent_oracle():
    torch.set_num_threads(2)
    actor, _, worlds = setup()
    protocol = {"phase_nodes": 500000, "phase_decisions": 50, "depth": 32}
    for arm in ("controller", "root_only"):

        def evaluate(planner=None, arm=arm):
            dynamics = KoiTailDynamics(
                4096, worlds.model_names, {"fixture": actor}, actor
            )
            kwargs = {} if planner is None else {"planner_type": planner}
            phases, selection, _ = evaluate_arm(
                worlds,
                dynamics,
                "parent",
                {"fixture": "fixture"},
                arm,
                protocol,
                **kwargs,
            )
            return dynamics, phases, selection

        _, old, old_selection = evaluate()
        dynamics, new, selection = evaluate(OpponentLastTurnPlanner)
        assert sum(p.result.work["decisions"] for p in new.values()) <= 100
        assert old_selection.parent_q is None
        assert selection.parent_q is not None and selection.plan is None
        assert study.compare_common(old, new) == 0
        # Explicitly exercise the two-policy replay path without presenting a
        # non-improving fixture as an actual adoption.
        plan = new["fixed"].result.plan
        oracle = study.verify_adoption(
            worlds,
            new,
            replace(selection, plan=plan),
            {"fixture": actor},
            dynamics.policy_cache,
        )
        assert oracle["parent"]["values"][str(plan.baseline_action)] == "7/25"
        assert oracle["parent"]["new_opponent_adapter_calls"] >= 0


def test_common_policy_comparison_rejects_value_and_map_changes():
    actor, _, worlds = setup()
    protocol = {"phase_nodes": 500000, "phase_decisions": 2000000, "depth": 32}
    outputs = []
    for planner in (None, OpponentLastTurnPlanner):
        dynamics = KoiTailDynamics(4096, worlds.model_names, {"fixture": actor}, actor)
        kwargs = {} if planner is None else {"planner_type": planner}
        phases, _, _ = evaluate_arm(
            worlds,
            dynamics,
            "parent",
            {"fixture": "fixture"},
            "controller",
            protocol,
            **kwargs,
        )
        outputs.append(phases)
    assert study.compare_common(*outputs) > 0
    altered = deepcopy(outputs[1])
    plan = altered["fixed"].result.plan
    plan.values[plan.action] += 1
    with pytest.raises(AssertionError):
        study.compare_common(outputs[0], altered)
    altered = deepcopy(outputs[1])
    plan = altered["fixed"].result.plan
    plan.root_policies[plan.action][b"extraneous"] = 0
    with pytest.raises(AssertionError):
        study.compare_common(outputs[0], altered)


def test_freeze_requires_complete_audited_source_and_checks_bindings(
    tmp_path, monkeypatch
):
    source, base = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    original = {"inputs": {}}
    monkeypatch.setattr(study, "check_source", lambda _: original)
    write(source / "protocol.json", original)
    write(source / "queue.json", {"status": "running"})
    summary = {
        "status": "complete",
        "sources": 512,
        "streams": 128,
        "chunk_bindings": {},
    }
    write(source / "summary.json", summary)
    receipt = tmp_path / "export.json"
    write(
        receipt,
        {
            "status": "passed",
            "summary_sha256": sha(source / "summary.json"),
            "bindings": {},
        },
    )
    with pytest.raises(AssertionError):
        study.freeze(base, source, receipt)
    assert not base.exists()
    write(source / "queue.json", {"status": "complete"})
    study.freeze(base, source, receipt)
    study.check(base)
    write(receipt, {"status": "tampered"})
    with pytest.raises(AssertionError):
        study.check(base)
    with pytest.raises(ValueError):
        study.freeze(base, source, receipt)


def test_source_record_digest_is_checked(tmp_path):
    write(tmp_path / "records.json", [{"index": 1}])
    write(
        tmp_path / "complete.json", {"records_sha256": sha(tmp_path / "records.json")}
    )
    assert study.source_records(tmp_path) == [{"index": 1}]
    write(tmp_path / "records.json", [])
    with pytest.raises(AssertionError):
        study.source_records(tmp_path)


def test_complete_only_audit_of_all_missing_roots(tmp_path, monkeypatch):
    from scripts.prospective_incumbent import cases

    source, candidate = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    original = {"chunk_size": 16, "cases": cases()}
    protocol = {
        "source": str(source),
        "chunks": list(range(32)),
        "scope": "Synthetic fixture",
    }
    monkeypatch.setattr(study, "check", lambda _: (protocol, original))
    write(candidate / "protocol.json", protocol)
    write(candidate / "queue.json", {"status": "running"})
    with pytest.raises(AssertionError):
        study.audit_and_summarize(candidate)
    write(candidate / "queue.json", {"status": "complete"})
    write(source / "summary.json", {"contrast_order": ["a", "b", "c", "d"]})
    for chunk in range(32):
        rows = [
            {
                "index": i,
                "status": "missing_root",
                "source": {**case, "final": [30] * 240},
            }
            for i, case in enumerate(cases())
            if i // 16 == chunk
        ]
        for root in (source, candidate):
            folder = root / f"chunk_{chunk:03}"
            write(folder / "records.json", rows)
            write(folder / "process.json", {"status": "complete"})
            write(
                folder / "complete.json",
                {
                    "status": "complete",
                    "records_sha256": sha(folder / "records.json"),
                    "protocol_sha256": sha(candidate / "protocol.json"),
                    "artifacts": {},
                },
            )
    result = study.audit_and_summarize(candidate)
    assert result["new_source_matches"] == 0 and result["reused_sources"] == 512
    assert result["certified_actual_mean"] == [0] * 4
    assert result["work"]["actual_tails"] == {}
    assert "development_strength_gate" not in result
    assert read(candidate / "summary.json") == result


def test_worker_reuses_bound_worlds_and_saves_verified_actual_tails(
    tmp_path, monkeypatch
):
    from koikoi.layout import PHASE, TERMINAL
    from scripts.check_belief_tail import add_step, empty_work
    from scripts.export_prospective_incumbent import audit_arm
    from scripts.prospective_incumbent import assess_three, cases, describe_selection
    from scripts.run_arbitration_study import PARENTS

    actor, root, worlds = setup()
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    original = {
        "phase_nodes": 500000,
        "phase_decisions": 50,
        "depth": 32,
        "capacity": 4096,
        "opponent_identities": {"fixture": "fixture"},
    }
    protocol = {"source": str(source), "chunks": [0]}
    monkeypatch.setattr(study, "check", lambda _: (protocol, original))
    monkeypatch.setattr(study.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(study, "NeuralAgent", lambda *args, **kwargs: actor)
    monkeypatch.setattr(study, "PublishedAgent", lambda name: actor)
    # The study's real identity names are wp/point. This synthetic actor is
    # deliberately bound under wp; keep the fixture-world identity separate.
    monkeypatch.setattr(study, "EndgameAgent", lambda value: value)
    worlds = replace(worlds, model_names=("wp",))
    original["opponent_identities"] = {"wp": "fixture"}
    env = root.clone()
    actions, work = [], empty_work()
    while env.state[PHASE] != TERMINAL:
        action = actor.act(env)
        actions.append(action)
        add_step(work, env.step(action))
    src = {
        **cases()[0],
        "parent": 0,
        "seat": worlds.viewer,
        "opponent": "wp",
        "selected_root": {
            "state": root.state.tolist(),
            "decision": 0,
            "public_identification": {"model": "wp"},
        },
        "actions": actions,
        "final": env.state.tolist(),
    }
    row = {
        "index": 0,
        "source": src,
        "status": "evaluated",
        "order": ["controller", "root_only"],
        "arms": {},
    }
    folder = source / "chunk_000"
    folder.mkdir()
    artifacts, selections = {}, {}
    for arm in row["order"]:
        dynamics = KoiTailDynamics(4096, worlds.model_names, {"wp": actor}, actor)
        phases, selection, timing = evaluate_arm(
            worlds, dynamics, PARENTS[0][1], {"wp": "fixture"}, arm, original
        )
        record = {
            "selection": describe_selection(selection),
            "seconds": timing,
            "oracle": {},
            "parent_adapter_calls": dynamics.baseline_calls,
            "opponent_adapter_calls": dynamics.policy_calls,
            "phases": {
                name: {"work": p.result.work, "refusal": p.result.refusal}
                for name, p in phases.items()
            },
            "artifact": f"{arm}.pkl.gz",
        }
        evidence = {"worlds": worlds, "phases": phases, "selection": selection}
        study.compressed_save(folder / record["artifact"], evidence)
        artifacts[record["artifact"]] = sha(folder / record["artifact"])
        row["arms"][arm] = record
        selections[arm] = selection
    row["assessment"] = assess_three(src, selections, actor, actor, PARENTS[0][1])
    write(folder / "records.json", [row])
    write(
        folder / "complete.json",
        {
            "records_sha256": sha(folder / "records.json"),
            "artifacts": artifacts,
        },
    )
    write(candidate / "protocol.json", protocol)
    study.worker(candidate, 0)
    result_folder = candidate / "chunk_000"
    assert read(result_folder / "process.json")["status"] == "complete"
    result = study.source_records(result_folder)[0]
    assert result["source"] == src
    assert result["assessment"] == row["assessment"]
    for arm, record in result["arms"].items():
        evidence = study.load(result_folder / record["artifact"])
        selection = audit_arm(evidence, record, src, arm, original)
        assert selection.parent_q is not None and selection.plan is None
    with pytest.raises(FileExistsError):
        study.worker(candidate, 0)
    # Exercise the complete exporter with one real native-worker result and
    # explicitly synthetic no-root rows filling the fixed cohort.
    original.update(chunk_size=16, cases=cases())
    protocol.update(chunks=list(range(32)), scope="Synthetic accounting fixture")
    write(candidate / "protocol.json", protocol)
    write(candidate / "queue.json", {"status": "complete"})
    write(source / "summary.json", {"contrast_order": ["a", "b", "c", "d"]})
    for chunk in range(32):
        fallback = [
            {
                "index": i,
                "status": "missing_root",
                "source": {**case, "final": [30] * 240},
            }
            for i, case in enumerate(cases())
            if i // 16 == chunk
        ]
        for directory, first in ((source, row), (candidate, result)):
            target = directory / f"chunk_{chunk:03}"
            existing = (
                read(target / "complete.json") if chunk == 0 else {"artifacts": {}}
            )
            entries = [first, *fallback[1:]] if chunk == 0 else fallback
            write(target / "records.json", entries)
            write(target / "process.json", {"status": "complete"})
            write(
                target / "complete.json",
                {
                    **existing,
                    "status": "complete",
                    "protocol_sha256": sha(candidate / "protocol.json"),
                    "records_sha256": sha(target / "records.json"),
                },
            )
    audited = study.audit_and_summarize(candidate)
    assert audited["costs"]["original"]["controller"]["parent_valued"] == 0
    assert audited["costs"]["certified"]["controller"]["parent_valued"] == 1
    assert audited["work"]["actual_tails"]["decisions"] == 3 * work["decisions"]
    assert audited["work"]["actual_tails"] == audited["work"]["export_tail_replay"]
    # Even rehashing a modified record cannot make a false outcome pass replay.
    entries = read(result_folder / "records.json")
    entries[0]["assessment"]["outcomes"]["controller"]["margin"] += 1
    write(result_folder / "records.json", entries)
    complete = read(result_folder / "complete.json")
    complete["records_sha256"] = sha(result_folder / "records.json")
    write(result_folder / "complete.json", complete)
    with pytest.raises(AssertionError):
        study.audit_and_summarize(candidate)
