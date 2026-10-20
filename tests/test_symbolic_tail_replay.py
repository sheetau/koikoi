from dataclasses import replace

import pytest
import torch

from scripts import symbolic_tail_replay as study
from scripts.belief_tail import KoiTailDynamics
from scripts.opponent_last_turn_tail import OpponentLastTurnPlanner
from scripts.prospective_incumbent import evaluate_arm
from scripts.run_arbitration_study import read, sha, write
from tests.test_symbolic_final_turn_tail import setup


@pytest.mark.parametrize("swap", [False, True])
def test_equal_caps_recover_an_independently_verified_improvement(swap):
    torch.set_num_threads(2)
    actor, _, worlds = setup(swap)
    protocol = {"phase_nodes": 500000, "phase_decisions": 1258, "depth": 32}
    for arm in ("controller", "root_only"):
        outputs = []
        for planner in (OpponentLastTurnPlanner, study.SymbolicFinalTurnPlanner):
            dynamics = KoiTailDynamics(
                4096, worlds.model_names, {"fixture": actor}, actor
            )
            phases, selection, _ = evaluate_arm(
                worlds,
                dynamics,
                "parent",
                {"fixture": "fixture"},
                arm,
                protocol,
                planner_type=planner,
            )
            outputs.append((phases, selection))
        old, previous = outputs[0]
        new, selected = outputs[1]
        assert previous.parent_q is None
        assert str(selected.parent_q) == "204/275"
        assert str(selected.gain) == ("53/550" if arm == "controller" else "24/275")
        assert sum(p.result.work["decisions"] for p in new.values()) <= 2516
        study.check_programs(new)
        assert study.compare_common(old, new)["common_completed_root_values"] == 0
        oracle = study.verify_adoption(
            worlds,
            new,
            selected,
            actor,
            {"fixture": actor},
            dynamics.policy_cache,
        )
        assert list(oracle["parent"]["values"].values()) == ["204/275"]
        assert list(oracle["selected"]["values"].values()) == [str(selected.selected_q)]
        assert oracle["selected"]["new_parent_adapter_calls"] >= 0
        with pytest.raises(ValueError, match="recognized"):
            fixed = new["fixed"]
            study.check_programs(
                {
                    "fixed": replace(
                        fixed,
                        result=replace(
                            fixed.result,
                            plan=replace(fixed.result.plan, leaf_rule="unknown"),
                        ),
                    )
                }
            )


def test_freeze_requires_complete_audited_source_and_checks_bindings(
    tmp_path, monkeypatch
):
    source, base = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    original = {"inputs": {}}
    monkeypatch.setattr(study, "check_source", lambda _: (original, original))
    write(source / "protocol.json", original)
    write(source / "queue.json", {"status": "running"})
    summary = {
        "status": "passed",
        "reused_sources": 512,
        "new_source_matches": 0,
        "protocol_sha256": sha(source / "protocol.json"),
        "chunk_bindings": {},
    }
    write(source / "summary.json", summary)
    receipt = tmp_path / "export.json"
    write(
        receipt,
        {
            "status": "passed",
            "summary_sha256": sha(source / "summary.json"),
            "chunk_bindings": {},
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
    assert result["symbolic_actual_mean"] == [0] * 4
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

    actor, root, worlds = setup(swap=True)
    source, candidate = tmp_path / "source", tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    original = {
        "phase_nodes": 500000,
        "phase_decisions": 1258,
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
            worlds,
            dynamics,
            PARENTS[0][1],
            {"wp": "fixture"},
            arm,
            original,
            planner_type=OpponentLastTurnPlanner,
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
    assert (
        result["assessment"]["outcomes"]["parent"]
        == row["assessment"]["outcomes"]["parent"]
    )
    assert result["assessment"]["calls"]["symbolic_parent_adapter_calls"] > 0
    for arm, record in result["arms"].items():
        evidence = study.load(result_folder / record["artifact"])
        selection = audit_arm(evidence, record, src, arm, original)
        assert str(selection.parent_q) == "204/275" and selection.plan is not None
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
    assert audited["costs"]["opponent_only"]["controller"]["parent_valued"] == 0
    assert audited["costs"]["symbolic"]["controller"]["parent_valued"] == 1
    assert audited["work"]["actual_tails"]["completions"] == 3
    assert audited["work"]["oracle_inference"]["new_parent_adapter_calls"] >= 0
    assert audited["work"]["actual_inference"]["symbolic_parent_adapter_calls"] > 0
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
