"""Protect final-data gates and distinguish replicated wins from point gains."""

import copy
import json

import pytest

from scripts import audit_final_published_test as audit
from scripts import run_final_published_test as run


def development():
    decision = {
        "public_qualified": {"warm": True, "cold": False},
        "cold_paired_benefit": False,
        "selected_arm": "warm",
        "canonical_case": "seed6710001_warm",
    }
    checked = {
        "status": "passed",
        "decision": copy.deepcopy(decision),
        "endpoints": {
            f"seed{seed}_{arm}": {"score_ci95": [0.51 if arm == "warm" else 0.49, 0.55]}
            for seed in run.SEEDS
            for arm in ("warm", "cold")
        },
        "paired": {str(seed): {"score_ci95": [-0.02, 0.01]} for seed in run.SEEDS},
    }
    return checked, {"provisional_decision": decision}


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unpassed",
        "wrong_decision",
        "wrong_canonical",
        "only_one_seed",
        "boundary",
    ],
)
def test_selection_gate_rejects_incomplete_or_reinterpreted_evidence(fault):
    checked, summary = development()
    if fault == "unpassed":
        checked["status"] = "running"
    elif fault == "wrong_decision":
        summary["provisional_decision"]["selected_arm"] = "cold"
    elif fault == "wrong_canonical":
        checked["decision"]["canonical_case"] = "seed6720001_warm"
        summary["provisional_decision"]["canonical_case"] = "seed6720001_warm"
    elif fault in ("only_one_seed", "boundary"):
        checked["endpoints"]["seed6720001_warm"]["score_ci95"][0] = (
            0.49 if fault == "only_one_seed" else 0.5
        )
    if fault is None:
        assert run.selected_arm(checked, summary) == "warm"
    else:
        with pytest.raises(ValueError):
            run.selected_arm(checked, summary)


def test_preparation_cannot_create_final_files_before_full_development_audit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run, "BASE", tmp_path / "final")
    monkeypatch.setattr(run.cold, "BASE", tmp_path / "development")
    with pytest.raises(FileNotFoundError):
        run.prepare()
    assert not run.BASE.exists()


@pytest.mark.parametrize("fault", ["revision", "dirty"])
def test_comparator_checkout_is_rechecked_before_loading(monkeypatch, fault):
    def checked_output(command, **kwargs):
        if command[1] == "rev-parse":
            return "different" if fault == "revision" else run.cold.UPSTREAM
        return " M koikoigame.py" if fault == "dirty" else ""

    monkeypatch.setattr(run.subprocess, "check_output", checked_output)
    monkeypatch.setattr(
        run, "load_source", lambda: pytest.fail("Drifted comparator was loaded")
    )
    with pytest.raises(ValueError, match="clean and pinned"):
        run.upstream_inputs()


@pytest.mark.parametrize(
    "fault", [None, "metadata", "raw_seed", "nested_metadata", "nested_raw_seed"]
)
def test_overlap_scan_checks_raw_records_as_well_as_report_metadata(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(run, "BASE", tmp_path / "own_final")
    report = {
        "seed_start": 7200000,
        "pairs": 2,
        "records": [{"seed": 7200000}, {"seed": 7200001}],
    }
    if fault in ("metadata", "nested_metadata"):
        report["seed_start"] = run.FIRST - 1
    elif fault in ("raw_seed", "nested_raw_seed"):
        report["records"][1]["seed"] = run.FIRST + 7
    if fault and fault.startswith("nested"):
        report = {"archived": [report]}
    (tmp_path / "previous.json").write_text(json.dumps(report), encoding="utf-8")
    scan = run.seed_scan(tmp_path)
    assert bool(scan["overlaps"]) == (fault is not None)
    assert len(scan["reports"]) == 1
    if fault is None:
        audit.assert_no_reserved_deals(report)
    else:
        with pytest.raises(AssertionError):
            audit.assert_no_reserved_deals(report)


def jobs():
    return {
        f"seed{seed}_warm__{variant}": {
            "case": f"seed{seed}_warm",
            "variant": variant,
            "opponent": f"rl-{variant}-assisted",
        }
        for variant in ("wp", "point")
        for seed in run.SEEDS
    }


@pytest.mark.parametrize("encoding", ["utf-16", "utf-8-sig"])
def test_historical_bom_report_is_checked_without_reencoding(
    tmp_path, monkeypatch, encoding
):
    monkeypatch.setattr(run, "BASE", tmp_path / "own_final")
    path = tmp_path / "historical.json"
    content = json.dumps({"seed_start": run.FIRST, "pairs": 1}).encode(encoding)
    path.write_bytes(content)
    assert run.seed_scan(tmp_path)["overlaps"] == [str(path)]
    assert path.read_bytes() == content


def test_final_command_explicitly_uses_fixed_reserved_split_and_checkpoint():
    p = {
        "jobs": jobs(),
        "models": {
            f"seed{s}_warm": {"checkpoint": f"frozen/{s}.pt"} for s in run.SEEDS
        },
    }
    command = run.evaluation_argv(p, "seed6720001_warm__point")
    for flag, expected in (
        ("--agent", "frozen/6720001.pt"),
        ("--split", "test"),
        ("--seed", "100000000"),
        ("--pairs", "10000"),
        ("--opponent", "rl-point-assisted"),
    ):
        assert command[command.index(flag) + 1] == expected


@pytest.mark.parametrize(
    "primary,secondary",
    [
        ((0.51, 0.51), (0.51, 0.51)),
        ((0.51, 0.5), (0.52, 0.52)),
        ((0.51, 0.51), (0.49, 0.53)),
        ((0.49, 0.53), (0.49, 0.53)),
    ],
)
def test_primary_secondary_and_replication_are_separate(primary, secondary):
    endpoints = {
        f"seed{seed}_warm__{variant}": {
            "win_ci95": [values[i], 0.6],
            "point_difference": 100,
        }
        for variant, values in (("wp", primary), ("point", secondary))
        for i, seed in enumerate(run.SEEDS)
    }
    expected = {
        "primary_wp_replicated": min(primary) > 0.5,
        "secondary_point_replicated": min(secondary) > 0.5,
    }
    assert run.conclusions(endpoints, jobs()) == expected
    independent = {name: {"score_ci95": r["win_ci95"]} for name, r in endpoints.items()}
    assert audit.conclusions(independent, jobs()) == expected


def test_missing_endpoint_never_counts_as_replicated_success():
    incomplete = {"only": {"variant": "wp"}}
    with pytest.raises(ValueError):
        run.conclusions({"only": {"win_ci95": [0.6, 0.7]}}, incomplete)
    with pytest.raises(AssertionError):
        audit.conclusions({"only": {"score_ci95": [0.6, 0.7]}}, incomplete)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failure",
        "not_finished",
        "wrong_resource",
        "before_prepared",
        "missing_job",
    ],
)
def test_independent_schedule_requires_all_four_completed_cpu_jobs(fault):
    records = {
        name: {
            "status": "complete",
            "returncode": 0,
            "resource": "cpu",
            "cpu_threads": 2,
            "pid": 100 + i,
            "started": 11 + i,
            "finished": 20 + i,
        }
        for i, name in enumerate(jobs())
    }
    manifest = {
        "status": "complete_awaiting_audit",
        "started": 10,
        "finished": 30,
        "jobs": records,
    }
    first = next(iter(records.values()))
    if fault == "failure":
        first["returncode"] = 1
    elif fault == "not_finished":
        first["status"] = "running"
    elif fault == "wrong_resource":
        first["resource"] = "gpu"
    elif fault == "before_prepared":
        manifest["started"] = 0
    elif fault == "missing_job":
        records.pop(next(iter(records)))
    if fault is None:
        audit.schedule(manifest, jobs(), 5)
    else:
        with pytest.raises(AssertionError):
            audit.schedule(manifest, jobs(), 5)


def test_worker_failure_cancels_only_owned_process_trees(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "BASE", tmp_path)
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    processes, stopped = [], []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(processes)
            self.code = 1 if not processes else None
            processes.append(self)

        def poll(self):
            return self.code

        def wait(self):
            assert self.code is not None
            return self.code

        def terminate(self):
            self.code = -9
            stopped.append(self.pid)

    def stop_tree(command, **kwargs):
        assert command[:2] == ["taskkill", "/PID"] and command[3:] == ["/T", "/F"]
        next(p for p in processes if p.pid == int(command[2])).terminate()

    monkeypatch.setattr(run.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(run.subprocess, "run", stop_tree)
    with pytest.raises(RuntimeError, match="failed"):
        run.supervise({"jobs": jobs(), "parallel_workers": 4})
    assert stopped == [101, 102, 103]
    manifest = json.loads((tmp_path / "process.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert sum(r["status"] == "aborted" for r in manifest["jobs"].values()) == 3


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "dependency_drift",
        "missing_binding",
        "wrong_comparator",
        "wrong_model",
        "source_drift",
        "development_split",
        "protocol_drift",
    ],
)
def test_frozen_inputs_are_rechecked_before_game_execution(
    tmp_path, monkeypatch, fault
):
    folder = tmp_path / "final"
    folder.mkdir()
    monkeypatch.setattr(run, "BASE", folder)
    dependency, teacher = tmp_path / "model.bin", tmp_path / "teacher.bin"
    dependency.write_bytes(b"model")
    teacher.write_bytes(b"teacher")
    dev_bindings = {str(dependency): run.sha(dependency)}
    upstream_bindings = {str(teacher): run.sha(teacher)}
    models = {f"seed{s}_warm": {"checkpoint": f"frozen/{s}.pt"} for s in run.SEEDS}
    development_info = {"selected_arm": "warm", "canonical_case": "seed6710001_warm"}
    comparators = {"point": {"discard": run.sha(teacher)}}
    monkeypatch.setattr(
        run, "development_inputs", lambda: (development_info, models, dev_bindings)
    )
    monkeypatch.setattr(
        run, "upstream_inputs", lambda: (comparators, upstream_bindings)
    )
    monkeypatch.setattr(
        run,
        "provenance",
        lambda: {
            "source_sha256": "wrong" if fault == "source_drift" else run.cold.SOURCE
        },
    )
    p = {
        "runner_sha256": run.sha(run.__file__),
        "auditor_sha256": run.sha(run.AUDITOR),
        "evaluation": copy.deepcopy(run.EVALUATION),
        "parallel_workers": 4,
        "provenance": {"source_sha256": run.cold.SOURCE},
        "upstream_revision": run.cold.UPSTREAM,
        "bindings": dev_bindings | upstream_bindings,
        "comparator_hashes": copy.deepcopy(comparators),
        "development": development_info,
        "models": copy.deepcopy(models),
        "jobs": jobs(),
        "seed_scan": {
            "range": [run.FIRST, run.FIRST + run.PAIRS - 1],
            "overlaps": [],
            "reports": {},
        },
    }
    if fault == "dependency_drift":
        dependency.write_bytes(b"different model")
    elif fault == "missing_binding":
        del p["bindings"][str(teacher)]
    elif fault == "wrong_comparator":
        p["comparator_hashes"]["point"]["discard"] = "other"
    elif fault == "wrong_model":
        p["models"]["seed6710001_warm"]["checkpoint"] = "different.pt"
    elif fault == "development_split":
        p["evaluation"]["split"] = "development"
    elif fault == "protocol_drift":
        run.write(folder / "process.json", {"protocol_sha256": "wrong"})
    run.write(folder / "protocol.json", p)
    if fault is None:
        assert run.validate_frozen() == p
    else:
        with pytest.raises(AssertionError):
            run.validate_frozen()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "development",
        "wrong_opponent",
        "wrong_model",
        "changed_source",
        "wrong_summary",
    ],
)
def test_independent_final_report_checks_test_identity_and_recomputes_results(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(audit, "PAIRS", 2)
    model = "frozen/model.pt"
    payload = {
        "config": {"seed": 1},
        "counters": {"updates": 520},
        "provenance": {"source_sha256": audit.cold.SOURCE},
    }
    counts = {
        "discard": 2,
        "pick_hand": 0,
        "pick_draw": 0,
        "koi": 0,
        "continued": 0,
        "stopped": 0,
    }
    r = {
        "protocol_version": 2,
        "profile": "paper8",
        "split": "test",
        "opponent": "rl-point-assisted",
        "seed_start": audit.FIRST,
        "pairs": 2,
        "matches": 4,
        "device": "cpu",
        "deterministic": True,
        "endgame": True,
        "opponent_endgame": True,
        "search": "none",
        "koi_agent": None,
        "upstream_revision": audit.cold.UPSTREAM,
        "agent": model,
        "provenance": {"source_sha256": audit.cold.SOURCE},
        "checkpoints": {
            model: {
                "sha256": "model-digest",
                **{"training_" + k: copy.deepcopy(v) for k, v in payload.items()},
            }
        },
        "agent_decisions": counts,
        "opponent_decisions": counts,
        "decisions": 4,
        "draw_adjusted_win_rate": 0.5,
        "win_rate_ci95": [0.5, 0.5],
        "mean_point_difference": 0,
        "point_difference_ci95": [0, 0],
        "records": [
            {
                "seed": audit.FIRST + i,
                "seat0_score": 1,
                "seat1_score": 0,
                "seat0_point_difference": 2,
                "seat1_point_difference": -2,
            }
            for i in range(2)
        ],
    }
    for key in (
        "search_simulated_decisions",
        "search_executed_decisions",
        "search_preparation_decisions",
        "searched_decisions",
        "changed_decisions",
    ):
        r[key] = 0
    if fault == "development":
        r["split"] = "development"
    elif fault == "wrong_opponent":
        r["opponent"] = "rl-wp-assisted"
    elif fault == "wrong_model":
        r["checkpoints"][model]["sha256"] = "different"
    elif fault == "changed_source":
        r["provenance"]["source_sha256"] = "different"
    elif fault == "wrong_summary":
        r["draw_adjusted_win_rate"] = 0.7
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    args = (
        path,
        {"opponent": "rl-point-assisted"},
        payload,
        {"sha256": "model-digest"},
        model,
    )
    if fault is None:
        result = audit.report(*args)
        assert result["score_mean"] == 0.5 and result["point_mean"] == 0
    else:
        with pytest.raises(AssertionError):
            audit.report(*args)
