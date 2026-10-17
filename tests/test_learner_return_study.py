"""Real CPU pipeline, frozen conditions and cross-study resource gates."""

import copy
from pathlib import Path

import pytest

from scripts import run_learner_return_study as study
from scripts.check_learner_gpu import train_slot_free
from scripts.run_arbitration_study import read, sha, write

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        not Path("runs/response_study/parent2_response/training_audit.json").exists(),
        reason="Completed response dependencies unavailable",
    ),
]


def test_fixed_protocol_and_capability_gate(tmp_path, monkeypatch):
    plan = study.make_plan()
    assert plan["evaluation"]["first_seed"] == 24000000
    assert plan["evaluation"]["pairs"] == 5000
    assert plan["updates"] == 520
    assert len(plan["cases"]) == 4 and len(plan["controls"]) == 4
    assert len(study.tasks(plan)) == 20
    for index in (1, 2):
        own = plan["cases"][f"parent{index}_learner"]["config"]
        mc = plan["cases"][f"parent{index}_lambda1"]["config"]
        assert own["seed"] == mc["seed"] == 19910001 + (index - 1) * 100000
        assert own["envs"] * own["horizon"] * plan["updates"] == 8519680
        assert own["gae_lambda"] == 0.95 and mc["gae_lambda"] == 1
        assert own["opponent_fraction"] == mc["opponent_fraction"] == 1
        assert own["published_opponent"] == mc["published_opponent"] == "wp_point"
    monkeypatch.setattr(study, "CAPABILITY", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        study.make_plan(require_capability=True)


def test_real_cpu_pipeline_and_artifact_tampering(tmp_path):
    base = tmp_path / "smoke"
    study.run(base, smoke=True)
    plan = study.checked(base)
    process, report = read(base / "process.json"), read(base / "analysis.json")
    assert process["status"] == "complete"
    assert process["analysis_sha256"] == sha(base / "analysis.json")
    assert len(process["jobs"]) == 10
    assert all(j["returncode"] == 0 for j in process["jobs"])
    assert report["replicated_learner_gain"] is None
    assert len(report["training"]) == 2 and len(report["outcomes"]) == 8
    assert sum(x["matches"] for x in report["outcomes"].values()) == 32
    assert sum(x["counters"]["decisions"] for x in report["training"].values()) == 256
    assert len(report["contrasts"]) == 12
    study.audit_schedule(plan, process)
    target = base / "parent1_learner/training_complete.json"
    original = target.read_bytes()
    changed = read(target)
    changed["counters"]["dropped_learner_tails"] += 1
    write(target, changed)
    with pytest.raises(AssertionError):
        study.audit_training(base, "parent1_learner", plan)
    target.write_bytes(original)
    target = base / "parent1_lambda1/rl-point-assisted.json"
    original = target.read_bytes()
    changed = read(target)
    changed["opponent"] = "rl-wp-assisted"
    write(target, changed)
    with pytest.raises(AssertionError):
        study.analyze(base)
    target.write_bytes(original)
    target = base / "protocol.json"
    original = target.read_bytes()
    changed = read(target)
    changed["updates"] += 1
    write(target, changed)
    with pytest.raises(ValueError, match="Protocol changed"):
        study.checked(base)
    target.write_bytes(original)
    assert study.checked(base) == plan
    with pytest.raises(ValueError, match="fresh"):
        study.run(base, smoke=True)


def test_prior_gpu_and_cpu_slots_are_separately_gated(tmp_path):
    plan = study.make_plan()
    plan["source_study"] = str(tmp_path)
    process = {
        "protocol_sha256": plan["source_protocol_sha256"],
        "status": "running",
        "jobs": [
            {"mode": "train", "case": c, "returncode": 0}
            for c in plan["source_training_cases"]
        ],
    }
    process["jobs"][-1].pop("returncode")
    assert not train_slot_free(process, plan["source_training_cases"])
    write(tmp_path / "process.json", process)
    assert not study.prior_ready(plan, "train")
    assert not study.prior_ready(plan, "evaluate")
    process["jobs"][-1]["returncode"] = 0
    assert train_slot_free(process, plan["source_training_cases"])
    write(tmp_path / "process.json", process)
    assert study.prior_ready(plan, "train")
    assert not study.prior_ready(plan, "evaluate")
    write(tmp_path / "analysis.json", {"status": "passed"})
    process.update(status="complete", analysis_sha256=sha(tmp_path / "analysis.json"))
    write(tmp_path / "process.json", process)
    assert study.prior_ready(plan, "evaluate")
    write(tmp_path / "analysis.json", {"status": "changed"})
    with pytest.raises(ValueError, match="analysis identity"):
        study.prior_ready(plan, "evaluate")
    process["status"] = "failed"
    with pytest.raises(RuntimeError, match="failed"):
        train_slot_free(process, plan["source_training_cases"])
    write(tmp_path / "process.json", process)
    with pytest.raises(RuntimeError, match="failed"):
        study.prior_ready(plan, "train")


def test_failure_retains_process_record_and_cleans_owned_workers(tmp_path, monkeypatch):
    real_plan = study.make_plan(smoke=True)
    monkeypatch.setattr(
        study, "make_plan", lambda *args, **kwargs: copy.deepcopy(real_plan)
    )
    calls = []

    def fail_launch(*args, **kwargs):
        raise OSError("test launch failure")

    def cleanup(active):
        calls.append(len(active))
        return []

    monkeypatch.setattr(study.subprocess, "Popen", fail_launch)
    monkeypatch.setattr(study, "stop_owned_workers", cleanup)
    base = tmp_path / "failed"
    with pytest.raises(OSError, match="test launch failure"):
        study.run(base, smoke=True)
    process = read(base / "process.json")
    assert process["status"] == "failed" and process["cleanup_failures"] == []
    assert calls == [0]
