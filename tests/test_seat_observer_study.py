"""Frozen seat treatment, resource gate and actual subprocess pipeline."""

import copy

import pytest

from scripts import run_seat_observer_study as study
from scripts.run_arbitration_study import read, sha, write

pytestmark = pytest.mark.upstream


def test_matched_smoke_plan_and_capability_required(tmp_path, monkeypatch):
    p = study.make_plan(smoke=True)
    assert p["kind"] == "seat_observer_smoke"
    assert len(p["cases"]) == len(p["controls"]) == 2
    assert len(study.tasks(p)) == 10
    assert p["controls"]["parent1_observer"]["format"] == "observer_selfplay_v1"
    left = p["cases"]["parent1_public"]["config"].copy()
    right = p["cases"]["parent1_masked"]["config"].copy()
    assert left.pop("seat_context") == "public"
    assert right.pop("seat_context") == "masked"
    assert left == right and left["policy_returns"] == "observer"
    assert p["updates"] * left["envs"] * left["horizon"] == 128
    assert p["evaluation"]["pairs"] == 2
    assert p["evaluation"]["first_seed"] == 26097000
    assert p["resources"] == {"trainers": 1, "evaluators": 1}
    monkeypatch.setattr(study, "CPU_CAPABILITY", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        study.make_plan(smoke=True)


def test_pending_prior_blocks_full_launch_and_all_evaluations(tmp_path, monkeypatch):
    source = tmp_path / "prior"
    write(source / "process.json", {"status": "running", "protocol_sha256": "fixed"})
    monkeypatch.setattr(study, "SOURCE", source)
    monkeypatch.setattr(study, "original_checked", lambda base: {})
    base = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="must complete"):
        study.run(base)
    assert not base.exists()
    p = {
        "kind": "seat_observer_smoke",
        "source_study": str(source),
        "source_protocol_sha256": "fixed",
    }
    assert study.prior_ready(p, "train")
    assert not study.prior_ready(p, "evaluate")
    p["kind"] = "seat_observer_development"
    assert not study.prior_ready(p, "train")
    write(source / "process.json", {"status": "failed", "protocol_sha256": "fixed"})
    with pytest.raises(RuntimeError, match="Prior study failed"):
        study.prior_ready(p, "evaluate")
    write(source / "process.json", {"status": "running", "protocol_sha256": "other"})
    with pytest.raises(ValueError, match="Prior protocol changed"):
        study.prior_ready(p, "evaluate")


def test_failed_launch_keeps_record_and_cleans_owned_workers(tmp_path, monkeypatch):
    plan = study.make_plan(smoke=True)
    monkeypatch.setattr(study, "make_plan", lambda *args, **kwargs: copy.deepcopy(plan))
    calls = []

    def fail(*args, **kwargs):
        raise OSError("deliberate launch failure")

    monkeypatch.setattr(study.subprocess, "Popen", fail)
    monkeypatch.setattr(
        study, "stop_owned_workers", lambda active: calls.append(len(active)) or []
    )
    base = tmp_path / "failed"
    with pytest.raises(OSError, match="deliberate"):
        study.run(base, smoke=True)
    assert calls == [0]
    assert read(base / "process.json")["status"] == "failed"


def test_real_pipeline_and_tampering(tmp_path):
    # This deliberately waits for the existing study's evaluation slots.
    # An observation timeout is not permission to restart this test.
    base = tmp_path / "pipeline"
    print("PIPELINE_ARTIFACT=" + str(base), flush=True)
    study.run(base, smoke=True)
    p = study.checked(base)
    process, report = read(base / "process.json"), read(base / "analysis.json")
    assert process["status"] == "complete"
    assert process["analysis_sha256"] == sha(base / "analysis.json")
    assert len(process["jobs"]) == 10 and all(
        job["returncode"] == 0 for job in process["jobs"]
    )
    assert report["replicated_seat_gain"] is None
    assert len(report["outcomes"]) == 8 and len(report["contrasts"]) == 12
    assert sum(row["matches"] for row in report["outcomes"].values()) == 32
    assert (
        sum(row["counters"]["decisions"] for row in report["training"].values()) == 256
    )
    assert report["matched_warmup"]["parent1"]["status"] == "passed"
    target = base / "parent1_public/training_complete.json"
    original = target.read_bytes()
    changed = read(target)
    changed["counters"]["value_rows"] += 1
    write(target, changed)
    with pytest.raises(AssertionError):
        study.audit_training(base, "parent1_public", p)
    target.write_bytes(original)
    target = base / "parent1_masked/rl-point-assisted.json"
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
    assert study.checked(base) == p
    with pytest.raises(ValueError, match="fresh"):
        study.run(base, smoke=True)
