"""Real process pipeline, immutable controls and artifact-failure checks."""

import copy

import pytest

from scripts import run_observer_selfplay_study as study
from scripts.run_arbitration_study import read, sha, write

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        not study.CAPABILITY.exists(), reason="Observer capability unavailable"
    ),
]


def test_frozen_comparison_and_capability_required(tmp_path, monkeypatch):
    p = study.make_plan()
    assert p["evaluation"]["first_seed"] == 27000000
    assert p["evaluation"]["pairs"] == 5000
    assert len(p["cases"]) == len(p["controls"]) == 4
    assert len(study.tasks(p)) == 20
    for index in (1, 2):
        left = p["cases"][f"parent{index}_observer"]["config"].copy()
        right = p["cases"][f"parent{index}_signed"]["config"].copy()
        assert left.pop("policy_returns") == "observer"
        assert right.pop("policy_returns") == "signed"
        assert left == right
        assert left["seed"] == 19910001 + (index - 1) * 100000
        assert p["updates"] * left["envs"] * left["horizon"] == 8519680
        assert left["opponent_fraction"] == 0
    monkeypatch.setattr(study, "CAPABILITY", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        study.make_plan()


def test_real_pipeline_and_tampering(tmp_path):
    base = tmp_path / "pipeline"
    study.run(base, smoke=True)
    p = study.checked(base)
    process, report = read(base / "process.json"), read(base / "analysis.json")
    assert process["status"] == "complete"
    assert process["analysis_sha256"] == sha(base / "analysis.json")
    assert len(process["jobs"]) == 10 and all(
        job["returncode"] == 0 for job in process["jobs"]
    )
    assert report["replicated_observer_gain"] is None
    assert len(report["outcomes"]) == 8 and len(report["contrasts"]) == 12
    assert sum(row["matches"] for row in report["outcomes"].values()) == 32
    assert (
        sum(row["counters"]["decisions"] for row in report["training"].values()) == 256
    )
    assert report["matched_warmup"]["parent1"]["status"] == "passed"
    target = base / "parent1_observer/training_complete.json"
    original = target.read_bytes()
    changed = read(target)
    changed["counters"]["value_rows"] += 1
    write(target, changed)
    with pytest.raises(AssertionError):
        study.audit_training(base, "parent1_observer", p)
    target.write_bytes(original)
    target = base / "parent1_signed/rl-point-assisted.json"
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
    # Retain a compact locator for recording this actual subprocess run.
    print("PIPELINE_ARTIFACT=" + str(base), flush=True)


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
