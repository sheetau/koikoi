"""Reduced real scheduler/training/evaluation/audit round trip and tampering."""

import copy
from pathlib import Path

import pytest
import torch

from scripts import run_response_study as study
from scripts.run_arbitration_study import read, write
from scripts.run_response_study import analyze, audit_training, checked, make_plan, run


@pytest.mark.cuda
@pytest.mark.upstream
def test_actual_smoke_study_and_rejections(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt").exists():
        pytest.skip("Retained research parent unavailable")
    base = tmp_path / "study"
    run(base, smoke=True)
    p = checked(base)
    process = read(base / "process.json")
    report = read(base / "analysis.json")
    assert process["status"] == "complete"
    assert len(process["jobs"]) == 8
    assert all(job["returncode"] == 0 for job in process["jobs"])
    assert report["kind"] == "response_smoke"
    assert report["replicated_response_gain"] is None
    assert len(report["training"]) == 2 and len(report["outcomes"]) == 6
    assert sum(row["matches"] for row in report["outcomes"].values()) == 24
    for job in process["jobs"]:
        concurrent = [
            j for j in process["jobs"] if j["started"] <= job["started"] < j["ended"]
        ]
        assert sum(j["mode"] == "train" for j in concurrent) <= 1
        assert sum(j["mode"] == "evaluate" for j in concurrent) <= 3
        if job["mode"] == "evaluate" and job["case"] in p["cases"]:
            trainer = next(
                j
                for j in process["jobs"]
                if j["mode"] == "train" and j["case"] == job["case"]
            )
            assert trainer["ended"] <= job["started"]
    target = base / "parent1_response/training_complete.json"
    original = target.read_bytes()
    tampered = read(target)
    tampered["counters"]["matches"] += 1
    write(target, tampered)
    with pytest.raises(AssertionError):
        audit_training(base, "parent1_response", p)
    target.write_bytes(original)
    target = base / "parent1_response/rl-wp-assisted.json"
    original = target.read_bytes()
    tampered = read(target)
    tampered["opponent"] = "rl-point-assisted"
    write(target, tampered)
    with pytest.raises(AssertionError):
        analyze(base)
    target.write_bytes(original)
    target = base / "protocol.json"
    original = target.read_bytes()
    tampered = copy.deepcopy(p)
    tampered["updates"] += 1
    write(target, tampered)
    with pytest.raises(ValueError, match="Protocol changed"):
        checked(base)
    target.write_bytes(original)
    assert checked(base) == p


def test_fixed_protocol_has_matched_budgets_and_fresh_deals():
    p = make_plan()
    assert p["updates"] == 520
    assert p["evaluation"]["first_seed"] == 23000000
    assert p["evaluation"]["pairs"] == 5000
    assert len(p["parents"]) == 2 and len(p["cases"]) == 4
    for index in (1, 2):
        response = p["cases"][f"parent{index}_response"]["config"].copy()
        control = p["cases"][f"parent{index}_selfplay"]["config"]
        response.update(opponent_fraction=0.0, published_opponent="")
        assert response == control
        assert response["envs"] * response["horizon"] * p["updates"] == 8519680


def test_scheduler_failure_stops_only_owned_workers(tmp_path, monkeypatch):
    plan = make_plan(smoke=True)
    monkeypatch.setattr(study, "make_plan", lambda smoke: plan)
    monkeypatch.setattr(study, "provenance", lambda: plan["source"])
    children = []
    killed = []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 800000 + len(children)
            self.returncode = (
                1 if command[command.index("--mode") + 1] == "train" else None
            )
            children.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert self.returncode is not None
            return self.returncode

        def terminate(self):
            killed.append(self.pid)
            self.returncode = -1

    def kill(command, **kwargs):
        assert command[0] == "taskkill" and command[-2:] == ["/T", "/F"]
        pid = int(command[2])
        child = next(c for c in children if c.pid == pid)
        child.terminate()

    monkeypatch.setattr(study.subprocess, "Popen", Child)
    monkeypatch.setattr(study.subprocess, "run", kill)
    monkeypatch.setattr(study.time, "sleep", lambda seconds: None)
    base = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="Worker failed"):
        run(base, smoke=True)
    report = read(base / "process.json")
    assert report["status"] == "failed" and report["cleanup_failures"] == []
    assert len(children) == 3
    assert killed == [children[1].pid, children[2].pid]
    assert all(child.poll() is not None for child in children)
