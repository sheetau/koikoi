"""Prespecified selection, resource/dependency gates and failed-worker cleanup."""

import copy
import json

import pytest

import scripts.run_cold_ppo_screen as screen
from scripts.audit_cold_ppo_screen import (
    independent_decision,
    metrics,
    raw_pairs,
    validate_schedule,
)


@pytest.mark.parametrize(
    "warm,cold,paired,expected",
    [
        ((0.51, 0.51), (0.52, 0.52), (0.001, 0.001), "cold"),
        ((0.51, 0.51), (0.52, 0.52), (0, 0.001), "warm"),
        ((0.51, 0.51), (0.49, 0.52), (0.001, 0.001), "warm"),
        ((0.5, 0.51), (0.52, 0.52), (-0.001, 0.001), "cold"),
        ((0.49, 0.52), (0.5, 0.52), (0.001, 0.001), None),
        ((0.49, 0.49), (0.49, 0.49), (-0.01, -0.01), None),
    ],
)
@pytest.mark.parametrize("audit", [False, True])
def test_fixed_selection_requires_replication_and_uses_canonical_seed(
    warm, cold, paired, expected, audit
):
    endpoints = {
        f"seed{seed}_{arm}": {"win_ci95": [values[i], 0.6]}
        for arm, values in (("warm", warm), ("cold", cold))
        for i, seed in enumerate(screen.SEEDS)
    }
    differences = {
        str(seed): {"score_difference_ci95": [paired[i], 0.1]}
        for i, seed in enumerate(screen.SEEDS)
    }
    if audit:
        result = independent_decision(
            {k: {"score_ci95": v["win_ci95"]} for k, v in endpoints.items()},
            {
                k: {"score_ci95": v["score_difference_ci95"]}
                for k, v in differences.items()
            },
        )
    else:
        result = screen.decision(endpoints, differences)
    assert result["selected_arm"] == expected
    assert result["canonical_case"] == (
        None if expected is None else "seed6710001_" + expected
    )
    assert result["cold_paired_benefit"] == all(x > 0 for x in paired)


def record(resource, status):
    return {"resource": resource, "status": status}


def test_scheduler_initially_starts_only_baseline_and_one_gpu():
    ready = screen.ready_jobs(screen.job_plan(["a", "b"]), {}, {"gpu": 1, "cpu": 2})
    assert [r["id"] for r in ready] == ["evaluate:initial", "train:a"]


def test_evaluation_requires_completed_initial_and_own_training():
    jobs = screen.job_plan(["a", "b"])
    records = {
        "evaluate:initial": record("cpu", "running"),
        "train:a": record("gpu", "complete"),
    }
    assert [
        j["id"] for j in screen.ready_jobs(jobs, records, {"gpu": 1, "cpu": 2})
    ] == ["train:b"]
    records["evaluate:initial"]["status"] = "complete"
    assert [
        j["id"] for j in screen.ready_jobs(jobs, records, {"gpu": 1, "cpu": 2})
    ] == ["train:b", "evaluate:a"]


def test_cpu_capacity_is_reserved_across_a_launch_batch():
    jobs = screen.job_plan(["a", "b", "c"])
    records = {"evaluate:initial": record("cpu", "complete")} | {
        "train:" + name: record("gpu", "complete") for name in ("a", "b", "c")
    }
    assert [
        j["id"] for j in screen.ready_jobs(jobs, records, {"gpu": 1, "cpu": 2})
    ] == ["evaluate:a", "evaluate:b"]


def test_failure_cancels_only_owned_live_workers_and_records_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(screen, "BASE", tmp_path)
    for name in ("initial", "a", "b"):
        (tmp_path / name).mkdir()
    processes = []
    stopped = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(processes)
            self.code = 1 if command[-1] == "evaluate:initial" else None
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
        assert (
            command[0] == "taskkill"
            and command[1] == "/PID"
            and command[3:] == ["/T", "/F"]
        )
        process = next(p for p in processes if p.pid == int(command[2]))
        process.terminate()

    monkeypatch.setattr(screen.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(screen.subprocess, "run", stop_tree)
    with pytest.raises(RuntimeError, match="evaluate:initial failed"):
        screen.supervise(
            {"cases": {"a": {}, "b": {}}, "parallelism": {"gpu": 1, "cpu": 2}}
        )
    manifest = json.loads((tmp_path / "process.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and stopped == [101]
    assert manifest["jobs"]["evaluate:initial"]["status"] == "failed"
    assert manifest["jobs"]["train:a"]["status"] == "aborted"
    assert "train:b" not in manifest["jobs"]


def test_initial_gate_rejects_deal_range_and_asymmetric_pairs():
    rows = [
        {
            "seed": screen.FIRST + i,
            "seat0_score": 1,
            "seat1_score": 0,
            "seat0_point_difference": 2,
            "seat1_point_difference": -2,
        }
        for i in range(screen.PAIRS)
    ]
    report = {
        "pairs": screen.PAIRS,
        "records": rows,
        "draw_adjusted_win_rate": 0.5,
        "mean_point_difference": 0,
    }
    screen.initial_gate(report)
    rows[0]["seed"] -= 1
    with pytest.raises(AssertionError):
        screen.initial_gate(report)
    rows[0]["seed"] += 1
    rows[1]["seat1_point_difference"] = 0
    with pytest.raises(AssertionError):
        screen.initial_gate(report)


@pytest.mark.parametrize(
    "fault", [None, "duplicate", "missing", "wrong_score", "odd_points", "nonfinite"]
)
def test_independent_raw_pair_fixture(fault):
    rows = [
        {
            "seed": 20,
            "seat0_score": 1,
            "seat1_score": 0,
            "seat0_point_difference": 2,
            "seat1_point_difference": -2,
        },
        {
            "seed": 21,
            "seat0_score": 1,
            "seat1_score": 0.5,
            "seat0_point_difference": 2,
            "seat1_point_difference": 0,
        },
    ]
    report = {"pairs": 2, "records": rows}
    if fault is None:
        scores, points = raw_pairs(report, first=20, count=2)
        assert scores.tolist() == [0.5, 0.75] and points.tolist() == [0, 1]
        return
    if fault == "duplicate":
        rows[1]["seed"] = 20
    elif fault == "missing":
        rows.pop()
    elif fault == "wrong_score":
        rows[1]["seat1_score"] = 1
    elif fault == "odd_points":
        rows[0]["seat0_point_difference"] = 3
    else:
        rows[0]["seat0_point_difference"] = float("nan")
    with pytest.raises(AssertionError):
        raw_pairs(report, first=20, count=2)


def test_live_metric_tail_is_not_mistaken_for_a_completed_update(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"updates":1}\n{"updates":2')
    assert metrics(path) == [{"updates": 1}]
    with pytest.raises(AssertionError):
        metrics(path, final=True)


def schedule_fixture():
    times = {
        "evaluate:initial": (0, 1),
        "train:a": (0, 2),
        "train:b": (2, 4),
        "train:c": (4, 6),
        "evaluate:a": (2, 8),
        "evaluate:b": (4, 9),
        "evaluate:c": (8, 11),
    }
    jobs = {}
    for job in screen.job_plan(["a", "b", "c"]):
        start, end = times[job["id"]]
        jobs[job["id"]] = job | {
            "started": start,
            "finished": end,
            "status": "complete",
            "returncode": 0,
        }
    return jobs


def test_independent_schedule_fixture_and_invalid_overlaps():
    jobs = schedule_fixture()
    validate_schedule(jobs, ["a", "b", "c"], {"cpu": 2, "gpu": 1})
    for job, field, value in [
        ("train:b", "started", 1),
        ("evaluate:c", "started", 6),
        ("evaluate:a", "started", 1),
        ("train:a", "returncode", 1),
    ]:
        bad = copy.deepcopy(jobs)
        bad[job][field] = value
        with pytest.raises(AssertionError):
            validate_schedule(bad, ["a", "b", "c"], {"cpu": 2, "gpu": 1})
