"""Guard unfinished GPU work and execute the real reduced capability path."""

import copy

import pytest

from scripts.check_seat_observer_device import CASES, read, run, training_ready


def state():
    return {
        "status": "running",
        "jobs": [
            {"mode": "train", "case": case, "returncode": 0, "started": 1, "ended": 2}
            for case in sorted(CASES)
        ],
    }


def test_gpu_readiness_requires_all_terminal_successful_workers():
    original = state()
    assert training_ready(original)
    partial = copy.deepcopy(original)
    partial["jobs"].pop()
    assert not training_ready(partial)
    live = copy.deepcopy(original)
    del live["jobs"][0]["returncode"]
    assert not training_ready(live)
    broken = copy.deepcopy(original)
    broken["jobs"][0]["returncode"] = 1
    with pytest.raises(ValueError, match="failed"):
        training_ready(broken)
    duplicate = copy.deepcopy(original)
    duplicate["jobs"].append(duplicate["jobs"][0])
    with pytest.raises(ValueError, match="duplicate"):
        training_ready(duplicate)


def test_live_guard_precedes_output_and_model_creation(monkeypatch, tmp_path):
    from scripts import check_seat_observer_device as module

    real_read = module.read

    def unfinished(path):
        if str(path).endswith("process.json"):
            return {
                "status": "running",
                "jobs": [],
                "protocol_sha256": module.sha(module.STUDY / "protocol.json"),
            }
        return real_read(path)

    monkeypatch.setattr(module, "read", unfinished)
    monkeypatch.setattr(
        module.SeatObserverTrainer,
        "from_parent",
        lambda *args: pytest.fail("GPU model created before training guard"),
    )
    target = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="unfinished GPU"):
        run(target)
    assert not target.exists()


def test_actual_reduced_capability(tmp_path):
    target = tmp_path / "capability"
    run(target, smoke=True)
    report = read(target / "audit.json")
    assert report["status"] == "passed" and report["kind"] == "cpu_smoke"
    assert report["total_physical_work"]["decisions"] == 2048
    assert report["total_physical_work"]["value_rows"] == 4096
    assert (
        report["initial_signatures"]["public"] == report["initial_signatures"]["masked"]
    )
    for context, row in report["conditions"].items():
        assert row["counters"]["updates"] == 3
        assert row["replay_work"]["decisions"] == 256
        assert len(row["optimizer_steps"]) == 96
        if context == "public":
            assert row["seat_column_l1"] > 0
        else:
            assert row["seat_column_l1"] == 0
    assert (
        report["warmup_signatures"]["public"]["value"]
        != report["warmup_signatures"]["masked"]["value"]
    )
    with pytest.raises(ValueError, match="fresh"):
        run(target, smoke=True)
    print(
        {
            "report": str(target / "audit.json"),
            "total_physical_work": report["total_physical_work"],
        }
    )
