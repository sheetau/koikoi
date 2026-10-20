import copy
import json
import os
import subprocess
import sys

import pytest

from koikoi.upstream import UPSTREAM_PATH
from scripts.audit_point_initialization_evaluation import replay
from scripts.evaluate_score_auxiliary import evaluate_cell, prepare, run_cells
from scripts.run_arbitration_study import read, sha


def test_missing_training_completion_cannot_create_main_output(tmp_path, monkeypatch):
    import scripts.evaluate_score_auxiliary as module

    def incomplete(path):
        if path.name == "protocol.json":
            return {"inputs": {}}
        raise FileNotFoundError("Training is still active")

    monkeypatch.setattr(module, "read", incomplete)
    monkeypatch.setattr(module, "sha", lambda _: module.TRAINING_SHA)
    with pytest.raises(FileNotFoundError, match="still active"):
        prepare(tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


@pytest.mark.upstream
@pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched")
def test_real_auxiliary_endpoint_cell_and_native_recorded_action_replay(
    tmp_path, record_property
):
    checkpoint = "runs/score_auxiliary_training_capability/score/cycle2/ppo.pt"
    cell = {
        "parent": 1,
        "policy": "score",
        "opponent": "point",
        "checkpoint": checkpoint,
        "first_seed": 110030001,
    }
    protocol = {
        "inputs": {checkpoint: sha(checkpoint)},
        "pairs": 1,
        "cells": {"fixture": cell},
    }
    path = tmp_path / "evaluation_protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    evaluate_cell(path, "fixture")
    report = read(tmp_path / "evaluation/fixture.json")
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation/fixture.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["seed"], row["seat"]) for row in rows] == [
        (110030001, 0),
        (110030001, 1),
    ]
    total = {key: 0 for key in report["work"]}
    for row in rows:
        for key, count in replay(row).items():
            total[key] += count
    assert total == report["work"]
    assert report["checkpoint_sha256"] == sha(checkpoint)
    assert report["protocol_sha256"] == sha(path)
    for field in ("points", "work"):
        corrupt = copy.deepcopy(rows[0])
        if field == "points":
            corrupt[field] += 2
        else:
            corrupt[field]["round_endings"] += 1
        with pytest.raises(AssertionError):
            replay(corrupt)
    with pytest.raises(ValueError, match="Preserve"):
        evaluate_cell(path, "fixture")
    record_property("producer_work", json.dumps(total))
    record_property("unique_deals", 1)


def test_failed_owned_cell_stops_other_worker_and_refuses_restart(
    tmp_path, monkeypatch
):
    import scripts.evaluate_score_auxiliary as module

    path = tmp_path / "evaluation_protocol.json"
    path.write_text(
        json.dumps({"inputs": {}, "cells": {"bad": {}, "pending": {}}}),
        encoding="utf-8",
    )
    original = subprocess.Popen
    spawned = []

    def launch(command, **kwargs):
        if command[0] == "taskkill":
            return original(command, **kwargs)
        assert "scripts.evaluate_score_auxiliary" in command
        code = (
            "import time; time.sleep(0.5); raise SystemExit(7)"
            if not spawned
            else "import time; time.sleep(120)"
        )
        worker = original([sys.executable, "-c", code], **kwargs)
        spawned.append(worker)
        return worker

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    try:
        with pytest.raises(RuntimeError, match="exited 7"):
            run_cells(path)
        assert len(spawned) == 2 and all(
            worker.poll() is not None for worker in spawned
        )
        assert (tmp_path / "evaluation_failed.json").exists()
        assert not (tmp_path / "evaluation_complete.json").exists()
        with pytest.raises(FileExistsError):
            run_cells(path)
        assert len(spawned) == 2
    finally:
        for worker in spawned:
            if worker.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(worker.pid), "/T", "/F"], check=False
                    )
                else:
                    worker.kill()
                worker.wait(timeout=10)
