import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.audit_point_initialization_evaluation import replay
from scripts.evaluate_point_initialization import cases, play_match, prepare


class FirstLegal:
    def act(self, env):
        return int(np.flatnonzero(env.legal())[0])


def test_seat_deal_schedule_and_independent_native_replay():
    schedule = cases(2, "point", 2, 103020001)
    assert [(r["seed"], r["seat"]) for r in schedule] == [
        (103020001, 0),
        (103020001, 1),
        (103020002, 0),
        (103020002, 1),
    ]
    agent = FirstLegal()
    row = play_match(agent, agent, schedule[0])
    assert replay(row) == row["work"]
    for field in ("points", "endpoint", "work", "actions"):
        bad = copy.deepcopy(row)
        if field == "points":
            bad[field] += 1
        elif field == "endpoint":
            bad[field][0] += 1
        elif field == "work":
            bad[field]["round_endings"] += 1
        else:
            bad[field][0] = 50
        with pytest.raises(AssertionError):
            replay(bad)


def test_training_completion_is_required_before_main_plan(tmp_path, monkeypatch):
    import scripts.evaluate_point_initialization as module

    def incomplete(path):
        if path.name == "protocol.json":
            return {"inputs": {}}
        raise FileNotFoundError("Training is still active")

    monkeypatch.setattr(module, "read", incomplete)
    monkeypatch.setattr(module, "sha", lambda _: module.TRAINING_SHA)
    with pytest.raises(FileNotFoundError, match="still active"):
        prepare(tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


def test_failed_cell_stops_owned_workers_and_refuses_restart(tmp_path, monkeypatch):
    import scripts.evaluate_point_initialization as module

    path = tmp_path / "evaluation_protocol.json"
    module.write_new(path, {"inputs": {}, "cells": {"bad": {}, "pending": {}}})
    original = subprocess.Popen
    spawned = []

    def launch(command, **kwargs):
        if command[0] == "taskkill":
            return original(command, **kwargs)
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
            module.run_cells(path)
        assert len(spawned) == 2 and all(p.poll() is not None for p in spawned)
        assert (tmp_path / "evaluation_failed.json").exists()
        assert not (tmp_path / "evaluation_complete.json").exists()
        with pytest.raises(FileExistsError):
            module.run_cells(path)
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
