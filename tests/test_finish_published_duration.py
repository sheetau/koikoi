"""Failure paths for the one-use duration completion queue."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import finish_published_duration as queue


def test_process_exit_requires_success_and_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="exited 7"):
        queue.wait_for_training(SimpleNamespace(exit_code=lambda: 7), tmp_path)
    with pytest.raises(FileNotFoundError):
        queue.wait_for_training(SimpleNamespace(exit_code=lambda: 0), tmp_path)
    queue.write(tmp_path / "training_complete.json", {"status": "complete"})
    queue.wait_for_training(SimpleNamespace(exit_code=lambda: 0), tmp_path)
    queue.write(tmp_path / "failure.json", {"error": "failure overrides completion"})
    with pytest.raises(RuntimeError, match="failure receipt"):
        queue.wait_for_training(SimpleNamespace(exit_code=lambda: 0), tmp_path)


def test_live_process_does_not_release_early(tmp_path, monkeypatch):
    queue.write(tmp_path / "training_complete.json", {"status": "complete"})
    states = iter([None, None, 0])
    sleeps = []
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)
    queue.wait_for_training(SimpleNamespace(exit_code=lambda: next(states)), tmp_path)
    assert sleeps == [30, 30]


def test_failed_job_preserves_log_and_refuses_overwrite(tmp_path):
    with pytest.raises(RuntimeError, match="exited"):
        queue.run_job("failure", ["no_such_duration_queue_module"], tmp_path)
    previous = (tmp_path / "failure.log").read_bytes()
    assert b"No module named" in previous
    with pytest.raises(FileExistsError):
        queue.run_job("failure", ["no_such_duration_queue_module"], tmp_path)
    assert (tmp_path / "failure.log").read_bytes() == previous
    with pytest.raises(FileExistsError):
        queue.finish(tmp_path, None)


def test_prepare_failure_never_starts_evaluation(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    queue.write(training / "running.json", {"pid": 123})
    queue.write(training / "protocol.json", {"inputs": {}})
    cap = tmp_path / "cap.json"
    queue.write(cap, {"status": "passed", "inputs": {}})
    monkeypatch.setattr(queue, "TRAINING", training)
    monkeypatch.setattr(queue, "OUTPUT", tmp_path / "evaluation")
    monkeypatch.setattr(queue, "CAPABILITY", cap)
    monkeypatch.setattr(queue, "sha", lambda path: "fixture")
    monkeypatch.setattr(queue, "wait_for_training", lambda *args: None)
    called, closed = [], []

    def fail(name, arguments, root):
        called.append(name)
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(queue, "run_job", fail)
    producer = SimpleNamespace(
        identity={"pid": 123},
        exit_code=lambda: None,
        close=lambda: closed.append(True),
    )
    root = tmp_path / "queue"
    with pytest.raises(RuntimeError, match="preparation failed"):
        queue.finish(root, producer)
    assert called == ["prepare"] and closed == [True]
    receipt = queue.read(root / "process.json")
    assert receipt["status"] == "failed" and receipt["completed"] == []


@pytest.mark.skipif(os.name != "nt", reason="Windows query handle")
def test_real_windows_handle_tracks_owned_child_after_exit():
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(7)"],
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    probe = None
    try:
        probe = queue.Producer(child.pid)
        assert probe.identity["pid"] == child.pid
        assert probe.identity["creation_filetime"] > 0
        assert probe.exit_code() is None
        child.communicate(timeout=10)
        assert probe.exit_code() == 7
        assert probe.exit_code() == 7
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if probe is not None:
            probe.close()
