import json
import subprocess
import sys

import pytest

from scripts import recover_published_duration as recovery
from scripts.recover_published_duration import compare_prefix, rows_info, write_new


def row(seed):
    return {
        "seed": seed,
        "actions": [1, 2],
        "work": {"decisions": 2, "round_endings": 1, "full_matches": 1},
    }


def test_saved_prefix_order_partial_tail_and_exact_repetition(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    prefix = (json.dumps(row(1)) + "\n").encode()
    old.write_bytes(prefix + b'{"seed":')
    info = rows_info(old, [{"seed": 1}, {"seed": 2}])
    assert info == {
        "complete_rows": 1,
        "prefix_bytes": len(prefix),
        "trailing_bytes": 8,
        "work": {"decisions": 2, "round_endings": 1, "full_matches": 1},
    }
    new.write_bytes(prefix + (json.dumps(row(2)) + "\n").encode())
    compare_prefix(old, new, info)
    new.write_bytes(prefix.replace(b"1, 2", b"1, 3"))
    with pytest.raises(ValueError, match="Repeated neural inference"):
        compare_prefix(old, new, info)
    with pytest.raises(ValueError, match="order or identity"):
        rows_info(old, [{"seed": 2}])
    old.write_bytes(prefix + prefix)
    with pytest.raises(ValueError, match="Too many"):
        rows_info(old, [{"seed": 1}])


def test_complete_line_corruption_and_new_receipt_refusal(tmp_path):
    path = tmp_path / "rows"
    path.write_bytes(b"{broken}\n")
    with pytest.raises(json.JSONDecodeError):
        rows_info(path, [{"seed": 1}])
    bad = row(1)
    bad["work"]["decisions"] = 3
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="work differs"):
        rows_info(path, [{"seed": 1}])
    receipt = tmp_path / "receipt"
    write_new(receipt, {"status": "failed"})
    original = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(receipt, {"status": "complete"})
    assert receipt.read_bytes() == original


def test_failed_worker_stops_other_owned_worker(tmp_path, monkeypatch):
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        if "scripts.evaluate_published_duration" not in command:
            return original(command, **kwargs)
        code = (
            "raise SystemExit(7)"
            if command[-1] == "fail"
            else "import time; time.sleep(30)"
        )
        process = original([sys.executable, "-c", code], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="exited 7"):
        recovery.run_workers(tmp_path, ["fail", "wait"])
    assert len(children) == 2 and all(p.poll() is not None for p in children)
    assert (tmp_path / "evaluation_fail.log").exists()
    assert (tmp_path / "evaluation_wait.log").exists()


def test_live_original_process_refuses_recovery(monkeypatch):
    class Live:
        closed = False

        def __init__(self, pid):
            pass

        def exit_code(self):
            return None

        def close(self):
            Live.closed = True

    monkeypatch.setattr(recovery, "Producer", Live)
    with pytest.raises(RuntimeError, match="still live"):
        recovery.require_old_processes_stopped()
    assert Live.closed
