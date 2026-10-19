"""Wait for the exact Windows learner process, then evaluate and audit once."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write

TRAINING = Path("runs/published_duration_study")
OUTPUT = Path("runs/published_duration_evaluation")
CAPABILITY = Path("runs/published_duration_evaluation_capability/audit.json")


class Producer:
    """Hold a query-only process handle; PID reuse cannot change its identity."""

    def __init__(self, pid):
        if os.name != "nt":
            raise RuntimeError("This queue monitors a Windows producer")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME)
        ] * 4
        self.api.GetProcessTimes.restype = wintypes.BOOL
        self.api.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.api.GetExitCodeProcess.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.OpenProcess(0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        self.identity = {
            "pid": pid,
            "creation_filetime": (times[0].dwHighDateTime << 32)
            | times[0].dwLowDateTime,
        }

    def exit_code(self):
        code = wintypes.DWORD()
        if not self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == 259 else code.value

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def checked(inputs):
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def wait_for_training(producer, root, poll=30):
    while True:
        if (root / "failure.json").exists():
            raise RuntimeError("Training failure receipt exists; preserve it")
        code = producer.exit_code()
        if code is not None:
            if code != 0:
                raise RuntimeError(f"Exact training process exited {code}")
            receipt = read(root / "training_complete.json")
            if receipt["status"] != "complete":
                raise RuntimeError("Training process exited without completion")
            return
        time.sleep(poll)


def jobs():
    return [
        (
            "prepare",
            ["scripts.evaluate_published_duration", "prepare", "--output", str(OUTPUT)],
        ),
        (
            "evaluate",
            [
                "scripts.evaluate_published_duration",
                "run",
                "--protocol",
                str(OUTPUT / "evaluation_protocol.json"),
            ],
        ),
        (
            "audit",
            ["scripts.audit_published_duration_evaluation", "--input", str(OUTPUT)],
        ),
    ]


def run_job(name, arguments, root):
    with (root / f"{name}.log").open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        write(root / "child.json", {"job": name, "pid": process.pid})
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} exited {code}; preserve outputs and log")


def finish(root, producer):
    # Atomic directory creation is the single-use claim, including failed runs.
    root.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        if OUTPUT.exists():
            raise ValueError("Main evaluation already exists")
        if producer.exit_code() is not None:
            raise ValueError("Attach only to the verified live training process")
        if read(TRAINING / "running.json")["pid"] != producer.identity["pid"]:
            raise ValueError("Producer PID differs from training receipt")
        cap = read(CAPABILITY)
        if cap["status"] != "passed":
            raise ValueError("Evaluation capability did not pass")
        inputs = {**read(TRAINING / "protocol.json")["inputs"], **cap["inputs"]}
        for path in (
            Path(__file__),
            Path(sys.executable),
            TRAINING / "protocol.json",
            CAPABILITY,
            Path("tests/test_finish_published_duration.py"),
            Path("runs/published_duration_queue_tests.xml"),
        ):
            inputs[str(path)] = sha(path)
        checked(inputs)
        write(
            root / "protocol.json",
            {
                "producer": producer.identity,
                "queue_pid": os.getpid(),
                "inputs": inputs,
                "commands": jobs(),
                "cwd": str(Path.cwd()),
                "scope": "Wait for exact producer exit 0 and completion, then fixed evaluation and audit once. No training restart or model promotion.",
            },
        )
        write(root / "process.json", {"status": "waiting_training", "pid": os.getpid()})
        wait_for_training(producer, TRAINING)
        for name, arguments in jobs():
            checked(inputs)
            write(root / "process.json", {"status": name, "completed": completed})
            run_job(name, arguments, root)
            completed.append(name)
        checked(inputs)
        audit = read(OUTPUT / "audit.json")
        if audit["status"] != "passed":
            raise ValueError("Evaluation audit did not pass")
        checked(audit["inputs"])
        write(
            root / "process.json",
            {
                "status": "complete",
                "completed": completed,
                "audit_sha256": sha(OUTPUT / "audit.json"),
                "qualified": audit["qualified"],
                "deployment_changed": False,
            },
        )
    except BaseException as exc:
        write(
            root / "process.json",
            {
                "status": "failed",
                "completed": completed,
                "error": repr(exc),
                "instruction": "Preserve all outputs. Inspect any child process before recovery; do not relaunch this queue.",
            },
        )
        raise
    finally:
        producer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer = Producer(args.producer_pid)
    print(json.dumps(producer.identity), flush=True)
    finish(args.output, producer)


if __name__ == "__main__":
    main()
