"""Wait for the existing training process, then audit and evaluate once."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.train_branch_policy_gradient import sha, write


def run(training_pid):
    base = Path("runs/branch_policy_gradient_training")
    queue = Path("runs/branch_policy_gradient_completion")
    if training_pid <= 0 or queue.exists():
        raise ValueError("Expected a live training PID and a fresh completion queue")
    queue.mkdir(parents=True)
    files = (
        "scripts/finish_branch_policy_study.py",
        "scripts/evaluate_branch_policy_gradient.py",
        "scripts/audit_branch_policy_study.py",
        "scripts/branch_policy_gradient.py",
        "scripts/train_branch_policy_gradient.py",
    )
    plan = {
        "training_pid": training_pid,
        "training_protocol_sha256": sha(base / "protocol.json"),
        "bindings": {name: sha(name) for name in files},
        "steps": [
            ["scripts.audit_branch_policy_study", "training"],
            ["scripts.evaluate_branch_policy_gradient"],
            ["scripts.audit_branch_policy_study", "evaluation"],
        ],
    }
    write(queue / "protocol.json", plan)
    write(queue / "process.json", {"pid": os.getpid(), "status": "waiting"})
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        while not (base / "complete.json").exists():
            alive = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        f"if (Get-Process -Id {training_pid} -ErrorAction SilentlyContinue) "
                        "{exit 0} else {exit 1}"
                    ),
                ],
                capture_output=True,
                check=False,
                creationflags=flags,
            )
            if alive.returncode:
                raise RuntimeError(
                    "Training process exited without its completion receipt"
                )
            time.sleep(15)
        for module, *arguments in plan["steps"]:
            if sha(base / "protocol.json") != plan["training_protocol_sha256"]:
                raise RuntimeError("Training protocol changed")
            if any(sha(name) != digest for name, digest in plan["bindings"].items()):
                raise RuntimeError("Queued implementation changed")
            stage = "_".join([module.rsplit(".", 1)[-1], *arguments])
            write(
                queue / "process.json",
                {
                    "pid": os.getpid(),
                    "status": "running",
                    "stage": stage,
                },
            )
            with (queue / f"{stage}.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as log:
                subprocess.run(
                    [sys.executable, "-m", module, *arguments],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    creationflags=flags,
                )
            print(json.dumps({"completed": stage}), flush=True)
        write(
            queue / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(queue / "protocol.json"),
                "training_audit_sha256": sha(base / "audit.json"),
                "evaluation_sha256": sha(base / "evaluation" / "summary.json"),
                "evaluation_audit_sha256": sha(base / "evaluation" / "audit.json"),
            },
        )
        write(queue / "process.json", {"pid": os.getpid(), "status": "complete"})
    except BaseException as error:
        write(
            queue / "process.json",
            {
                "pid": os.getpid(),
                "status": "failed",
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    run(parser.parse_args().training_pid)
