"""Hold the exact learner handle, then audit training and evaluate once."""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.evaluate_score_auxiliary import (
    TRAINING,
    TRAINING_SHA,
    checked,
    read,
    sha,
    write_new,
)
from scripts.finish_point_initialization import run_job
from scripts.finish_published_duration import Producer

OUTPUT = Path("runs/score_auxiliary_evaluation")
CAPABILITY = Path("runs/score_auxiliary_evaluation_capability")


def wait_for_training(producer, root, poll=30):
    while True:
        if (root / "failed.json").exists():
            raise RuntimeError("Training failure receipt exists")
        code = producer.exit_code()
        if code is not None:
            if code != 0:
                raise RuntimeError(f"Exact training process exited {code}")
            receipt = read(root / "complete.json")
            if (
                receipt["status"] != "complete"
                or receipt["protocol_sha256"] != TRAINING_SHA
            ):
                raise RuntimeError("Training exited without matching completion")
            return
        time.sleep(poll)


def jobs():
    return [
        (
            "training_audit",
            ["scripts.audit_score_auxiliary_training", "--root", str(TRAINING)],
        ),
        (
            "prepare",
            ["scripts.evaluate_score_auxiliary", "prepare", "--output", str(OUTPUT)],
        ),
        (
            "evaluate",
            [
                "scripts.evaluate_score_auxiliary",
                "run",
                "--protocol",
                str(OUTPUT / "evaluation_protocol.json"),
            ],
        ),
        (
            "evaluation_audit",
            ["scripts.audit_score_auxiliary_evaluation", "--input", str(OUTPUT)],
        ),
    ]


def finish(root, producer):
    root.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        if OUTPUT.exists() or (TRAINING / "training_audit.json").exists():
            raise ValueError("Preserve existing main audit/evaluation")
        if producer.exit_code() is not None:
            raise ValueError("Attach only to the verified live training process")
        if read(TRAINING / "started.json")["pid"] != producer.identity["pid"]:
            raise ValueError("Held process differs from training receipt")
        assert sha(TRAINING / "protocol.json") == TRAINING_SHA
        capability = read(CAPABILITY / "audit.json")
        assert capability["status"] == "passed" and not capability["qualified"]
        assert capability["protocol_sha256"] == sha(
            CAPABILITY / "evaluation_protocol.json"
        )
        assert read(CAPABILITY / "evaluation_protocol.json")["mode"] == "capability"
        inputs = read(TRAINING / "protocol.json")["inputs"] | capability["inputs"]
        for path in (
            Path(__file__),
            Path(sys.executable),
            Path("scripts/finish_point_initialization.py"),
            Path("scripts/finish_published_duration.py"),
            Path("scripts/audit_score_auxiliary_training.py"),
            Path("scripts/audit_score_auxiliary_collection.py"),
            Path("tests/test_audit_score_auxiliary_training.py"),
            Path("tests/test_finish_score_auxiliary.py"),
            Path("runs/score_auxiliary_training_auditor_tests.xml"),
            Path("runs/score_auxiliary_queue_tests.xml"),
            TRAINING / "protocol.json",
            TRAINING / "started.json",
            CAPABILITY / "audit.json",
        ):
            inputs[str(path)] = sha(path)
        for name, count in (
            ("score_auxiliary_training_auditor_tests.xml", 10),
            ("score_auxiliary_queue_tests.xml", 5),
        ):
            suite = ET.parse(Path("runs") / name).getroot().find("testsuite")
            assert int(suite.attrib["tests"]) == count
            assert all(
                int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped")
            )
        checked(inputs)
        write_new(
            root / "protocol.json",
            {
                "producer": producer.identity,
                "queue_pid": os.getpid(),
                "inputs": inputs,
                "commands": jobs(),
                "cwd": str(Path.cwd()),
                "scope": "Exact producer exit0 and matching completion; training audit, fixed evaluation and native/statistical audit once. No restart or promotion. Mechanism diagnostics are separate.",
            },
        )
        print({"status": "waiting", "producer": producer.identity}, flush=True)
        wait_for_training(producer, TRAINING)
        checked(inputs)
        for name, arguments in jobs():
            print({"status": "starting", "stage": name}, flush=True)
            run_job(name, arguments, root)
            completed.append(name)
            checked(inputs)
        audit = read(OUTPUT / "audit.json")
        assert audit["status"] == "passed"
        checked(audit["inputs"])
        write_new(
            root / "complete.json",
            {
                "status": "complete",
                "completed": completed,
                "training_audit_sha256": sha(TRAINING / "training_audit.json"),
                "evaluation_audit_sha256": sha(OUTPUT / "audit.json"),
                "qualified": audit["qualified"],
                "deployment_changed": False,
                "protocol_sha256": sha(root / "protocol.json"),
            },
        )
    except BaseException as error:
        write_new(root / "failed.json", {"completed": completed, "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    args = parser.parse_args()
    producer = Producer(args.producer_pid)
    try:
        finish(args.output, producer)
    finally:
        producer.close()
