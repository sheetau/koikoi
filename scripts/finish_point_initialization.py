"""Wait on the exact Windows learner, then evaluate and audit the fixed study once."""

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.evaluate_point_initialization import (
    TRAINING,
    TRAINING_SHA,
    checked,
    write_new,
)
from scripts.finish_published_duration import Producer
from scripts.run_arbitration_study import read, sha

OUTPUT = Path("runs/point_initialization_evaluation")
CAPABILITY = Path("runs/point_initialization_evaluation_capability")


def wait_for_training(producer, root, poll=30):
    while True:
        if (root / "failed.json").exists():
            raise RuntimeError("Training failure receipt exists")
        code = producer.exit_code()
        if code is not None:
            if code != 0:
                raise RuntimeError(f"Exact training process exited {code}")
            complete = read(root / "complete.json")
            if (
                complete["status"] != "complete"
                or complete["protocol_sha256"] != TRAINING_SHA
            ):
                raise RuntimeError("Training exited without matching completion")
            return
        time.sleep(poll)


def jobs():
    return [
        (
            "prepare",
            [
                "scripts.evaluate_point_initialization",
                "prepare",
                "--output",
                str(OUTPUT),
            ],
        ),
        (
            "evaluate",
            [
                "scripts.evaluate_point_initialization",
                "run",
                "--protocol",
                str(OUTPUT / "evaluation_protocol.json"),
            ],
        ),
        (
            "audit",
            ["scripts.audit_point_initialization_evaluation", "--input", str(OUTPUT)],
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
        try:
            write_new(root / f"{name}_started.json", {"pid": process.pid})
            code = process.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; preserve outputs")
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=True,
                        capture_output=True,
                    )
                else:
                    process.terminate()
                process.wait(timeout=10)
    write_new(root / f"{name}_complete.json", {"exit_code": 0})


def finish(root, producer):
    root.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        if OUTPUT.exists():
            raise ValueError("Main evaluation already exists")
        if producer.exit_code() is not None:
            raise ValueError("Attach only to a verified live training process")
        if read(TRAINING / "started.json")["pid"] != producer.identity["pid"]:
            raise ValueError("Training receipt and held process differ")
        assert sha(TRAINING / "protocol.json") == TRAINING_SHA
        cap = read(CAPABILITY / "audit.json")
        assert cap["status"] == "passed"
        assert cap["protocol_sha256"] == sha(CAPABILITY / "evaluation_protocol.json")
        assert read(CAPABILITY / "evaluation_protocol.json")["mode"] == "capability"
        inputs = read(TRAINING / "protocol.json")["inputs"] | cap["inputs"]
        for path in (
            Path(__file__),
            Path("scripts/finish_published_duration.py"),
            Path(sys.executable),
            TRAINING / "protocol.json",
            TRAINING / "started.json",
            CAPABILITY / "audit.json",
            Path("tests/test_finish_point_initialization.py"),
            Path("runs/point_initialization_queue_tests.xml"),
        ):
            inputs[str(path)] = sha(path)
        suite = (
            ET.parse("runs/point_initialization_queue_tests.xml")
            .getroot()
            .find("testsuite")
        )
        assert int(suite.attrib["tests"]) >= 4
        assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
        checked(inputs)
        write_new(
            root / "protocol.json",
            {
                "producer": producer.identity,
                "queue_pid": os.getpid(),
                "inputs": inputs,
                "commands": jobs(),
                "cwd": str(Path.cwd()),
                "scope": "Exact producer exit0 and completion, then fixed evaluation and audit once. No restart or model promotion.",
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
        write_new(
            root / "complete.json",
            {
                "status": "complete",
                "completed": completed,
                "audit_sha256": sha(OUTPUT / "audit.json"),
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
