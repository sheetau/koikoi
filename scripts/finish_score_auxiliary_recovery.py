"""Fixed evaluation after process-isolated recovery; original sources stay frozen.

Only artifact roots are adapted. All numerical training/evaluation algorithms,
seeds, endpoints and simultaneous-band criteria remain the original protocol.
"""

import argparse
import json
import os
from pathlib import Path

from scripts.finish_point_initialization import run_job
from scripts.finish_published_duration import Producer
from scripts.finish_score_auxiliary import wait_for_training
from scripts.recover_score_auxiliary import OUTPUT as TRAINING
from scripts.recover_score_auxiliary import PROTOCOL_SHA
from scripts.run_arbitration_study import read, sha
from scripts.train_score_auxiliary_study import checked, write_new

QUEUE = Path("runs/score_auxiliary_recovery_completion")
EVALUATION = Path("runs/score_auxiliary_evaluation")


def relocation_bindings():
    return {
        str(path): sha(path)
        for path in (
            Path(__file__),
            TRAINING / "recovery_protocol.json",
            TRAINING / "repeated_prefix.json",
            TRAINING / "complete.json",
        )
    }


def bind_relocation(path):
    """Bind relocation provenance before any scores or outcomes are computed."""
    value = read(path)
    value["inputs"].update(relocation_bindings())
    checked(value["inputs"])
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def prepare_evaluation():
    from scripts import evaluate_score_auxiliary as evaluator

    evaluator.TRAINING = TRAINING
    evaluator.prepare(EVALUATION)
    bind_relocation(EVALUATION / "evaluation_protocol.json")


def jobs():
    return [
        (
            "training_audit",
            ["scripts.audit_score_auxiliary_training", "--root", str(TRAINING)],
        ),
        ("prepare", ["scripts.finish_score_auxiliary_recovery", "prepare-evaluation"]),
        (
            "evaluate",
            [
                "scripts.evaluate_score_auxiliary",
                "run",
                "--protocol",
                str(EVALUATION / "evaluation_protocol.json"),
            ],
        ),
        (
            "evaluation_audit",
            ["scripts.audit_score_auxiliary_evaluation", "--input", str(EVALUATION)],
        ),
    ]


def finish(producer):
    if EVALUATION.exists() or (TRAINING / "training_audit.json").exists():
        raise ValueError("Preserve existing main evaluation/audit")
    if producer.exit_code() is not None:
        raise ValueError("Attach to the live recovery producer")
    assert read(TRAINING / "started.json")["pid"] == producer.identity["pid"]
    assert sha(TRAINING / "protocol.json") == PROTOCOL_SHA
    recovery = read(TRAINING / "recovery_protocol.json")
    inputs = dict(recovery["inputs"])
    # Freeze all reused stage implementations and their previously passed evidence.
    old_queue = read("runs/score_auxiliary_completion/protocol.json")
    inputs.update(old_queue["inputs"])
    for path in (
        Path(__file__),
        TRAINING / "recovery_protocol.json",
        TRAINING / "started.json",
        Path("tests/test_finish_score_auxiliary_recovery.py"),
        Path("runs/score_auxiliary_recovery_queue_tests.xml"),
    ):
        inputs[str(path)] = sha(path)
    checked(inputs)
    QUEUE.mkdir(parents=True, exist_ok=False)
    write_new(
        QUEUE / "protocol.json",
        {
            "producer": producer.identity,
            "queue_pid": os.getpid(),
            "inputs": inputs,
            "commands": jobs(),
            "cwd": str(Path.cwd()),
            "scope": "Process-isolated recovery of the original fixed study. Only training artifact roots relocated. Original failure remains preserved. Exact producer exit0, prefix equality and all six endpoints required before original audit/evaluation/audit stages. No promotion.",
        },
    )
    completed = []
    try:
        print({"waiting": producer.identity}, flush=True)
        wait_for_training(producer, TRAINING)
        assert read(TRAINING / "repeated_prefix.json")["status"] == "passed"
        checked(inputs)
        for name, arguments in jobs():
            print({"starting": name}, flush=True)
            run_job(name, arguments, QUEUE)
            completed.append(name)
            checked(inputs)
        audit = read(EVALUATION / "audit.json")
        assert audit["status"] == "passed"
        checked(audit["inputs"])
        write_new(
            QUEUE / "complete.json",
            {
                "status": "complete",
                "completed": completed,
                "training_audit_sha256": sha(TRAINING / "training_audit.json"),
                "evaluation_audit_sha256": sha(EVALUATION / "audit.json"),
                "qualified": audit["qualified"],
                "deployment_changed": False,
                "protocol_sha256": sha(QUEUE / "protocol.json"),
            },
        )
    except BaseException as error:
        write_new(QUEUE / "failed.json", {"error": repr(error), "completed": completed})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("wait", "prepare-evaluation"))
    parser.add_argument("--producer-pid", type=int)
    args = parser.parse_args()
    if args.command == "prepare-evaluation":
        prepare_evaluation()
    else:
        if args.producer_pid is None:
            parser.error("--producer-pid required")
        producer = Producer(args.producer_pid)
        try:
            finish(producer)
        finally:
            producer.close()
