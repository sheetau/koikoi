"""Wait for the frozen study, audit it once, then publish its complete summary."""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.audit_koi_temperature_study import audit, checked
from scripts.run_arbitration_study import read, sha, write


def summary(root):
    output = root / "summary.json"
    if output.exists():
        raise ValueError("Preserve completed summary")
    report = read(root / "independent_audit.json")
    assert report["status"] == "passed"
    checked(report["inputs"])
    training = {
        key: sum(c[key] for c in report["training_counters"].values())
        for key in ("updates", "decisions", "rounds", "matches")
    }
    assert training["updates"] == 2080 and training["decisions"] == 34078720
    result = {
        "status": "complete",
        "analysis": report["analysis"],
        "training_work": training,
        "training_counters": report["training_counters"],
        "evaluation_work": report["evaluation_work"],
        "evaluation_replay_work": report["evaluation_replay_work"],
        "shared_evaluation_deals": report["shared_evaluation_deals"],
        "inputs": {
            **report["inputs"],
            str(root / "independent_audit.json"): sha(root / "independent_audit.json"),
            str(Path(__file__)): sha(__file__),
        },
        "conclusion": (
            "Both seeds pass both prespecified mixture-win contrasts; this development result needs a separately designed confirmation."
            if report["analysis"]["replicated_strength_gain"]
            else "The prespecified replicated improvement criterion fails; secondary results do not qualify the candidate."
        ),
        "deployment_changed": False,
        "scope": "Complete development results. No final confirmation, canonical-model replacement or general novelty claim.",
    }
    write(output, result)
    print(
        json.dumps({"status": "complete", "conclusion": result["conclusion"]}),
        flush=True,
    )
    return result


def finish(root, wait):
    protocol = root / "finish_protocol.json"
    if protocol.exists():
        raise ValueError(
            "A finalizer was already launched; inspect its process and receipts"
        )
    paths = [Path(__file__), root / "protocol.json"] + [
        Path(p)
        for p in (
            "scripts/audit_koi_temperature_study.py",
            "scripts/audit_koi_temperature_capability.py",
            "tests/test_audit_koi_temperature_study.py",
            "runs/koi_temperature_audit_tests.xml",
            "scripts/koi_temperature_analysis.py",
            "scripts/run_arbitration_study.py",
        )
    ]
    inputs = {str(p): sha(p) for p in paths}
    write(
        protocol,
        {
            "inputs": inputs,
            "scope": "Wait, independent full audit, complete summary; no training or selection changes.",
        },
    )
    try:
        while True:
            state = read(root / "process.json")["status"]
            if state == "complete_awaiting_audit":
                break
            if state not in ("training", "evaluating") or not wait:
                raise RuntimeError(f"Producer is not complete: {state}")
            write(
                root / "finish_process.json",
                {"status": "waiting", "producer_status": state},
            )
            time.sleep(30)
        checked(inputs)
        checked(read(root / "protocol.json")["inputs"])
        write(root / "finish_process.json", {"status": "auditing"})
        audit(root)
        checked(inputs)
        write(root / "finish_process.json", {"status": "summarizing"})
        summary(root)
        checked(inputs)
        write(root / "finish_process.json", {"status": "complete"})
    except BaseException as error:
        write(root / "finish_process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    finish(args.output, args.wait)
