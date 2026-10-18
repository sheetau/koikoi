"""After the fixed tables finish, collect both references and run fixed analysis."""

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.run_arbitration_study import sha, write


def run(folder):
    receipt = folder / "completion_process.json"
    if receipt.exists():
        raise ValueError("Preserve any previous completion queue")
    paths = [Path(__file__)] + [
        Path(p)
        for p in (
            "scripts/collect_round_value_reference.py",
            "scripts/round_value_reference.py",
            "scripts/analyze_round_value_accuracy.py",
            "scripts/round_value_uncertainty.py",
            "tests/test_round_value_reference.py",
            "tests/test_round_value_uncertainty.py",
            "docs/ROUND_VALUE_ACCURACY.rst",
            "runs/round_value_accuracy_tests.xml",
        )
    ]
    inputs = {str(p): sha(p) for p in paths}
    tests = ET.parse("runs/round_value_accuracy_tests.xml").getroot().find("testsuite")
    assert tests is not None and int(tests.attrib["tests"]) == 6
    assert all(int(tests.attrib[k]) == 0 for k in ("failures", "errors", "skipped"))
    write(
        folder / "completion_protocol.json",
        {
            "inputs": inputs,
            "jobs": ["reference_parent1", "reference_parent2", "analysis"],
            "maximum_table_wait_seconds": 7200,
        },
    )
    completed = []
    started = time.monotonic()
    write(receipt, {"status": "waiting_tables", "completed": completed})
    try:
        while True:
            state = json.loads(
                (folder / "tables_process.json").read_text(encoding="utf-8")
            )
            if state["status"] == "complete":
                break
            if state["status"] != "running" or time.monotonic() - started > 7200:
                raise RuntimeError(
                    "Table runner failed or exceeded the queue's fixed wait"
                )
            time.sleep(5)
        jobs = [
            (
                f"reference_parent{parent}",
                [
                    sys.executable,
                    "-m",
                    "scripts.collect_round_value_reference",
                    "--parent",
                    str(parent),
                    "--output",
                    str(folder / f"reference_parent{parent}"),
                ],
            )
            for parent in (1, 2)
        ] + [
            (
                "analysis",
                [
                    sys.executable,
                    "-m",
                    "scripts.analyze_round_value_accuracy",
                    str(folder),
                ],
            )
        ]
        for name, command in jobs:
            assert all(sha(p) == h for p, h in inputs.items())
            write(
                receipt, {"status": "running", "current": name, "completed": completed}
            )
            with (folder / f"completion_{name}.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as log:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                raise RuntimeError(
                    f"{name} exited {result.returncode}; preserve its log and outputs"
                )
            completed.append(name)
            print(
                json.dumps({"completed": name, "seconds": time.monotonic() - started}),
                flush=True,
            )
        assert all(sha(p) == h for p, h in inputs.items())
        write(
            folder / "completion_audit.json",
            {
                "status": "passed",
                "inputs": inputs,
                "completed": completed,
                "protocol_sha256": sha(folder / "completion_protocol.json"),
                "analysis_sha256": sha(folder / "analysis.json"),
                "seconds": time.monotonic() - started,
            },
        )
        write(
            receipt,
            {
                "status": "complete",
                "audit_sha256": sha(folder / "completion_audit.json"),
            },
        )
    except Exception as exc:
        write(receipt, {"status": "failed", "completed": completed, "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder)
