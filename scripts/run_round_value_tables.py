"""Execute the four fixed table cells, preserving logs and auditing each cell."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_arbitration_study import sha, write


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh study directory")
    paths = [
        Path(__file__),
        Path("scripts/build_round_value_table.py"),
        Path("scripts/audit_round_value_table.py"),
        Path("scripts/round_transition_values.py"),
        Path("docs/ROUND_VALUE_ACCURACY.rst"),
    ]
    inputs = {str(p): sha(p) for p in paths}
    jobs = [
        {"parent": p, "table": table, "seed": seed}
        for p in (1, 2)
        for table, seed in (("A", 42110001), ("B", 52110001))
    ]
    folder.mkdir(parents=True)
    write(
        folder / "tables_protocol.json",
        {"inputs": inputs, "samples": 128, "batch": 256, "jobs": jobs},
    )
    completed = []
    start = time.perf_counter()
    for job in jobs:
        assert all(sha(p) == h for p, h in inputs.items())
        name = f"parent{job['parent']}_{job['table']}"
        target = folder / name
        write(
            folder / "tables_process.json",
            {"status": "running", "current": name, "completed": completed},
        )
        command = [
            sys.executable,
            "-m",
            "scripts.build_round_value_table",
            "--parent",
            str(job["parent"]),
            "--samples",
            "128",
            "--batch-size",
            "256",
            "--seed",
            str(job["seed"]),
            "--output",
            str(target),
        ]
        with (folder / f"{name}.log").open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
            if result.returncode == 0:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.audit_round_value_table",
                        str(target),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        if result.returncode:
            write(
                folder / "tables_process.json",
                {
                    "status": "failed",
                    "current": name,
                    "exit_code": result.returncode,
                    "completed": completed,
                },
            )
            raise RuntimeError(f"Cell {name} failed; preserve its output and log")
        completed.append(
            {
                **job,
                "name": name,
                "audit_sha256": sha(target / "audit.json"),
                "independent_audit_sha256": sha(target / "independent_audit.json"),
            }
        )
        print(
            json.dumps({"completed": name, "seconds": time.perf_counter() - start}),
            flush=True,
        )
    assert all(sha(p) == h for p, h in inputs.items())
    work = dict.fromkeys(("decisions", "round_endings", "terminal_match_endings"), 0)
    for job in completed:
        report = json.loads(
            (folder / job["name"] / "audit.json").read_text(encoding="utf-8")
        )
        for key in work:
            work[key] += report["work"][key]
    assert work["round_endings"] == 1933312
    write(
        folder / "tables_audit.json",
        {
            "status": "passed",
            "inputs": inputs,
            "protocol_sha256": sha(folder / "tables_protocol.json"),
            "completed": completed,
            "work": work,
            "seconds": time.perf_counter() - start,
            "independent_complete_matches": 0,
            "policy_updates": 0,
        },
    )
    write(
        folder / "tables_process.json",
        {"status": "complete", "audit_sha256": sha(folder / "tables_audit.json")},
    )
    print(json.dumps({"status": "passed", "work": work}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
