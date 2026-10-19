"""Recover the failed duration evaluator in a fresh directory, preserving evidence."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.evaluate_published_duration import cases
from scripts.finish_published_duration import Producer, checked
from scripts.run_arbitration_study import read, sha

PROTOCOL_SHA = "76e1b6af987e21f4e05e270aaabcd85ee9a89e68621589d1802a3a7543b24f18"
SOURCE = Path("runs/published_duration_evaluation")
QUEUE = Path("runs/published_duration_completion")
CAPABILITY = Path("runs/published_duration_recovery_capability/receipt.json")
OLD_PIDS = (22404, 32628, 60412, 63224, 62760, 32088, 36092, 18716)


def write_new(path, value):
    # No periodic replace of a status file: readers cannot deny its replacement.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows_info(path, schedule):
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    lines = raw[:end].splitlines()
    if len(lines) > len(schedule):
        raise ValueError("Too many saved matches")
    work = {"decisions": 0, "round_endings": 0, "full_matches": 0}
    for line, case in zip(lines, schedule):
        row = json.loads(line)
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError("Saved match order or identity changed")
        if row["work"]["full_matches"] != 1 or row["work"]["decisions"] != len(
            row["actions"]
        ):
            raise ValueError("Saved work differs")
        for name in work:
            value = row["work"][name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid work")
            work[name] += value
    return {
        "complete_rows": len(lines),
        "prefix_bytes": end,
        "trailing_bytes": len(raw) - end,
        "work": work,
    }


def schedule_for(protocol, key):
    cell = protocol["cells"][key]
    return cases(
        cell["parent"], cell["opponent"], protocol["pairs"], cell["first_seed"]
    )


def verify_cell(root, protocol, key):
    path = root / "evaluation" / f"{key}.jsonl"
    info = rows_info(path, schedule_for(protocol, key))
    report = read(path.with_suffix(".json"))
    cell = protocol["cells"][key]
    if not (
        info["complete_rows"] == 2 * protocol["pairs"]
        and info["trailing_bytes"] == 0
        and report["status"] == "complete"
        and report["cell"] == key
        and all(report[k] == v for k, v in cell.items())
        and report["work"] == info["work"]
        and report["rows_sha256"] == sha(path)
        and report["protocol_sha256"] == sha(root / "evaluation_protocol.json")
        and report["checkpoint_sha256"] == sha(cell["checkpoint"])
        and report["runtime"]["device"] == "cpu"
        and report["runtime"]["cpu_threads"] == 2
    ):
        raise ValueError("Incomplete or inconsistent saved cell")
    return info


def compare_prefix(old, new, info):
    prefix = old.read_bytes()[: info["prefix_bytes"]]
    with new.open("rb") as stream:
        if stream.read(len(prefix)) != prefix:
            raise ValueError("Repeated neural inference differs from preserved prefix")


def require_old_processes_stopped():
    for pid in OLD_PIDS:
        try:
            process = Producer(pid)
        except OSError as exc:
            if exc.winerror == 87:  # ERROR_INVALID_PARAMETER: PID absent.
                continue
            raise
        try:
            if process.exit_code() is None:
                raise RuntimeError(
                    f"Original process PID {pid} still live; do not recover"
                )
        finally:
            process.close()


def capability(root):
    import torch

    from scripts.audit_published_duration_evaluation import replay

    original = Path("runs/published_duration_evaluation_capability")
    protocol = read(original / "evaluation_protocol.json")
    original_audit = read(original / "audit.json")
    if protocol["mode"] != "capability" or original_audit["status"] != "passed":
        raise ValueError("Require completed original capability")
    checked(original_audit["inputs"])
    inputs = dict(original_audit["inputs"])
    for path in (
        Path(__file__),
        Path("tests/test_recover_published_duration.py"),
        Path("runs/published_duration_recovery_tests.xml"),
        Path("scripts/finish_published_duration.py"),
        original / "audit.json",
    ):
        inputs[str(path)] = sha(path)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(
        original / "evaluation_protocol.json", root / "evaluation_protocol.json"
    )
    keys = ["parent1_short_wp", "parent1_short_point"]
    run_workers(root, keys)
    torch.set_num_threads(2)
    work = {"decisions": 0, "round_endings": 0, "full_matches": 0}
    for key in keys:
        info = verify_cell(root, protocol, key)
        old, new = (
            folder / "evaluation" / f"{key}.jsonl" for folder in (original, root)
        )
        compare_prefix(old, new, rows_info(old, schedule_for(protocol, key)))
        with new.open(encoding="utf-8") as stream:
            for line in stream:
                replay(json.loads(line))
        for name in work:
            work[name] += info["work"][name]
        for path in (new, new.with_suffix(".json")):
            inputs[str(path)] = sha(path)
    inputs[str(root / "evaluation_protocol.json")] = sha(
        root / "evaluation_protocol.json"
    )
    checked(inputs)
    write_new(
        root / "receipt.json",
        {
            "status": "passed",
            "inputs": inputs,
            "cells": keys,
            "new_execution_work": work,
            "prefixes_exact": keys,
            "scope": "Eight repeated capability matches through original CPU inference; byte-exact equality with original rows and independent native CPU action replay. No strength evidence or new independent deals.",
        },
    )


def prepare(root):
    require_old_processes_stopped()
    protocol_path = SOURCE / "evaluation_protocol.json"
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("Original evaluation protocol changed")
    protocol = read(protocol_path)
    checked(protocol["inputs"])
    if (
        read(SOURCE / "evaluation_process.json")["status"] != "failed"
        or read(QUEUE / "process.json")["status"] != "failed"
    ):
        raise ValueError("Original jobs must be terminal failed")
    capability = read(CAPABILITY)
    if (
        capability["status"] != "passed"
        or capability["new_execution_work"]["full_matches"] != 8
        or capability["cells"] != capability["prefixes_exact"]
        or capability["cells"] != ["parent1_short_wp", "parent1_short_point"]
    ):
        raise ValueError("Recovery capability must pass")
    checked(capability["inputs"])
    inputs = dict(protocol["inputs"])
    for folder in (SOURCE, QUEUE):
        for path in folder.rglob("*"):
            if path.is_file():
                inputs[str(path)] = sha(path)
    for path in (
        Path(__file__),
        Path("tests/test_recover_published_duration.py"),
        CAPABILITY,
    ):
        inputs[str(path)] = sha(path)
    completed, partial, pending = [], {}, []
    for key in protocol["cells"]:
        path = SOURCE / "evaluation" / f"{key}.jsonl"
        if path.with_suffix(".json").exists():
            verify_cell(SOURCE, protocol, key)
            completed.append(key)
        else:
            pending.append(key)
            if path.exists():
                partial[key] = rows_info(path, schedule_for(protocol, key))
    if set(completed) != set(read(SOURCE / "evaluation_process.json")["completed"]):
        raise ValueError("Failed receipt and saved completed cells disagree")
    checked(inputs)
    root.mkdir(parents=True, exist_ok=False)
    (root / "evaluation").mkdir()
    shutil.copyfile(protocol_path, root / "evaluation_protocol.json")
    for key in completed:
        for suffix in (".jsonl", ".json"):
            name = key + suffix
            shutil.copyfile(SOURCE / "evaluation" / name, root / "evaluation" / name)
    write_new(
        root / "recovery_protocol.json",
        {
            "source": str(SOURCE),
            "failed_queue": str(QUEUE),
            "inputs": inputs,
            "protocol_sha256": PROTOCOL_SHA,
            "copied": completed,
            "rerun_whole_cells": pending,
            "preserved_partial": partial,
            "scope": "Copy completed cells byte-for-byte; rerun every incomplete cell on its original full schedule. Compare all preserved complete prefix rows exactly. No outcome-based selection; original failed files remain unchanged. Partial recorded work is additional duplicate execution, not extra independent evidence. Unrecorded interrupted-match work is unknown.",
        },
    )


def run_workers(root, keys):
    pending, running = list(keys), {}
    try:
        while pending or running:
            while pending and len(running) < 2:
                key = pending.pop(0)
                log = (root / f"evaluation_{key}.log").open("x", encoding="utf-8")
                try:
                    worker = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "scripts.evaluate_published_duration",
                            "cell",
                            "--protocol",
                            str(root / "evaluation_protocol.json"),
                            "--cell",
                            key,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0,
                    )
                except BaseException:
                    log.close()
                    raise
                running[key] = (worker, log)
                print(json.dumps({"started": key, "pid": worker.pid}), flush=True)
            for key, (worker, log) in list(running.items()):
                result = worker.poll()
                if result is not None:
                    log.close()
                    del running[key]
                    if result != 0:
                        raise RuntimeError(f"{key} exited {result}")
                    print(json.dumps({"completed": key}), flush=True)
            if running:
                time.sleep(0.5)
    finally:
        for worker, log in running.values():
            if worker.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                        check=True,
                        capture_output=True,
                    )
                else:
                    worker.terminate()
                worker.wait(timeout=10)
            log.close()


def run(root):
    recovery_path = root / "recovery_protocol.json"
    recovery = read(recovery_path)
    checked(recovery["inputs"])
    require_old_processes_stopped()
    write_new(
        root / "recovery_started.json",
        {"pid": os.getpid(), "protocol_sha256": sha(recovery_path)},
    )
    try:
        protocol = read(root / "evaluation_protocol.json")
        if sha(root / "evaluation_protocol.json") != PROTOCOL_SHA:
            raise ValueError("Copied protocol changed")
        run_workers(root, recovery["rerun_whole_cells"])
        for key in protocol["cells"]:
            verify_cell(root, protocol, key)
        for key in recovery["copied"]:
            for suffix in (".json", ".jsonl"):
                if sha(root / "evaluation" / (key + suffix)) != sha(
                    SOURCE / "evaluation" / (key + suffix)
                ):
                    raise ValueError("Copied completed cell changed")
        for key, info in recovery["preserved_partial"].items():
            compare_prefix(
                SOURCE / "evaluation" / f"{key}.jsonl",
                root / "evaluation" / f"{key}.jsonl",
                info,
            )
        checked(recovery["inputs"])
        write_new(
            root / "evaluation_process.json",
            {"status": "complete_awaiting_audit", "completed": list(protocol["cells"])},
        )
        with (root / "audit.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.audit_published_duration_evaluation",
                    "--input",
                    str(root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        audit = read(root / "audit.json")
        if audit["status"] != "passed":
            raise ValueError("Independent audit did not pass")
        checked(recovery["inputs"])
        write_new(
            root / "recovery_complete.json",
            {
                "status": "complete",
                "recovery_protocol_sha256": sha(recovery_path),
                "audit_sha256": sha(root / "audit.json"),
                "qualified": audit["qualified"],
                "deployment_changed": False,
                "prefixes_exact": list(recovery["preserved_partial"]),
                "additional_recorded_duplicate_work": {
                    name: sum(
                        info["work"][name]
                        for info in recovery["preserved_partial"].values()
                    )
                    for name in ("decisions", "round_endings", "full_matches")
                },
                "unrecorded_interrupted_work": "unknown",
            },
        )
    except BaseException as exc:
        write_new(root / "recovery_failed.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capability", "prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"capability": capability, "prepare": prepare, "run": run}[args.command](
        args.output
    )
