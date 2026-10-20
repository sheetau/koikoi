"""Fixed score-auxiliary evaluation with complete-training/audit gating."""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.published_opponent import runtime_identity
from koikoi.upstream import PublishedAgent
from scripts.evaluate_point_initialization import (
    cases,
    checked,
    play_match,
    stop_workers,
    write_new,
)
from scripts.run_arbitration_study import PARENTS, read, sha
from scripts.score_auxiliary_analysis import OPPONENTS, POLICIES

TRAINING = Path("runs/score_auxiliary_study")
TRAINING_SHA = "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
FIXTURE = Path("runs/score_auxiliary_training_capability")
FIXTURE_SHA = "49e24dc6f33247b69b8922d71d2716fcdaf676220b75c2630785ea829b833f9e"
CAPABILITY = Path("runs/score_auxiliary_evaluation_capability")


def evaluate_cell(protocol_path, key):
    protocol = read(protocol_path)
    checked(protocol["inputs"])
    cell = protocol["cells"][key]
    torch.set_num_threads(2)
    candidate = NeuralAgent(cell["checkpoint"], device="cpu", deterministic=True)
    opponent = EndgameAgent(PublishedAgent(cell["opponent"]))
    assert candidate.profile == "paper8" and candidate.requires_endgame
    runtime = runtime_identity("cpu")
    folder = protocol_path.parent / "evaluation"
    folder.mkdir(exist_ok=True)
    totals = {"decisions": 0, "round_endings": 0, "full_matches": 0}
    path = folder / f"{key}.jsonl"
    if (folder / f"{key}.json").exists():
        raise ValueError("Preserve completed evaluation")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(
            cases(
                cell["parent"], cell["opponent"], protocol["pairs"], cell["first_seed"]
            ),
            1,
        ):
            row = play_match(candidate, opponent, case)
            stream.write(json.dumps(row) + "\n")
            for name, count in row["work"].items():
                totals[name] += count
            if index % 512 == 0:
                stream.flush()
                print(json.dumps({"cell": key, "matches": index}), flush=True)
    assert runtime == runtime_identity("cpu")
    checked(protocol["inputs"])
    write_new(
        folder / f"{key}.json",
        {
            "status": "complete",
            "cell": key,
            **cell,
            "work": totals,
            "runtime": runtime,
            "checkpoint_sha256": sha(cell["checkpoint"]),
            "rows_sha256": sha(path),
            "protocol_sha256": sha(protocol_path),
        },
    )


def run_cells(protocol_path):
    protocol = read(protocol_path)
    checked(protocol["inputs"])
    root = protocol_path.parent
    write_new(
        root / "evaluation_started.json",
        {"pid": os.getpid(), "protocol_sha256": sha(protocol_path)},
    )
    pending, running, completed = list(protocol["cells"]), {}, []
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
                            "scripts.evaluate_score_auxiliary",
                            "cell",
                            "--protocol",
                            str(protocol_path),
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
                    completed.append(key)
                    print(json.dumps({"completed": key}), flush=True)
            if running:
                time.sleep(0.5)
        checked(protocol["inputs"])
        write_new(
            root / "evaluation_complete.json",
            {
                "status": "complete_awaiting_audit",
                "completed": completed,
                "protocol_sha256": sha(protocol_path),
            },
        )
    except BaseException as error:
        stop_workers(running)
        write_new(
            root / "evaluation_failed.json",
            {"error": repr(error), "completed": completed},
        )
        raise


def prepare(root, capability=False):
    root = Path(root)
    if root.exists():
        raise ValueError("Preserve existing evaluation directory")
    if sha(TRAINING / "protocol.json") != TRAINING_SHA:
        raise ValueError("Changed main training protocol")
    source = read(TRAINING / "protocol.json")
    checked(source["inputs"])
    inputs = dict(source["inputs"])
    inputs[str(TRAINING / "protocol.json")] = TRAINING_SHA
    models = {}
    if capability:
        report = FIXTURE / "report.json"
        assert sha(report) == FIXTURE_SHA
        fixture = read(report)
        assert fixture["passed"] and not fixture["strength_evidence"]
        checked(fixture["files"])
        inputs.update(fixture["files"])
        inputs[str(report)] = FIXTURE_SHA
        for group in (1, 2):
            for policy in POLICIES[1:]:
                models[group, policy] = FIXTURE / policy / "cycle2/ppo.pt"
    else:
        # No main output exists until all six learners and their independent
        # audit are complete. A future audit producer supplies this contract.
        complete = read(TRAINING / "complete.json")
        assert complete["status"] == "complete"
        assert complete["protocol_sha256"] == TRAINING_SHA
        assert complete["evaluation"] == "not_started"
        assert not complete["deployment_changed"]
        assert not (TRAINING / "failed.json").exists()
        assert set(complete["arms"]) == set(source["arms"])
        audit_path = TRAINING / "training_audit.json"
        training_audit = read(audit_path)
        assert training_audit["format"] == "score_auxiliary_training_audit_v1"
        assert training_audit["status"] == "passed"
        assert training_audit["protocol_sha256"] == TRAINING_SHA
        assert training_audit["auditor_sha256"] == sha(
            "scripts/audit_score_auxiliary_training.py"
        )
        assert training_audit["complete_sha256"] == sha(TRAINING / "complete.json")
        assert set(training_audit["arms"]) == set(source["arms"])
        checked(training_audit["inputs"])
        inputs.update(training_audit["inputs"])
        inputs[str(audit_path)] = sha(audit_path)
        inputs["scripts/audit_score_auxiliary_training.py"] = sha(
            "scripts/audit_score_auxiliary_training.py"
        )
        inputs[str(TRAINING / "complete.json")] = sha(TRAINING / "complete.json")
        for group in (1, 2):
            for policy in POLICIES[1:]:
                arm = f"seed{group}_{policy}"
                folder = TRAINING / arm
                receipt = read(folder / "complete.json")
                assert receipt == complete["arms"][arm]
                endpoint = folder / "update_00000520"
                assert Path(receipt["endpoint"]) == endpoint
                assert receipt["manifest_sha256"] == sha(endpoint / "manifest.json")
                assert receipt["metrics_sha256"] == sha(folder / "metrics.jsonl")
                assert receipt["protocol_sha256"] == TRAINING_SHA
                assert training_audit["arms"][arm]["endpoint_manifest_sha256"] == sha(
                    endpoint / "manifest.json"
                )
                assert (
                    receipt["ppo_work"]["decisions"] == source["ppo_decisions_per_arm"]
                )
                assert (
                    receipt["auxiliary_work"]["matches"]
                    == source["auxiliary_matches_per_arm"]
                )
                models[group, policy] = endpoint / "ppo.pt"
        cap = read(CAPABILITY / "audit.json")
        assert cap["status"] == "passed" and not cap["qualified"]
        assert read(CAPABILITY / "evaluation_protocol.json")["mode"] == "capability"
        assert cap["protocol_sha256"] == sha(CAPABILITY / "evaluation_protocol.json")
        checked(cap["inputs"])
        inputs.update(cap["inputs"])
        inputs[str(CAPABILITY / "audit.json")] = sha(CAPABILITY / "audit.json")
    for group, (path, digest) in enumerate(PARENTS, 1):
        assert sha(path) == digest
        models[group, "parent"] = Path(path)
    for path in models.values():
        inputs[str(path)] = sha(path)
    for name in (
        "scripts/evaluate_score_auxiliary.py",
        "scripts/audit_score_auxiliary_evaluation.py",
        "scripts/score_auxiliary_analysis.py",
        "scripts/audit_score_auxiliary_statistics.py",
        "scripts/evaluate_point_initialization.py",
        "scripts/audit_point_initialization_evaluation.py",
        "scripts/point_initialization_analysis.py",
        "scripts/directional_return_statistics.py",
        "scripts/directional_return_probe.py",
        "scripts/phase_temperature_ppo.py",
        "tests/test_score_auxiliary_analysis.py",
        "tests/test_score_auxiliary_evaluation.py",
        "runs/score_auxiliary_evaluation_tests.xml",
    ):
        inputs[name] = sha(name)
    suite = (
        ET.parse("runs/score_auxiliary_evaluation_tests.xml")
        .getroot()
        .find("testsuite")
    )
    assert int(suite.attrib["tests"]) >= 10
    assert all(int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))
    checked(inputs)
    cells = {}
    for group in (1, 2):
        for policy in POLICIES:
            for opponent in OPPONENTS:
                first_seed = (
                    (110010001 + 10000 * OPPONENTS.index(opponent))
                    if capability
                    else source["evaluation"][opponent + "_first_seed"]
                )
                cells[f"parent{group}_{policy}_{opponent}"] = {
                    "parent": group,
                    "policy": policy,
                    "opponent": opponent,
                    "checkpoint": str(models[group, policy]),
                    "first_seed": first_seed,
                }
    root.mkdir(parents=True, exist_ok=False)
    write_new(
        root / "evaluation_protocol.json",
        {
            "mode": "capability" if capability else "main",
            "pairs": 2 if capability else 4096,
            "inputs": inputs,
            "cells": cells,
            "bootstrap": {
                "seed": source["evaluation"]["bootstrap_seed"],
                "replicates": source["evaluation"]["bootstrap_replicates"],
            },
            "main_training_protocol_sha256": TRAINING_SHA,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "capability", "run", "cell"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.mode in ("prepare", "capability"):
        if args.output is None:
            parser.error("--output required")
        prepare(args.output, args.mode == "capability")
    elif args.mode == "run":
        if args.protocol is None:
            parser.error("--protocol required")
        run_cells(args.protocol)
    else:
        if args.protocol is None or args.cell is None:
            parser.error("--protocol and --cell required")
        evaluate_cell(args.protocol, args.cell)
