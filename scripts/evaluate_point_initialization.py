"""Fixed eight-policy Point-initialization evaluation with exclusive receipts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_opponent import runtime_identity
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.audit_point_initialization_checkpoint import audit as audit_checkpoint
from scripts.phase_temperature_ppo import PhaseTemperatureAgent
from scripts.point_initialization_analysis import OPPONENTS, POLICIES
from scripts.published_duration_study import assert_equal
from scripts.run_arbitration_study import PARENTS, read, sha

TRAINING = Path("runs/point_initialization_study")
TRAINING_SHA = "defd0e993e2c85efb844164b0c1edfa4ac6bae917c31ea6a6eb33d7ba4618048"
CAPABILITY = Path("runs/point_initialization_evaluation_capability")


def checked(inputs):
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError(f"Changed evaluation input: {name}")


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def cases(parent, opponent, pairs, first_seed):
    if parent not in (1, 2) or opponent not in OPPONENTS or pairs < 1:
        raise ValueError("Invalid Point-initialization cell")
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "replicate": i,
            "seat": seat,
            "seed": first_seed + i,
        }
        for i in range(pairs)
        for seat in (0, 1)
    ]


def play_match(candidate, opponent, case):
    env = ReferenceEnv(case["seed"], "paper8")
    initial_hash = hashlib.sha256(env.state.tobytes()).hexdigest()
    actions, rounds = [], 0
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise ValueError("Incomplete match at safety bound")
        actor = candidate if env.state[PLAYER] == case["seat"] else opponent
        action = int(actor.act(env))
        assert env.legal()[action]
        result = env.step(action)
        assert int(result[0]) != 99
        actions.append(action)
        rounds += int(result[1])
    points = int(env.state[POINTS] - env.state[POINTS + 1]) * (1 - 2 * case["seat"])
    return {
        **case,
        "initial_sha256": initial_hash,
        "points": points,
        "actions": actions,
        "endpoint": env.state.tolist(),
        "work": {"decisions": len(actions), "round_endings": rounds, "full_matches": 1},
    }


def evaluate_cell(protocol_path, key):
    protocol = read(protocol_path)
    checked(protocol["inputs"])
    cell = protocol["cells"][key]
    torch.set_num_threads(2)
    candidate = (
        NeuralAgent(cell["checkpoint"], device="cpu", deterministic=True)
        if cell["policy"] == "parent"
        else PhaseTemperatureAgent(cell["checkpoint"])
    )
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


def stop_workers(running):
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
                            "scripts.evaluate_point_initialization",
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
    if root.exists():
        raise ValueError("Preserve existing evaluation")
    if sha(TRAINING / "protocol.json") != TRAINING_SHA:
        raise ValueError("Main training protocol changed")
    source = read(TRAINING / "protocol.json")
    checked(source["inputs"])
    inputs = dict(source["inputs"])
    inputs[str(TRAINING / "protocol.json")] = TRAINING_SHA
    models, audits = {}, {}
    if capability:
        fixture = Path("runs/point_initialization_capability")
        complete = read(fixture / "independent_audit.json")
        assert complete["status"] == "passed"
        checked(complete["inputs"])
        inputs.update(complete["inputs"])
        inputs[str(fixture / "independent_audit.json")] = sha(
            fixture / "independent_audit.json"
        )
        for parent in (1, 2):
            for policy in POLICIES[1:]:
                models[parent, policy] = fixture / policy / "initial.pt"
    else:
        # Reading completion is mandatory before any main output directory exists.
        complete = read(TRAINING / "complete.json")
        assert (
            complete["status"] == "complete" and complete["evaluation"] == "not_started"
        )
        assert not (TRAINING / "failed.json").exists()
        assert complete["protocol_sha256"] == TRAINING_SHA
        assert set(complete["arms"]) == set(source["configs"])
        inputs[str(TRAINING / "complete.json")] = sha(TRAINING / "complete.json")
        totals = dict.fromkeys(("decisions", "rounds", "matches", "updates"), 0)
        initial = {}
        for parent in (1, 2):
            for policy in POLICIES[1:]:
                arm = f"seed{parent}_{policy}"
                folder = TRAINING / arm
                receipt = read(folder / "complete.json")
                assert receipt == complete["arms"][arm]
                path = folder / "update_00000520.pt"
                assert receipt["checkpoint_sha256"] == sha(path)
                assert receipt["metrics_sha256"] == sha(folder / "metrics.jsonl")
                assert receipt["protocol_sha256"] == TRAINING_SHA
                for stage in (20, 520):
                    audits[f"{arm}/{stage}"] = audit_checkpoint(TRAINING, arm, stage)
                assert (
                    receipt["counters"]
                    == audits[f"{arm}/520"]["checkpoint"]["counters"]
                )
                for name in totals:
                    totals[name] += receipt["counters"][name]
                initial[parent, policy] = torch.load(
                    folder / "initial.pt", map_location="cpu", weights_only=False
                )
                models[parent, policy] = path
                for name in (
                    "initial.pt",
                    "config.json",
                    "metrics.jsonl",
                    "complete.json",
                    "update_00000020.pt",
                ):
                    inputs[str(folder / name)] = sha(folder / name)
        assert totals == complete["total_training_work"]
        assert totals["decisions"] == source["total_decisions"]
        for parent in (1, 2):
            for policy in ("point_fixed", "point_matched"):
                for key in (
                    "value",
                    "optimizer",
                    "env",
                    "torch_rng",
                    "cuda_rng",
                    "python_rng",
                    "numpy_rng",
                ):
                    assert_equal(
                        initial[parent, policy][key], initial[parent, "wp_fixed"][key]
                    )
            assert_equal(
                initial[parent, "point_fixed"]["actor"],
                initial[parent, "point_matched"]["actor"],
            )
        cap = read(CAPABILITY / "audit.json")
        assert cap["status"] == "passed"
        assert cap["protocol_sha256"] == sha(CAPABILITY / "evaluation_protocol.json")
        assert read(CAPABILITY / "evaluation_protocol.json")["mode"] == "capability"
        checked(cap["inputs"])
        inputs.update(cap["inputs"])
        inputs[str(CAPABILITY / "audit.json")] = sha(CAPABILITY / "audit.json")
    for parent, (path, digest) in enumerate(PARENTS, 1):
        assert sha(path) == digest
        models[parent, "parent"] = Path(path)
    for path in models.values():
        inputs[str(path)] = sha(path)
    for name in (
        "scripts/evaluate_point_initialization.py",
        "scripts/point_initialization_analysis.py",
        "scripts/audit_point_initialization_checkpoint.py",
        "scripts/audit_point_initialization_evaluation.py",
        "scripts/directional_return_statistics.py",
        "scripts/directional_return_probe.py",
        "tests/test_point_initialization_analysis.py",
        "tests/test_point_initialization_evaluation.py",
        "tests/test_point_checkpoint_evidence.py",
        "runs/point_initialization_evaluation_tests.xml",
    ):
        inputs[name] = sha(name)
    suite = (
        ET.parse("runs/point_initialization_evaluation_tests.xml")
        .getroot()
        .find("testsuite")
    )
    assert int(suite.attrib["tests"]) >= 9
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    checked(inputs)
    cells = {}
    for parent in (1, 2):
        for policy in POLICIES:
            for opponent in OPPONENTS:
                first = (
                    (103010001 + 10000 * OPPONENTS.index(opponent))
                    if capability
                    else source["evaluation"][opponent + "_first_seed"]
                )
                cells[f"parent{parent}_{policy}_{opponent}"] = {
                    "parent": parent,
                    "policy": policy,
                    "opponent": opponent,
                    "checkpoint": str(models[parent, policy]),
                    "first_seed": first,
                }
    root.mkdir(parents=True, exist_ok=False)
    write_new(
        root / "evaluation_protocol.json",
        {
            "mode": "capability" if capability else "main",
            "pairs": 2 if capability else 4096,
            "inputs": inputs,
            "cells": cells,
            "training_audits": audits,
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
