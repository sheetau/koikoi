"""Strict CPU duration evaluation, gated on completed training; separate capability."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_opponent import runtime_identity
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.audit_published_duration_checkpoint import audit
from scripts.published_duration_analysis import OPPONENTS, POLICIES
from scripts.published_duration_study import INITIAL, SEEDS, checked
from scripts.run_arbitration_study import PARENTS, read, sha, write


def cases(parent, opponent, pairs, first_seed):
    if parent not in (1, 2) or opponent not in OPPONENTS or pairs < 1:
        raise ValueError("Invalid duration cell")
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
    write(
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
    """At most two owned CPU workers; a failure stops only owned live workers."""
    protocol = read(protocol_path)
    checked(protocol["inputs"])
    root = protocol_path.parent
    if (root / "evaluation_process.json").exists():
        raise ValueError("Existing execution record; inspect before recovery")
    pending, running, completed = list(protocol["cells"]), {}, []
    try:
        while pending or running:
            while pending and len(running) < 2:
                key = pending.pop(0)
                log = (root / f"evaluation_{key}.log").open("x", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "scripts.evaluate_published_duration",
                    "cell",
                    "--protocol",
                    str(protocol_path),
                    "--cell",
                    key,
                ]
                try:
                    worker = subprocess.Popen(
                        command,
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
            for key, (worker, log) in list(running.items()):
                result = worker.poll()
                if result is not None:
                    log.close()
                    del running[key]
                    if result != 0:
                        raise RuntimeError(
                            f"Evaluation cell {key} failed with exit {result}"
                        )
                    completed.append(key)
            write(
                root / "evaluation_process.json",
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "workers": {key: p.pid for key, (p, _) in running.items()},
                    "completed": completed,
                },
            )
            if running:
                time.sleep(0.5)
        checked(protocol["inputs"])
        write(
            root / "evaluation_process.json",
            {"status": "complete_awaiting_audit", "completed": completed},
        )
    except BaseException as exc:
        for worker, log in running.values():
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            log.close()
        write(
            root / "evaluation_process.json",
            {"status": "failed", "completed": completed, "error": repr(exc)},
        )
        raise


def prepare(root, capability=False):
    training = Path("runs/published_duration_study")
    source = read(training / "protocol.json")
    checked(source["inputs"])
    inputs = dict(source["inputs"])
    inputs[str(training / "protocol.json")] = sha(training / "protocol.json")
    models, audits = {}, {}
    if not capability:
        complete = read(training / "training_complete.json")
        assert (
            complete["status"] == "complete" and complete["evaluation"] == "not started"
        )
        for parent, seed in enumerate(SEEDS, 1):
            folder = training / f"seed{seed}"
            receipt = read(folder / "complete.json")
            assert receipt == complete["cases"][str(seed)]
            assert (
                receipt["status"] == "complete"
                and receipt["counters"]["updates"] == source["long"]
            )
            assert receipt["final_sha256"] == sha(
                folder / f"update_{source['long']:08d}.pt"
            )
            for stage in (source["warmup"], source["short"], source["long"]):
                audits[f"seed{seed}/{stage}"] = audit(training, seed, stage)
            for name, stage in (("short", source["short"]), ("long", source["long"])):
                models[parent, name] = folder / f"update_{stage:08d}.pt"
            for name in (
                "initial.pt",
                "complete.json",
                "metrics.jsonl",
                "config.json",
                "update_00000020.pt",
            ):
                inputs[str(folder / name)] = sha(folder / name)
        inputs[str(training / "training_complete.json")] = sha(
            training / "training_complete.json"
        )
    else:
        models = {
            (parent, name): INITIAL for parent in (1, 2) for name in ("short", "long")
        }
    for parent in (1, 2):
        models[parent, "parent"] = Path(PARENTS[parent - 1][0])
    for path in models.values():
        inputs[str(path)] = sha(path)
    for name in (
        "scripts/evaluate_published_duration.py",
        "scripts/published_duration_analysis.py",
        "scripts/audit_published_duration_checkpoint.py",
        "scripts/audit_published_duration_evaluation.py",
        "tests/test_published_duration_analysis.py",
        "tests/test_published_duration_evaluation.py",
    ):
        inputs[name] = sha(name)
    if not capability:
        ready = Path("runs/published_duration_evaluation_capability")
        cap = read(ready / "audit.json")
        assert cap["status"] == "passed"
        assert cap["protocol_sha256"] == sha(ready / "evaluation_protocol.json")
        checked(cap["inputs"])
        inputs[str(ready / "audit.json")] = sha(ready / "audit.json")
    root.mkdir(parents=True, exist_ok=False)
    cells = {}
    for parent in (1, 2):
        for policy in POLICIES:
            for opponent in OPPONENTS:
                first = (
                    (99030001 + 10000 * OPPONENTS.index(opponent))
                    if capability
                    else source["evaluation"][opponent + "_seed"]
                )
                cells[f"parent{parent}_{policy}_{opponent}"] = {
                    "parent": parent,
                    "policy": policy,
                    "opponent": opponent,
                    "checkpoint": str(models[parent, policy]),
                    "first_seed": first,
                }
    write(
        root / "evaluation_protocol.json",
        {
            "mode": "capability" if capability else "main",
            "pairs": 2 if capability else source["evaluation"]["pairs_per_opponent"],
            "inputs": inputs,
            "cells": cells,
            "training_audits": audits,
            "bootstrap": source["bootstrap"],
            "main_training_protocol_sha256": sha(training / "protocol.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
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


if __name__ == "__main__":
    main()
