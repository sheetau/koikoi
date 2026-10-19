"""Four fixed-budget learners, then complete paired CPU evaluation of six policies."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.koi_temperature_analysis import OPPONENTS, POLICIES, cases
from scripts.koi_temperature_ppo import (
    KoiTemperatureAgent,
    KoiTemperatureConfig,
    KoiTemperatureTrainer,
)
from scripts.run_arbitration_study import PARENTS, read, sha, write

CAPABILITY = Path("runs/koi_temperature_capability_corrected")
CALIBRATION = Path("runs/phase_exploration_calibration.json")


def checked(inputs):
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Changed study input: {path}")


def make_plan():
    capability = read(CAPABILITY / "audit.json")
    assert capability["status"] == "passed" and capability["matched_initial_exact"]
    assert all(a["exact_resume"] for a in capability["arms"].values())
    inputs = dict(capability["inputs"])
    independent = read(CAPABILITY / "independent_audit.json")
    assert independent["status"] == "passed"
    for name, digest in independent["inputs"].items():
        assert name not in inputs or inputs[name] == digest
        inputs[name] = digest
    calibration = read(CALIBRATION)
    assert (
        sha(CALIBRATION)
        == "d191ca3bafe0c3bb1f268f612a3dfb80f2223dc97292ed87d0571c6024487e0a"
    )
    paths = [
        Path(__file__),
        CAPABILITY / "audit.json",
        CAPABILITY / "independent_audit.json",
        CALIBRATION,
        Path("scripts/koi_temperature_analysis.py"),
        Path("docs/KOI_TEMPERATURE_STUDY.rst"),
        Path("tests/test_koi_temperature_study.py"),
        Path("runs/koi_temperature_study_tests.xml"),
    ]
    for name in (
        "runs/koi_temperature_ppo_tests_initial.xml",
        "runs/koi_temperature_study_tests.xml",
    ):
        suite = ET.parse(name).getroot().find("testsuite")
        assert int(suite.attrib["tests"]) >= 3
        assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    for path, digest in PARENTS:
        assert sha(path) == digest
        paths.append(Path(path))
    paths += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{opponent}.pt"
        for phase in ("discard", "pick", "koikoi")
        for opponent in OPPONENTS
    ]
    paths += list(UPSTREAM_PATH.glob("*.py"))
    inputs.update({str(p): sha(p) for p in paths})
    checked(inputs)
    configs = {
        f"seed{index}_{mode}": asdict(
            KoiTemperatureConfig(device="cuda", seed=seed, koi_temperature=temperature)
        )
        for index, seed in ((1, 96110001), (2, 96210001))
        for mode, temperature in (
            ("fixed", 0.05),
            ("calibrated", calibration["phase_temperatures"][3]),
        )
    }
    return {
        "inputs": inputs,
        "configs": configs,
        "updates": 520,
        "pairs": 2048,
        "evaluation_seed": 97110001,
        "bootstrap_seed": 97310001,
        "bootstrap_replicates": 10000,
        "primary": "Both seeds calibrated-minus-fixed and calibrated-minus-retained-parent mixture win lower bounds > 0",
        "policies": POLICIES,
        "opponents": OPPONENTS,
        "full_matches": 49152,
        "shared_deals": 4096,
        "initialization": "Original published WP, not the retained adapted parent",
        "endpoint": "Only update520; no intermediate outcome selection",
    }


def train_arm(root, parent, mode):
    plan = read(root / "protocol.json")
    checked(plan["inputs"])
    folder = root / f"seed{parent}_{mode}"
    folder.mkdir(exist_ok=False)
    config = KoiTemperatureConfig(**plan["configs"][f"seed{parent}_{mode}"])
    trainer = KoiTemperatureTrainer(config)
    trainer.save(folder / "initial.pt")
    write(folder / "config.json", asdict(config))
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for update in range(1, plan["updates"] + 1):
            metrics = trainer.update()
            if not all(
                np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))
            ):
                raise FloatingPointError("Nonfinite learning metrics")
            row = {
                "update": update,
                "metrics": metrics,
                "counters": trainer.counters.copy(),
                "seconds": time.perf_counter() - start,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if update == 20 or update % 100 == 0 or update == plan["updates"]:
                trainer.save(folder / f"update_{update:08d}.pt")
            if update % 20 == 0:
                print(
                    json.dumps({"parent": parent, "mode": mode, "update": update}),
                    flush=True,
                )
    trainer.save(folder / "final.pt")
    checked(plan["inputs"])
    write(
        folder / "complete.json",
        {
            "status": "complete",
            "counters": trainer.counters,
            "checkpoint_sha256": sha(folder / "final.pt"),
        },
    )


def play_match(candidate, opponent, case):
    env = ReferenceEnv(case["seed"], "paper8")
    actions, rounds = [], 0
    while env.state[PHASE] != TERMINAL:
        actor = candidate if env.state[PLAYER] == case["seat"] else opponent
        action = int(actor.act(env))
        if not env.legal()[action]:
            raise ValueError("Illegal policy action")
        result = env.step(action)
        assert int(result[0]) != 99
        rounds += int(result[1])
        actions.append(action)
    points = int(env.state[POINTS] - env.state[POINTS + 1]) * (1 - 2 * case["seat"])
    return {
        **case,
        "points": points,
        "actions": actions,
        "endpoint": env.state.tolist(),
        "work": {"decisions": len(actions), "round_endings": rounds, "full_matches": 1},
    }


def evaluate_cell(
    root, parent, policy, opponent_name, checkpoint, pairs=2048, seed_base=97110001
):
    torch.set_num_threads(2)
    candidate = (
        NeuralAgent(checkpoint, device="cpu", deterministic=True)
        if policy == "parent"
        else KoiTemperatureAgent(checkpoint)
    )
    opponent = EndgameAgent(PublishedAgent(opponent_name))
    digest = sha(checkpoint)
    folder = root / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"parent{parent}_{policy}_{opponent_name}"
    totals = dict.fromkeys(("decisions", "round_endings", "full_matches"), 0)
    with (folder / (stem + ".jsonl")).open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for index, case in enumerate(cases(parent, opponent_name, pairs, seed_base), 1):
            row = play_match(candidate, opponent, case)
            stream.write(json.dumps(row) + "\n")
            for key, value in row["work"].items():
                totals[key] += value
            if index % 512 == 0:
                stream.flush()
                print(json.dumps({"cell": stem, "matches": index}), flush=True)
    assert sha(checkpoint) == digest
    write(
        folder / (stem + ".json"),
        {
            "status": "complete",
            "parent": parent,
            "policy": policy,
            "opponent": opponent_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "work": totals,
            "rows_sha256": sha(folder / (stem + ".jsonl")),
        },
    )
    return stem


def run(root):
    if root.exists():
        raise ValueError("Use a fresh study directory")
    plan = make_plan()
    root.mkdir(parents=True, exist_ok=False)
    write(root / "protocol.json", plan)
    try:
        for parent in (1, 2):
            for mode in ("fixed", "calibrated"):
                command = [
                    sys.executable,
                    "-m",
                    "scripts.run_koi_temperature_study",
                    "--output",
                    str(root),
                    "--train-arm",
                    mode,
                    "--parent",
                    str(parent),
                ]
                child = subprocess.Popen(command)
                write(
                    root / "process.json",
                    {
                        "status": "training",
                        "parent": parent,
                        "arm": mode,
                        "pid": child.pid,
                    },
                )
                if child.wait() != 0:
                    raise RuntimeError(
                        "Training worker failed; preserve partial outputs"
                    )
        checked(plan["inputs"])
        write(root / "process.json", {"status": "evaluating", "completed_cells": 0})
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            jobs = []
            for parent in (1, 2):
                for policy in POLICIES:
                    checkpoint = (
                        Path(PARENTS[parent - 1][0])
                        if policy == "parent"
                        else root / f"seed{parent}_{policy}/final.pt"
                    )
                    for opponent in OPPONENTS:
                        jobs.append(
                            pool.submit(
                                evaluate_cell,
                                root,
                                parent,
                                policy,
                                opponent,
                                checkpoint,
                                plan["pairs"],
                                plan["evaluation_seed"],
                            )
                        )
            for index, future in enumerate(concurrent.futures.as_completed(jobs), 1):
                stem = future.result()
                write(
                    root / "process.json",
                    {
                        "status": "evaluating",
                        "completed_cells": index,
                        "last_completed": stem,
                    },
                )
        checked(plan["inputs"])
        write(root / "process.json", {"status": "complete_awaiting_audit"})
    except BaseException as error:
        write(root / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-arm", choices=("fixed", "calibrated"))
    parser.add_argument("--parent", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.train_arm:
        if args.parent is None:
            parser.error("Training worker requires parent index")
        train_arm(args.output, args.parent, args.train_arm)
    else:
        run(args.output)
