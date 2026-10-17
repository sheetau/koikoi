"""Fixed paired-rollout regression study queued after the arbitration PPO study."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.evaluate import duel, paired_interval
from koikoi.train import provenance
from scripts.run_arbitration_study import PARENTS, agent, read, sha, write
from scripts.train_paired_arbitration import PairedRolloutAgent, collect, fit

PREVIOUS = Path("runs/arbitration_study")
SOURCES = [
    "scripts/run_paired_arbitration_study.py",
    "scripts/train_paired_arbitration.py",
    "scripts/paired_rollout_targets.py",
    "scripts/run_arbitration_study.py",
]


def protocol(base):
    parents = []
    for path, expected in PARENTS:
        if sha(path) != expected:
            raise ValueError("Parent identity changed")
        parents.append({"path": path, "sha256": expected})
    cases = {}
    for index in (1, 2):
        for context in ("full", "no_match"):
            name = f"parent{index}_{context}"
            cases[name] = {
                "path": str(base / name / "final.pt"),
                "kind": "regression",
                "parent": parents[index - 1],
                "context": context,
            }
        for arm, label in (("full", "ppo_gate"), ("ppo", "ppo_actor")):
            cases[f"parent{index}_{label}"] = {
                "path": str(PREVIOUS / f"parent{index}_{arm}" / "final.pt"),
                "kind": "previous",
                "parent": parents[index - 1],
            }
        cases[f"parent{index}"] = {
            "path": parents[index - 1]["path"],
            "kind": "parent",
            "parent": parents[index - 1],
        }
    return {
        "source": provenance(),
        "scripts": {path: sha(path) for path in SOURCES},
        "previous_protocol_sha256": sha(PREVIOUS / "protocol.json"),
        "parents": parents,
        "cases": cases,
        "collection": {
            "roots_per_parent": 65536,
            "root_batch": 256,
            "seeds": [19210001, 19220001],
            "device": "cuda",
        },
        "fit": {
            "contexts": ["full", "no_match"],
            "seeds": [19310001, 19320001],
            "epochs": 20,
            "lr": 0.0003,
            "minibatch": 1024,
            "weight_decay": 0.0001,
            "validation": "whole source matches, fixed hash modulo5; final epoch always selected",
        },
        "evaluation": {
            "first_seed": 22000000,
            "pairs": 1000,
            "device": "cpu",
            "threads": 2,
            "opponents": ["rl-wp-assisted", "rl-point-assisted"],
            "profile": "paper8",
            "greedy": True,
            "public_endgame": True,
        },
        "criteria": "Development study. Compare full regression with unchanged parent, PPO gate and continued PPO on identical pairs, separately for each parent and both public opponents. Compare full versus no_match to test the incremental contribution of explicit controller match features; expert proposals still encode match context. Direct regression-versus-parent duels are additional evidence. Retain all endpoints and results. No best-seed selection; parent1 remains canonical. A replicated win-gain claim requires positive paired95% CI lower bounds at both parents for the specified opponent, reported separately by opponent. Do not infer general superiority from selected development wins.",
        "work_accounting": "Count source decisions/matches separately from paired branch decisions/endpoints. The two regression contexts share one dataset per parent; do not count dataset generation twice. Report regression work separately from the PPO budget and inherited expert training. Equal root counts are not equal simulator-decision counts.",
        "stop_rule": "Exactly65536 roots per parent and20 fitting epochs per context; final endpoints only,1000 duplicate pairs per matchup. Any later confirmation uses a new protocol and fresh deals. Abort on source drift or worker failure. GPU work begins only after all six previous training jobs finish; evaluations begin only after the previous study is complete.",
    }


def checked(base):
    p = read(base / "protocol.json")
    if p["source"]["source_sha256"] != provenance()["source_sha256"]:
        raise ValueError("Production source changed")
    if any(sha(path) != digest for path, digest in p["scripts"].items()):
        raise ValueError("Study script changed")
    if sha(PREVIOUS / "protocol.json") != p["previous_protocol_sha256"]:
        raise ValueError("Previous protocol changed")
    if any(sha(parent["path"]) != parent["sha256"] for parent in p["parents"]):
        raise ValueError("Parent checkpoint changed")
    return p


def wait_previous(base, training_only):
    p = checked(base)
    old = read(PREVIOUS / "protocol.json")
    while True:
        status = read(PREVIOUS / "process.json")["status"]
        if status == "failed":
            raise RuntimeError("Previous study failed")
        ready = all(
            (PREVIOUS / case / "training_complete.json").exists()
            for case in old["cases"]
        )
        if ready and (training_only or status == "complete"):
            break
        time.sleep(10)
    for name, case in p["cases"].items():
        if case["kind"] == "previous":
            receipt = read(Path(case["path"]).parent / "training_complete.json")
            if sha(case["path"]) != receipt["checkpoint_sha256"]:
                raise ValueError("Previous endpoint changed: " + name)


def train_parent(base, index):
    p = checked(base)
    folder = base / f"parent{index}"
    dataset = folder / "targets.pt"
    collect(
        p["parents"][index - 1]["path"],
        dataset,
        p["collection"]["seeds"][index - 1],
        p["collection"]["roots_per_parent"],
        p["collection"]["root_batch"],
        "cuda",
    )
    for context in p["fit"]["contexts"]:
        case = f"parent{index}_{context}"
        target = base / case / "final.pt"
        payload = fit(
            dataset,
            target,
            context,
            p["fit"]["seeds"][index - 1],
            "cuda",
            p["fit"]["epochs"],
        )
        write(
            target.parent / "training_complete.json",
            {
                "checkpoint_sha256": sha(target),
                "dataset_sha256": sha(dataset),
                "counters": payload["counters"],
                "history": payload["history"],
            },
        )


def evaluate(base, case, opponent):
    p = checked(base)
    torch.set_num_threads(2)
    spec = p["cases"][case]
    if sha(spec["path"]) != read(base / "endpoint_identities.json")[case]:
        raise ValueError("Frozen evaluation endpoint changed")
    other = spec["parent"]["path"] if opponent == "parent" else opponent
    if spec["kind"] == "regression":
        a = PairedRolloutAgent(spec["path"])
    else:
        a = agent(spec["path"])
    b = agent(other)
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (opponent + ".json")
    records = []
    start = time.perf_counter()
    with target.with_suffix(".jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for seed in range(
            p["evaluation"]["first_seed"],
            p["evaluation"]["first_seed"] + p["evaluation"]["pairs"],
        ):
            left, right = duel(a, b, seed, "paper8"), duel(b, a, seed, "paper8")
            row = {
                "seed": seed,
                "seat0_score": left[0],
                "seat1_score": 1 - right[0],
                "seat0_point_difference": left[1],
                "seat1_point_difference": -right[1],
                "decisions": left[3] + right[3],
            }
            records.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(records) % 100 == 0:
                print(
                    case,
                    opponent,
                    len(records),
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    scores = [(r["seat0_score"] + r["seat1_score"]) / 2 for r in records]
    points = [
        (r["seat0_point_difference"] + r["seat1_point_difference"]) / 2 for r in records
    ]
    write(
        target,
        {
            "case": case,
            "candidate": spec["path"],
            "checkpoint_sha256": sha(spec["path"]),
            "opponent": other,
            "pairs": len(records),
            "matches": 2 * len(records),
            "win_score": float(np.mean(scores)),
            "win_ci95": paired_interval(scores),
            "point_difference": float(np.mean(points)),
            "points_ci95": paired_interval(points),
            "decisions": sum(r["decisions"] for r in records),
            "raw_sha256": sha(target.with_suffix(".jsonl")),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def child(base, mode, case, opponent=None):
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scripts.run_paired_arbitration_study",
        "--base",
        str(base),
        "--mode",
        mode,
        "--case",
        case,
    ]
    if opponent:
        command.extend(["--opponent", opponent])
    with (folder / (mode + ("_" + opponent if opponent else "") + ".log")).open(
        "x", encoding="utf-8"
    ) as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(base):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    write(base / "protocol.json", protocol(base))
    write(
        base / "process.json",
        {"status": "waiting_for_previous_training", "started": time.time()},
    )
    try:
        wait_previous(base, training_only=True)
        write(base / "process.json", {"status": "training"})
        for index in (1, 2):
            child(base, "train", f"parent{index}")
        write(base / "process.json", {"status": "waiting_for_previous_evaluation"})
        wait_previous(base, training_only=False)
        p = checked(base)
        write(
            base / "endpoint_identities.json",
            {case: sha(spec["path"]) for case, spec in p["cases"].items()},
        )
        write(base / "process.json", {"status": "evaluating"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            for case, spec in p["cases"].items():
                for opponent in p["evaluation"]["opponents"] + (
                    ["parent"] if spec["kind"] == "regression" else []
                ):
                    futures.append(pool.submit(child, base, "evaluate", case, opponent))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except BaseException as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise
    write(base / "process.json", {"status": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/paired_arbitration_study")
    )
    parser.add_argument("--mode", choices=("run", "train", "evaluate"), default="run")
    parser.add_argument("--case")
    parser.add_argument("--opponent")
    args = parser.parse_args()
    if args.mode == "run":
        run(args.base)
    elif args.mode == "train":
        train_parent(args.base, int(args.case[-1]))
    else:
        evaluate(args.base, args.case, args.opponent)


if __name__ == "__main__":
    main()
