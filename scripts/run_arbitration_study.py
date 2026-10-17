"""Fixed-budget development study of public expert arbitration and continued PPO."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import (
    ArbitrationAgent,
    ArbitrationConfig,
    ArbitrationTrainer,
    read_checkpoint,
)
from koikoi.evaluate import duel, make_agent, paired_interval
from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.train import provenance

PARENTS = (
    (
        "runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt",
        "37d06849261e0c1f0c31a9d2f228388fa1f391b670ce6772093a1c276091cc36",
    ),
    (
        "runs/cold_ppo_screen/seed6720001_warm/update_00000520.pt",
        "5e8a123d5fafb6ebae8c1ac8d094c9de65bf364381cc10b78208337051e181cc",
    ),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol():
    parents = []
    for path, expected in PARENTS:
        payload, digest = read_checkpoint(path)
        if digest != expected:
            raise ValueError("Parent checkpoint changed")
        parents.append(
            {"path": path, "sha256": digest, "counters": payload["counters"]}
        )
    cases = {}
    for index, seed in enumerate((19010001, 19020001)):
        for arm in ("full", "phase", "ppo"):
            config = (
                PublishedConfig(seed=seed)
                if arm == "ppo"
                else ArbitrationConfig(seed=seed, context=arm)
            )
            cases[f"parent{index + 1}_{arm}"] = {
                "arm": arm,
                "parent": parents[index],
                "config": asdict(config),
            }
    return {
        "source": provenance(),
        "runner_sha256": sha(__file__),
        "cases": cases,
        "parents": parents,
        "updates": 520,
        "decisions_per_case": 8519680,
        "evaluation": {
            "first_seed": 21000000,
            "pairs": 1000,
            "device": "cpu",
            "threads": 2,
            "profile": "paper8",
            "greedy": True,
            "endgame": True,
            "opponents": ["rl-wp-assisted", "rl-point-assisted", "parent"],
        },
        "selection": "Development only. Evaluate final update520 of every arm. Compare full minus phase and full minus continued PPO on identical pairs, separately for both parents and both public opponents. No intermediate selection. Stronger-than-parent evidence requires positive paired win-score CI against both public opponents for both parents; direct parent duels are additional evidence. Point differences and disagreements are secondary. Canonical parent is always parent1. Select no new deployment model on inconclusive results.",
        "stop_rule": "Six runs, exactly520 updates each, including20 critic-only updates. Evaluate1000 duplicate pairs for every endpoint/opponent plus both parents against the two public opponents. Retain every result. Any later experiment uses a new protocol and fresh seeds. No final-test seeds are used.",
        "budget": "Additional training only; inherited parent training and original RL-Point training are separate dependencies. Same simulator-decision budget, not equal FLOPs or trainable parameter counts.",
    }


def checked_protocol(base):
    p = read(base / "protocol.json")
    if (
        p["runner_sha256"] != sha(__file__)
        or p["source"]["source_sha256"] != provenance()["source_sha256"]
    ):
        raise ValueError("Study source changed")
    for parent in p["parents"]:
        if sha(parent["path"]) != parent["sha256"]:
            raise ValueError("Parent changed")
    return p


def train(base, case):
    p = checked_protocol(base)
    spec = p["cases"][case]
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.jsonl").exists():
        raise ValueError("Existing training run; do not overwrite")
    if spec["arm"] == "ppo":
        payload, _ = read_checkpoint(spec["parent"]["path"])
        trainer = PublishedTrainer(PublishedConfig(**spec["config"]), payload["actor"])
        trainer.model.value.load_state_dict(payload["value"])
        trainer.provenance["weight_initialization"] = spec["parent"]
    else:
        trainer = ArbitrationTrainer.from_parent(
            ArbitrationConfig(**spec["config"]), spec["parent"]["path"]
        )
    write(folder / "config.json", spec["config"])
    trainer.save(folder / "initial.pt")
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        while trainer.counters["updates"] < p["updates"]:
            metrics = (
                trainer.update()
                | trainer.counters
                | {"elapsed_seconds": time.perf_counter() - start}
            )
            stream.write(json.dumps(metrics) + "\n")
            stream.flush()
            if trainer.counters["updates"] % 20 == 0:
                print(json.dumps(metrics), flush=True)
                trainer.save(folder / "latest.pt")
        trainer.save(folder / "final.pt")
    if trainer.counters["decisions"] != p["decisions_per_case"]:
        raise ValueError("Training accounting mismatch")
    write(
        folder / "training_complete.json",
        {
            "checkpoint_sha256": sha(folder / "final.pt"),
            "counters": trainer.counters,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def agent(path):
    if path.startswith("rl-"):
        return make_agent(path, 0, True, "cpu")
    payload, _ = read_checkpoint(path)
    if payload.get("format") == "arbitration_v1":
        return ArbitrationAgent(path)
    return make_agent(path, 0, True, "cpu")


def evaluate(base, case, opponent):
    p = checked_protocol(base)
    torch.set_num_threads(p["evaluation"]["threads"])
    if case in p["cases"]:
        spec = p["cases"][case]
        candidate = str(base / case / "final.pt")
        if (
            sha(candidate)
            != read(base / case / "training_complete.json")["checkpoint_sha256"]
        ):
            raise ValueError("Endpoint changed")
        parent = spec["parent"]["path"]
    else:
        parent = p["parents"][int(case[-1]) - 1]["path"]
        candidate = parent
    other = parent if opponent == "parent" else opponent
    a, b = agent(candidate), agent(other)
    target = base / case / (opponent + ".json")
    raw = target.with_suffix(".jsonl")
    records = []
    start = time.perf_counter()
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for seed in range(
            p["evaluation"]["first_seed"],
            p["evaluation"]["first_seed"] + p["evaluation"]["pairs"],
        ):
            left = duel(a, b, seed, "paper8")
            right = duel(b, a, seed, "paper8")
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
            "candidate": candidate,
            "checkpoint_sha256": sha(candidate),
            "opponent": other,
            "pairs": len(records),
            "matches": 2 * len(records),
            "win_score": float(np.mean(scores)),
            "win_ci95": paired_interval(scores),
            "point_difference": float(np.mean(points)),
            "points_ci95": paired_interval(points),
            "decisions": sum(r["decisions"] for r in records),
            "raw_sha256": sha(raw),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )


def subprocess_job(base, mode, case, opponent=None):
    folder = base / case
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scripts.run_arbitration_study",
        "--base",
        str(base),
        "--mode",
        mode,
        "--case",
        case,
    ]
    if opponent:
        command += ["--opponent", opponent]
    with (folder / (mode + ("_" + opponent if opponent else "") + ".log")).open(
        "x", encoding="utf-8"
    ) as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(base):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    p = protocol()
    write(base / "protocol.json", p)
    write(base / "process.json", {"status": "running", "started": time.time()})
    futures = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for index in (1, 2):
                for opponent in p["evaluation"]["opponents"][:2]:
                    futures.append(
                        pool.submit(
                            subprocess_job, base, "evaluate", f"parent{index}", opponent
                        )
                    )
            for case in p["cases"]:
                for future in futures:
                    if future.done():
                        future.result()
                subprocess_job(base, "train", case)
                for opponent in p["evaluation"]["opponents"]:
                    futures.append(
                        pool.submit(subprocess_job, base, "evaluate", case, opponent)
                    )
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except BaseException as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise
    write(base / "process.json", {"status": "complete", "finished": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/arbitration_study"))
    parser.add_argument("--mode", choices=("run", "train", "evaluate"), default="run")
    parser.add_argument("--case")
    parser.add_argument("--opponent")
    args = parser.parse_args()
    if args.mode == "run":
        run(args.base)
    elif args.mode == "train":
        train(args.base, args.case)
    else:
        evaluate(args.base, args.case, args.opponent)


if __name__ == "__main__":
    main()
