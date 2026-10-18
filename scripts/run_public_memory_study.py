"""Frozen public-memory/masked-control study on both adapted parents."""

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent
from scripts.public_memory_match import play_memory_match
from scripts.public_memory_tail import (
    MemoryTailAgent,
    MemoryTailConfig,
    MemoryTailTrainer,
    load_public_memory,
)
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_final_round_training_study import checked_inputs, configure

BASE = Path("runs/final_round_training_study")
DIAGNOSTIC = Path("runs/revealed_opponent_identification")
POLICIES = ("parent", "public", "masked")


def cases(parent, opponent):
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "seat": seat,
            "replicate": rep,
            "seed": 91110001 + 10000 * (opponent == "point") + rep,
        }
        for seat in (0, 1)
        for rep in range(1024)
    ]


def make_plan():
    inputs = {}
    for path in (
        BASE / "independent_audit.json",
        DIAGNOSTIC / "independent_audit.json",
        Path("runs/public_memory_tail_capability/audit.json"),
    ):
        audit = read(path)
        assert audit["status"] == "passed"
        for filename, digest in audit["inputs"].items():
            assert sha(filename) == digest, filename
            inputs[filename] = digest
        inputs[str(path)] = sha(path)
    files = [Path(__file__)] + [
        Path(p)
        for p in (
            "scripts/audit_public_memory_study.py",
            "scripts/run_final_round_training_study.py",
            "scripts/audit_final_round_training_study.py",
            "scripts/audit_revealed_opponent_identification.py",
            "scripts/run_arbitration_study.py",
            "docs/PUBLIC_MEMORY_TAIL.rst",
            "docs/PUBLIC_MEMORY_STUDY.rst",
            "tests/test_public_memory_protocol.py",
            "runs/public_memory_protocol_tests.xml",
        )
    ]
    parents = []
    for index, (path, digest) in enumerate(PARENTS, 1):
        assert sha(path) == digest
        files.append(Path(path))
        files.append(BASE / f"parent{index}/bank.pt")
        parents.append(
            {
                "index": index,
                "path": path,
                "sha256": digest,
                "seed": 90110001 + 10000 * (index - 1),
            }
        )
    inputs.update({str(p): sha(p) for p in files})
    return {
        "kind": "public_memory_tail_v1",
        "inputs": inputs,
        "core_source_sha256": provenance()["source_sha256"],
        "parents": parents,
        "updates": 520,
        "pairs_per_opponent": 1024,
        "bootstrap_seed": 91310001,
        "bootstrap_replicates": 10000,
        "selection": "Only update520; all four balanced-mixture public-minus-masked/parent lower bounds positive.",
    }


def train_arm(root, parent, mode, plan):
    from scripts.audit_public_memory_study import audit_endpoint

    folder = root / f"parent{parent['index']}_{mode}"
    bank = torch.load(
        BASE / f"parent{parent['index']}/bank.pt",
        weights_only=False,
        map_location="cpu",
    )
    memory = load_public_memory(BASE, DIAGNOSTIC, parent["index"])
    config = MemoryTailConfig(
        device="cuda", seed=parent["seed"], public_memory=mode == "public"
    )
    trainer = MemoryTailTrainer.from_parent(config, parent["path"], *bank, memory)
    trainer.save(folder / "initial.pt")
    write(folder / "config.json", asdict(config))
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for update in range(1, 521):
            metrics = trainer.update()
            row = {
                "update": update,
                "seconds": time.perf_counter() - start,
                "counters": trainer.counters.copy(),
                "metrics": metrics,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if update == 20 or update % 100 == 0 or update == 520:
                trainer.save(folder / f"update_{update:08d}.pt")
            if update % 20 == 0:
                print(
                    json.dumps({"parent": parent["index"], "mode": mode, **row}),
                    flush=True,
                )
    trainer.save(folder / "final.pt")
    del trainer
    checked_inputs(plan)
    write(folder / "training_audit.json", audit_endpoint(root, parent["index"], mode))


def evaluate_cell(root_name, parent, opponent_name):
    configure()
    root = Path(root_name)
    plan = read(root / "protocol.json")
    checked_inputs(plan)
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    predictors = {n: StrictPublishedPrediction(a) for n, a in models.items()}
    opponent = EndgameAgent(models[opponent_name])
    actors = {
        "parent": NeuralAgent(parent["path"], deterministic=True),
        **{
            mode: MemoryTailAgent(
                root / f"parent{parent['index']}_{mode}/final.pt", predictors
            )
            for mode in ("public", "masked")
        },
    }
    native = BatchEnv(1, "cpu", "paper8")
    path = root / f"eval_parent{parent['index']}_{opponent_name}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for policy, actor in actors.items():
            for case in cases(parent["index"], opponent_name):
                row = play_memory_match(case, actor, opponent, native)
                stream.write(json.dumps({"policy": policy, **row}) + "\n")
                stream.flush()
    return str(path)


def stratified_interval(values, seed=91310001, draws=10000):
    """Input[stratum, paired deal]; fixed equal weighting of strata."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("Expected strata by paired deals")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, 128):
        count = min(128, draws - start)
        indices = rng.integers(
            0, values.shape[1], (count, values.shape[0], values.shape[1])
        )
        means.extend(
            values[np.arange(values.shape[0])[None, :, None], indices].mean(axis=(1, 2))
        )
    return np.quantile(means, [0.025, 0.975]).tolist()


def contrasts(root):
    results = []
    for parent in (1, 2):
        points = np.zeros((3, 2, 1024, 2))
        for opponent_index, name in enumerate(("wp", "point")):
            seen = set()
            with (root / f"eval_parent{parent}_{name}.jsonl").open(
                encoding="utf-8"
            ) as stream:
                for line in stream:
                    row = json.loads(line)
                    i, j, k = (
                        POLICIES.index(row["policy"]),
                        row["replicate"],
                        row["seat"],
                    )
                    assert (i, j, k) not in seen
                    seen.add((i, j, k))
                    points[i, opponent_index, j, k] = (
                        row["terminal_points"][k] - row["terminal_points"][1 - k]
                    )
            assert len(seen) == 6144
        for comparison, candidate, baseline in (
            ("public-minus-masked", 1, 2),
            ("public-minus-parent", 1, 0),
            ("masked-minus-parent", 2, 0),
        ):
            for scope, indices in (("mixture", [0, 1]), ("wp", [0]), ("point", [1])):
                record = {"parent": parent, "comparison": comparison, "scope": scope}
                for metric, outcomes in (
                    ("win", (np.sign(points) + 1) / 2),
                    ("point", points),
                ):
                    changes = (
                        outcomes[candidate, indices] - outcomes[baseline, indices]
                    ).mean(axis=2)
                    record[metric] = {
                        "change": float(changes.mean()),
                        "interval95": stratified_interval(changes),
                        "candidate_mean": float(outcomes[candidate, indices].mean()),
                        "baseline_mean": float(outcomes[baseline, indices].mean()),
                    }
                results.append(record)
    return results


def primary_success(results):
    primary = [
        r
        for r in results
        if r["scope"] == "mixture" and r["comparison"] != "masked-minus-parent"
    ]
    assert len(primary) == 4
    return all(r["win"]["interval95"][0] > 0 for r in primary)


def main():
    from scripts.audit_public_memory_study import audit_matched_warmup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    plan = make_plan()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", plan)
    try:
        for parent in plan["parents"]:
            for mode in ("public", "masked"):
                write(
                    args.output / "process.json",
                    {
                        "status": "running",
                        "stage": "training",
                        "parent": parent["index"],
                        "mode": mode,
                    },
                )
                train_arm(args.output, parent, mode, plan)
            audit_matched_warmup(args.output, parent["index"])
        write(
            args.output / "process.json", {"status": "running", "stage": "evaluation"}
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(evaluate_cell, str(args.output), parent, opponent)
                for parent in plan["parents"]
                for opponent in ("wp", "point")
            ]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps({"evaluation_complete": future.result()}), flush=True)
        checked_inputs(plan)
        result = contrasts(args.output)
        files = [
            p
            for p in args.output.rglob("*")
            if p.is_file() and p.name != "process.json"
        ]
        write(
            args.output / "analysis.json",
            {
                "status": "complete",
                "contrasts": result,
                "replicated_gain": primary_success(result),
                "inputs": {str(p): sha(p) for p in files},
                "scope": "Prospective balanced-mixture development comparison; independent audit pending; no promotion.",
            },
        )
        write(args.output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(args.output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
