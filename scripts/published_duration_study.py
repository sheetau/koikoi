"""Fixed short/long-prefix comparison using the unchanged published PPO trainer."""

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.upstream import UPSTREAM_PATH, load_source
from scripts.run_arbitration_study import PARENTS, read, sha, write

SEEDS = (99110001, 99210001)
INITIAL = Path("runs/pretrained_ppo_screen/initial/initial.pt")
INITIAL_SHA = "290e301702428a1762ffec2004ec9e9fafbb35a7d68a79e849c8e47a1126893e"
SEED_SCAN = Path("runs/published_duration_seed_scan.json")


def checked(inputs):
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError(f"Changed study input: {name}")


def bindings():
    load_source()
    assert sha(INITIAL) == INITIAL_SHA
    for path, digest in PARENTS:
        assert sha(path) == digest
    paths = [
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    ]
    paths += list(UPSTREAM_PATH.glob("*.py"))
    paths += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{opponent}.pt"
        for phase in ("discard", "pick", "koikoi")
        for opponent in ("wp", "point")
    ]
    paths += [
        INITIAL,
        Path(__file__),
        Path("docs/PUBLISHED_DURATION_STUDY.rst"),
        Path("scripts/run_arbitration_study.py"),
        Path("scripts/directional_return_statistics.py"),
        Path("scripts/directional_return_probe.py"),
        Path("scripts/audit_pretrained_ppo_screen.py"),
        Path("runs/directional_return_diagnostic/audit.json"),
        Path("runs/directional_return_diagnostic/summary.json"),
        SEED_SCAN,
    ]
    paths += [Path(p) for p, _ in PARENTS]
    return {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in paths}


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def capability(root):
    scan = read(SEED_SCAN)
    assert not scan["overlaps"] and not scan["unreadable"]
    inputs = bindings()
    root.mkdir(parents=True, exist_ok=False)
    original = torch.load(INITIAL, map_location="cpu", weights_only=False)
    rows = []
    for seed in (99010001, 99020001):
        trainer = PublishedTrainer(PublishedConfig(seed=seed, critic_warmup_updates=1))
        before = copy.deepcopy(trainer.model.actor.checkpoint()["current"]["weights"])
        assert_equal(before, original["actor"]["current"]["weights"])
        assert (trainer.model.value[-1].weight == 0).all() and (
            trainer.model.value[-1].bias == 0
        ).all()
        trainer.update()
        assert_equal(before, trainer.model.actor.checkpoint()["current"]["weights"])
        trainer.update()
        moved = any(
            not torch.equal(before[phase][key], value)
            for phase, state in trainer.model.actor.checkpoint()["current"][
                "weights"
            ].items()
            for key, value in state.items()
        )
        assert moved
        checkpoint = root / f"seed{seed}_before.pt"
        trainer.save(checkpoint)
        expected_metrics = trainer.update()
        expected = root / f"seed{seed}_expected.pt"
        trainer.save(expected)
        counts = trainer.counters.copy()
        del trainer
        restored = PublishedTrainer.restore(checkpoint)
        observed_metrics = restored.update()
        observed = root / f"seed{seed}_observed.pt"
        restored.save(observed)
        assert_equal(expected_metrics, observed_metrics)
        left = torch.load(expected, map_location="cpu", weights_only=False)
        right = torch.load(observed, map_location="cpu", weights_only=False)
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "torch_rng",
            "cuda_rng",
            "python_rng",
            "numpy_rng",
            "runtime",
            "config",
        ):
            assert_equal(left[key], right[key])
        rows.append(
            {
                "seed": seed,
                "exact_resume": True,
                "warmup_actor_unchanged": True,
                "actor_moved": moved,
                "unique_counters": counts,
                "repeated_verification_decisions": 16384,
                "snapshots": {str(p): sha(p) for p in (checkpoint, expected, observed)},
            }
        )
        del restored
    checked(inputs)
    write(
        root / "audit.json",
        {
            "status": "passed",
            "inputs": inputs,
            "cases": rows,
            "scope": "Production tensor shapes; shortened one-update warmup. Exact saved next-update check including model, Adam, state and RNG. No strength evaluation.",
        },
    )


def plan(root, capability_root):
    inputs = bindings()
    audit = read(capability_root / "audit.json")
    assert audit["status"] == "passed" and audit["inputs"] == inputs
    assert len(audit["cases"]) == 2 and all(r["exact_resume"] for r in audit["cases"])
    for row in audit["cases"]:
        checked(row["snapshots"])
    inputs[str(capability_root / "audit.json")] = sha(capability_root / "audit.json")
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "protocol.json",
        {
            "inputs": inputs,
            "configs": {
                str(seed): asdict(PublishedConfig(seed=seed)) for seed in SEEDS
            },
            "warmup": 20,
            "short": 520,
            "long": 4020,
            "decisions_per_long_run": 65863680,
            "total_training_decisions": 131727360,
            "evaluation": {
                "pairs_per_opponent": 4096,
                "wp_seed": 99310001,
                "point_seed": 99320001,
                "matches": 98304,
                "shared_initial_deals": 8192,
                "device": "cpu",
                "threads": 2,
                "greedy": True,
                "endgame": True,
                "search": False,
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 99410001,
                "simultaneous_family": 4,
            },
            "primary": "Both seeds: long-short and long-retained-parent equal WP/Point-mixture win simultaneous lower bounds >0",
            "selection": "Only prescribed 520/4020 endpoints. Candidate always seed99110001 long if qualified; final confirmation remains required.",
        },
    )


def run(root):
    if (root / "running.json").exists():
        raise ValueError("Existing execution record; inspect state before any recovery")
    try:
        for seed in SEEDS:
            write(
                root / "running.json",
                {"pid": os.getpid(), "seed": seed, "state": "training"},
            )
            train(root, seed)
    except Exception as exc:
        write(
            root / "failure.json",
            {"pid": os.getpid(), "seed": seed, "error": repr(exc)},
        )
        raise
    write(
        root / "training_complete.json",
        {
            "status": "complete",
            "cases": {
                str(seed): read(root / f"seed{seed}" / "complete.json")
                for seed in SEEDS
            },
            "evaluation": "not started",
        },
    )
    write(root / "running.json", {"pid": os.getpid(), "state": "training complete"})


def train(root, seed):
    protocol = read(root / "protocol.json")
    checked(protocol["inputs"])
    config = PublishedConfig(**protocol["configs"][str(seed)])
    folder = root / f"seed{seed}"
    folder.mkdir(exist_ok=False)
    trainer = PublishedTrainer(config)
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)
    assert_equal(
        trainer.model.actor.checkpoint()["current"]["weights"],
        initial["actor"]["current"]["weights"],
    )
    trainer.save(folder / "initial.pt")
    write(folder / "config.json", asdict(config))
    start = time.monotonic()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for update in range(1, protocol["long"] + 1):
            metrics = trainer.update()
            if not all(
                np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))
            ):
                raise FloatingPointError("Nonfinite training metric")
            assert trainer.counters["updates"] == update
            row = {
                "update": update,
                "metrics": metrics,
                "counters": trainer.counters.copy(),
                "seconds": time.monotonic() - start,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if update in (protocol["warmup"], protocol["short"], protocol["long"]):
                trainer.save(folder / f"update_{update:08d}.pt")
            if update % 100 == 0 or update == protocol["long"]:
                trainer.save(folder / "recovery.pt")
                checked(protocol["inputs"])
            if update % 20 == 0:
                print(
                    json.dumps(
                        {"seed": seed, "update": update, "seconds": row["seconds"]}
                    ),
                    flush=True,
                )
    assert trainer.counters["decisions"] == protocol["decisions_per_long_run"]
    checked(protocol["inputs"])
    write(
        folder / "complete.json",
        {
            "status": "complete",
            "counters": trainer.counters,
            "final_sha256": sha(folder / f"update_{protocol['long']:08d}.pt"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capability", "plan", "train", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capability", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == "capability":
        capability(args.output)
    elif args.mode == "plan":
        if args.capability is None:
            parser.error("--capability required")
        plan(args.output, args.capability)
    elif args.mode == "run":
        run(args.output)
    else:
        if args.seed is None:
            parser.error("--seed required")
        train(args.output, args.seed)


if __name__ == "__main__":
    main()
