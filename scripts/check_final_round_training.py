"""Production-size CUDA capability check; toy-source roots are not strength data."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from scripts.final_round_training import TailConfig, TailTrainer, prefix_to_final_round
from scripts.run_arbitration_study import PARENTS, sha, write


class StopFirst:
    def act(self, env):
        legal = np.flatnonzero(env.legal())
        return 48 if 48 in legal else int(legal[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = TailConfig(
        device="cuda",
        envs=128,
        minibatch=512,
        epochs=3,
        critic_warmup_updates=1,
        seed=8809001,
    )
    states, records = [], []
    env = BatchEnv(1, "cpu", "paper8")
    for i in range(4):
        state, row = prefix_to_final_round(
            {"seed": 31010001 + i, "seat": i % 2}, StopFirst(), StopFirst(), env
        )
        assert state is not None
        states.append(state)
        records.append(row)
    write(args.output / "source.json", records)
    path, expected = PARENTS[0]
    assert sha(path) == expected
    trainer = TailTrainer.from_parent(
        config,
        path,
        torch.from_numpy(np.stack(states)),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
    )
    trainer.save(args.output / "initial.pt")
    initial = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    metrics = []
    start = time.perf_counter()
    for update in range(3):
        metrics.append(trainer.update())
        if update == 0:
            assert (
                trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                == initial
            )
        print(
            json.dumps(
                {
                    "update": update + 1,
                    "counters": trainer.counters,
                    "metrics": metrics[-1],
                }
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - start
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] != initial
    assert trainer.counters["conditional_tail_completions"] == 384
    assert trainer.counters["matches"] == 0
    for name, actor in trainer.opponents.items():
        assert (
            actor.checkpoint()["current"]["parameters_sha256"]
            == trainer.provenance["tail_opponents"][name]
        )
    optimizer_steps = []
    for group in trainer.optimizer.param_groups:
        counts = []
        for parameter in group["params"]:
            value = trainer.optimizer.state[parameter]
            assert all(bool(torch.isfinite(v).all()) for v in value.values())
            counts.append(int(value["step"]))
        optimizer_steps.append(counts)
    trainer.save(args.output / "final.pt")
    inputs = [
        Path(__file__),
        Path("scripts/final_round_training.py"),
        Path("scripts/fixed_microbatch_continuation.py"),
        Path("tests/test_final_round_training.py"),
        Path("runs/final_round_training_tests.xml"),
        Path(path),
        args.output / "initial.pt",
        args.output / "final.pt",
        args.output / "source.json",
    ]
    write(
        args.output / "audit.json",
        {
            "status": "passed",
            "inputs": {str(p): sha(p) for p in inputs},
            "config": asdict(config),
            "counters": trainer.counters,
            "metrics": metrics,
            "optimizer_steps": optimizer_steps,
            "seconds": elapsed,
            "provenance": trainer.provenance,
            "scope": "Three-update production-shape capability; four toy-policy natural roots; no strength claim.",
        },
    )


if __name__ == "__main__":
    main()
