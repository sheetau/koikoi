"""Bounded full-learning feasibility benchmark; never a strength experiment."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.published_train import PublishedConfig, PublishedTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/published_ppo_benchmark")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = PublishedConfig(seed=6680001, critic_warmup_updates=3)
    start = time.perf_counter()
    trainer = PublishedTrainer(config)
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - start
    initial_digest = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    trainer.save(args.output / "initial.pt")
    rows = []
    for _ in range(config.critic_warmup_updates):
        rows.append(trainer.update() | trainer.counters.copy())
        print(json.dumps(rows[-1]), flush=True)
    assert (
        trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        == initial_digest
    )
    warmup_counts = trainer.counters.copy()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(5):
        rows.append(trainer.update() | trainer.counters.copy())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()
    reserved_bytes = torch.cuda.max_memory_reserved()
    timed_counts = {k: v - warmup_counts[k] for k, v in trainer.counters.items()}
    assert timed_counts["policy_updates"] == 5 and timed_counts["warmup_updates"] == 0
    assert timed_counts["decisions"] == 5 * config.envs * config.horizon
    current_digest = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    assert current_digest != initial_digest
    trainer.save(args.output / "latest.pt")
    report = {
        "status": "complete",
        "purpose": "full-learning feasibility, not strength or scale-gate training",
        "timing_scope": "five full updates including collection, GAE, critic and actor Adam; excludes initialization, three critic-only warmups and checkpoint IO",
        "memory_scope": "PyTorch CUDA allocator, not entire device/driver memory",
        "config": asdict(config),
        "provenance": trainer.provenance,
        "runtime": trainer.runtime,
        "initialization_seconds": initialization_seconds,
        "critic_only_counts": warmup_counts,
        "timed_counts": timed_counts,
        "total_counts": trainer.counters,
        "elapsed_seconds": elapsed,
        "matches_per_second": timed_counts["matches"] / elapsed,
        "decisions_per_second": timed_counts["decisions"] / elapsed,
        "baseline_allocated_bytes": baseline_bytes,
        "peak_allocated_bytes": peak_bytes,
        "peak_reserved_bytes": reserved_bytes,
        "initial_parameters_sha256": initial_digest,
        "current_parameters_sha256": current_digest,
        "metrics": rows,
        "files_sha256": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("initial.pt", "latest.pt")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("metrics", "config", "provenance")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
