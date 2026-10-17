"""Measure simulator and optional policy throughput after warmup."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .env import BatchEnv
from .estimators import native_gae
from .model import PolicyValue, sample_action
from .train import Config, Trainer, provenance, signed_gae


def benchmark_training(config, steps, frozen_opponent=None):
    if config.published_opponent and frozen_opponent is not None:
        raise ValueError("Published opponent cannot also use --frozen-opponent")
    if bool(config.opponent_fraction) != (
        frozen_opponent is not None or bool(config.published_opponent)
    ):
        raise ValueError(
            "Positive opponent_fraction and --frozen-opponent are required together"
        )
    trainer = Trainer(config)
    if frozen_opponent is not None:
        trainer.set_frozen_opponent(frozen_opponent)
    for _ in range(5):
        trainer.update()
    if trainer.env.device.type == "cuda":
        torch.cuda.synchronize(trainer.env.device)
    initial = trainer.counters.copy()
    start = time.perf_counter()
    for _ in range(steps):
        trainer.update()
    if trainer.env.device.type == "cuda":
        torch.cuda.synchronize(trainer.env.device)
    elapsed = time.perf_counter() - start
    counts = {key: value - initial[key] for key, value in trainer.counters.items()}
    counts.setdefault("learner_decisions", counts["decisions"])
    return {
        "kind": "mixed_opponent_learning"
        if config.opponent_fraction
        else "self_play_learning",
        "includes_learning": True,
        "warmup_updates": 5,
        "warmup_counters": initial,
        "seconds": elapsed,
        **counts,
        "config": asdict(config),
        "decisions_per_second": counts["decisions"] / elapsed,
        "learner_decisions_per_second": counts["learner_decisions"] / elapsed,
        "matches_per_second": counts["matches"] / elapsed,
        "provenance": trainer.provenance,
    }


def benchmark_returns(env, horizon, repeats):
    torch.manual_seed(73)
    device, n = env.device, env.count
    shape = horizon, n
    inputs = (
        torch.randn(shape, device=device),
        torch.randn(shape, device=device),
        torch.rand(shape, device=device) < 0.01,
        torch.randint(2, shape, device=device, dtype=torch.int32),
        torch.randn(n, device=device),
        torch.randint(2, (n,), device=device, dtype=torch.int32),
        0.95,
    )
    functions = {
        "torch": lambda: signed_gae(*inputs),
        "native": lambda: native_gae(env.backend, *inputs),
    }
    outputs = {name: function() for name, function in functions.items()}
    for reference, actual in zip(outputs["torch"], outputs["native"]):
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)
    timings = {}
    for name, function in functions.items():
        for _ in range(5):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[name + "_mean_ms"] = (time.perf_counter() - start) * 1000 / repeats
    return {
        "kind": "returns_only",
        "envs": n,
        "horizon": horizon,
        "repeats": repeats,
        "device": str(device),
        "includes_learning": False,
        "bit_identical": True,
        **timings,
        "provenance": provenance(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--policy", action="store_true")
    p.add_argument(
        "--returns",
        action="store_true",
        help="Compare native and PyTorch signed GAE only",
    )
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument(
        "--training-config",
        type=Path,
        help="Benchmark full learning from this config instead of simulation",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable captured minibatches for --training-config",
    )
    p.add_argument(
        "--cuda-rollout",
        action="store_true",
        help="Enable captured rollout/returns for --training-config",
    )
    p.add_argument(
        "--frozen-opponent",
        type=Path,
        help="Include fixed-opponent inference for --training-config",
    )
    p.add_argument("--profile", default="research12")
    args = p.parse_args()
    if min(args.steps, args.horizon, args.envs) < 1:
        p.error("Benchmark sizes must be positive")
    if sum((args.returns, args.policy, args.training_config is not None)) > 1:
        p.error("Select returns, policy, or full training throughput")
    if args.cuda_graph and not args.training_config:
        p.error("--cuda-graph requires --training-config")
    if args.cuda_rollout and not args.training_config:
        p.error("--cuda-rollout requires --training-config")
    if args.frozen_opponent and not args.training_config:
        p.error("--frozen-opponent requires --training-config")
    if args.training_config:
        config = json.loads(args.training_config.read_text(encoding="utf-8"))
        if args.cuda_graph:
            config["cuda_graph"] = True
        if args.cuda_rollout:
            config["cuda_rollout"] = True
        try:
            result = benchmark_training(
                Config(**config), args.steps, args.frozen_opponent
            )
        except ValueError as error:
            p.error(str(error))
        print(json.dumps(result, indent=2))
        return
    torch.set_num_threads(4)
    env = BatchEnv(args.envs, args.device, args.profile)
    if args.returns:
        print(json.dumps(benchmark_returns(env, args.horizon, args.steps), indent=2))
        return
    model = PolicyValue().to(args.device).eval() if args.policy else None
    counts = torch.zeros(2, dtype=torch.int64, device=args.device)
    with torch.no_grad():
        for t in range(args.steps + 100):
            if t == 100:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
            obs, mask = env.observe()
            logits = (
                model(obs, mask)[0]
                if model
                else torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                    ~mask, -1e9
                )
            )
            actions, _ = sample_action(logits)
            result = env.step(actions)
            if t >= 100:
                counts += result[:, 1:3].sum(0)
            env.reset_done()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rounds, matches = counts.cpu().tolist()
    print(
        json.dumps(
            {
                "kind": "policy_inference" if model else "random_policy",
                "profile": args.profile,
                "envs": args.envs,
                "seconds": elapsed,
                "decisions": args.steps * args.envs,
                "rounds": rounds,
                "matches": matches,
                "decisions_per_second": args.steps * args.envs / elapsed,
                "matches_per_second": matches / elapsed,
                "includes_learning": False,
                "provenance": provenance(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
