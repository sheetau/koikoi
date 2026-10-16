"""Measure simulator and optional policy throughput after warmup."""

import argparse
import json
import time

import torch

from .env import BatchEnv
from .estimators import native_gae
from .model import PolicyValue, sample_action
from .train import provenance, signed_gae


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
    p.add_argument("--profile", default="research12")
    args = p.parse_args()
    if min(args.steps, args.horizon, args.envs) < 1:
        p.error("Benchmark sizes must be positive")
    if args.returns and args.policy:
        p.error("Select returns or policy throughput, not both")
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
