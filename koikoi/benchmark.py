"""Measure simulator and optional policy throughput after warmup."""

import argparse
import json
import time

import torch

from .env import BatchEnv
from .model import PolicyValue, sample_action
from .train import provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--policy", action="store_true")
    p.add_argument("--profile", default="research12")
    args = p.parse_args()
    torch.set_num_threads(4)
    env = BatchEnv(args.envs, args.device, args.profile)
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
