"""Continue retained residual weights with fresh Adam; not an optimizer resume.

Isolated from the hash-bound original trainer. Counts here are additional work.
No study is authorized merely by these engineering utilities.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from koikoi.layout import PLAYER
from koikoi.public_stock import PROFILES
from koikoi.public_stock_native import PublicStockBatch
from koikoi.public_stock_residual import load_residual
from koikoi.train import signed_gae
from scripts.export_public_stock_results import read, sha
from scripts.run_arbitration_study import PARENTS


def initialise(args):
    """Require an unchanged retained V2 checkpoint and a fresh optimizer."""
    receipt = "docs/paper/generated/public_stock_v3_evidence.json"
    retained = {Path(p).resolve(): h for p, h in read(receipt)["bindings"].items()}
    path = Path(args.checkpoint)
    digest = sha(path)
    if retained.get(path.resolve()) != digest:
        raise ValueError("Checkpoint is not a retained V3 input")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["config"]["profile"] != args.profile
        or payload["config"]["parent"] != args.parent
        or payload["parent_sha256"] != PARENTS[args.parent][1]
        or payload["counters"]["updates"] != 128
        or payload["counters"]["decisions"] != 4194304
    ):
        raise ValueError("Wrong retained starting policy or training count")
    for name, expected in payload["bindings"].items():
        if sha(name) != expected:
            raise ValueError(f"Retained training source changed: {name}")
    model, _ = load_residual(path, args.device)
    if any(p.requires_grad for p in model.teacher.parameters()):
        raise ValueError("Original teacher must remain frozen")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, eps=1e-5
    )
    lineage = {
        "checkpoint": path.as_posix(), "checkpoint_sha256": digest,
        "receipt": receipt, "receipt_sha256": sha(receipt),
        "prior_counters": payload["counters"],
        "optimizer": "fresh Adam; no optimizer, RNG or environment-state resume",
        "counter_scope": "counters are additional; cumulative_counters include the retained V2 adaptation only",
    }
    return model, payload, optimizer, lineage


def train(args):
    if min(args.updates, args.envs, args.horizon, args.epochs, args.minibatch, args.save_every) < 1:
        raise ValueError("Training counts must be positive")
    if args.envs * args.horizon < 2 or not all(
        math.isfinite(x) for x in (args.lr, args.lam, args.kl, args.entropy)
    ) or args.lr <= 0 or not 0 <= args.lam <= 1 or min(args.kl, args.entropy) < 0:
        raise ValueError("Invalid training batch or optimizer settings")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.device
    model, initial, optimizer, lineage = initialise(args)
    parent, parent_hash = initial["parent"], initial["parent_sha256"]
    batch = PublicStockBatch(
        range(args.deal_seed, args.deal_seed + args.envs), args.profile, device
    )
    counters = {"decisions": 0, "rounds": 0, "matches": 0, "updates": 0}
    paths = [
        Path(__file__),
        Path("scripts/train_public_stock_residual.py"),
        Path("scripts/export_public_stock_results.py"),
        Path("scripts/run_arbitration_study.py"),
        *map(
            Path,
            [
                "koikoi/public_stock.py",
                "koikoi/public_stock_native.py",
                "koikoi/native/public_stock.h",
                "koikoi/public_stock_policy.py",
                "koikoi/public_stock_residual.py",
                "koikoi/train.py",
                "koikoi/reference.py",
                "koikoi/rule_variants.py",
                "koikoi/native/core.h",
            ],
        ),
    ]
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config = vars(args)
    runtime = {
        "torch": torch.__version__, "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    (out / "config.json").write_text(
        json.dumps({"config": config, "bindings": bindings, "lineage": lineage, "runtime": runtime}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    started = time.perf_counter()
    for update in range(args.updates):
        buffers = [[] for _ in range(8)]
        totals = torch.zeros(2, device=device, dtype=torch.int64)
        with torch.no_grad():
            for _ in range(args.horizon):
                obs, mask = model.encode(batch)
                players = batch.state[:, PLAYER].clone()
                logits, value = model(obs, mask)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                result = batch.step(action)
                done = result[:, 2].bool()
                # Pure match-win utility, including draws as zero. No score
                # shaping: amplified stakes cannot increase the reward scale.
                reward = result[:, 0].float() * (1 - 2 * players)
                for buf, tensor in zip(
                    buffers,
                    (
                        obs,
                        mask,
                        action,
                        distribution.log_prob(action),
                        value,
                        reward,
                        done,
                        players,
                    ),
                    strict=True,
                ):
                    buf.append(tensor)
                totals += result[:, 1:3].sum(0)
                batch.reset_done()
            obs, mask = model.encode(batch)
            _, bootstrap = model(obs, mask)
            obs, masks, actions, old_logp, values, rewards, dones, players = map(
                torch.stack, buffers
            )
            advantages, returns = signed_gae(
                rewards,
                values,
                dones,
                players,
                bootstrap,
                batch.state[:, PLAYER],
                args.lam,
            )
        n = args.envs * args.horizon
        obs, masks = obs.reshape(n, 710), masks.reshape(n, 50)
        actions, old_logp, advantages, returns = (
            x.reshape(n) for x in (actions, old_logp, advantages, returns)
        )
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        for _ in range(args.epochs):
            permutation = torch.randperm(n, device=device)
            for idx in permutation.split(args.minibatch):
                logits, value = model(obs[idx], masks[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(actions[idx]) - old_logp[idx]).exp()
                policy_loss = -torch.minimum(
                    ratio * advantages[idx], ratio.clamp(0.8, 1.2) * advantages[idx]
                ).mean()
                base = obs[idx, 660:].masked_fill(~masks[idx], -1e9)
                kl = torch.nn.functional.kl_div(
                    logits.log_softmax(1), base.softmax(1), reduction="batchmean"
                )
                loss = (
                    args.kl * kl
                    + policy_loss
                    + 0.5 * (value - returns[idx]).square().mean()
                    - args.entropy * dist.entropy().mean()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        rounds, matches = totals.cpu().tolist()
        counters["decisions"] += n
        counters["rounds"] += rounds
        counters["matches"] += matches
        counters["updates"] += 1
        if (update + 1) % args.save_every == 0 or update + 1 == args.updates:
            record = dict(
                **counters,
                seconds=time.perf_counter() - started,
                loss=float(loss.detach()),
                profile=args.profile,
                seed=args.seed,
            )
            print(json.dumps(record), flush=True)
            with (out / "progress.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as f:
                f.write(json.dumps(record) + "\n")
            torch.save(
                {
                    "format": "public_stock_residual_v1",
                    "parent": parent,
                    "parent_sha256": parent_hash,
                    "model": model.correction_state(),
                    "config": config,
                    "counters": counters,
                    "seconds": record["seconds"],
                    "bindings": bindings,
                    "runtime": runtime,
                    "training_method": "warm_start_fresh_adam_v1",
                    "lineage": lineage,
                    "cumulative_counters": {
                        k: initial["counters"][k] + v for k, v in counters.items()
                    },
                },
                out / f"policy_add_{update + 1:05d}.pt",
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--parent", type=int, choices=[0, 1], required=True)
    p.add_argument("--kl", type=float, default=0.02)
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--deal-seed", type=int, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=0.99)
    p.add_argument("--entropy", type=float, default=0.01)
    train(p.parse_args())


if __name__ == "__main__":
    main()
