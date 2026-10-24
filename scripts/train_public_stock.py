"""Matched-budget self-play development; training counts are not strength evidence."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from koikoi.layout import PLAYER
from koikoi.public_stock import PROFILES
from koikoi.public_stock_native import PublicStockBatch
from koikoi.public_stock_policy import PublicStockPolicy
from koikoi.train import signed_gae


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    model = PublicStockPolicy(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    batch = PublicStockBatch(
        range(args.deal_seed, args.deal_seed + args.envs), args.profile, device
    )
    counters = {"decisions": 0, "rounds": 0, "matches": 0, "updates": 0}
    paths = [
        Path(__file__),
        *map(
            Path,
            [
                "koikoi/public_stock.py",
                "koikoi/public_stock_native.py",
                "koikoi/native/public_stock.h",
                "koikoi/public_stock_policy.py",
                "koikoi/train.py",
                "koikoi/reference.py",
                "koikoi/rule_variants.py",
                "koikoi/native/core.h",
            ],
        ),
    ]
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config = vars(args)
    (out / "config.json").write_text(
        json.dumps({"config": config, "bindings": bindings}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    started = time.perf_counter()
    for update in range(args.updates):
        buffers = [[] for _ in range(8)]
        totals = torch.zeros(2, device=device, dtype=torch.int64)
        with torch.no_grad():
            for _ in range(args.horizon):
                obs, mask = batch.observe()
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
            obs, mask = batch.observe()
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
        obs, masks = obs.reshape(n, 660), masks.reshape(n, 50)
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
                loss = (
                    policy_loss
                    + 0.5 * (value - returns[idx]).square().mean()
                    - args.entropy * dist.entropy().mean()
                )
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
                    "model": model.state_dict(),
                    "config": config,
                    "counters": counters,
                    "seconds": record["seconds"],
                    "bindings": bindings,
                },
                out / f"policy_{update + 1:05d}.pt",
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=301)
    p.add_argument("--deal-seed", type=int, default=810000000)
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam", type=float, default=0.99)
    p.add_argument("--entropy", type=float, default=0.01)
    train(p.parse_args())


if __name__ == "__main__":
    main()
