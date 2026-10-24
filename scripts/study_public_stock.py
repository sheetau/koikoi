"""Recorded, non-overwriting public-stock development runs."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.public_stock import PROFILES, PublicStockEnv
from koikoi.public_stock_native import PublicStockBackend, PublicStockBatch
from koikoi.public_stock_policy import load_policy
from koikoi.public_stock_search import rollout_actions, view
from koikoi.reference import LIGHTS, RIBBONS, TANE


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def greedy_actions(obs, legal):
    cards = obs[:, :528].reshape(-1, 48, 11)
    weights = torch.ones(48, device=obs.device)
    weights[list(RIBBONS)], weights[list(TANE)], weights[list(LIGHTS)] = 2, 3, 5
    field = cards[:, :, 1].reshape(-1, 12, 4)
    captures = (field * weights.reshape(12, 4)).sum(2).repeat_interleave(4, 1)
    matched = field.sum(2).repeat_interleave(4, 1) > 0
    scores = weights.expand(len(obs), -1).clone()
    hand_or_stock = (obs[:, 528] > 0) | (obs[:, 656] > 0)
    scores[hand_or_stock] = torch.where(matched, captures + weights, -weights)[
        hand_or_stock
    ]
    scores = torch.cat(
        (scores, obs.new_full((len(obs), 1), 100), obs.new_full((len(obs), 1), -100)), 1
    )
    return scores.masked_fill(~legal, -torch.inf).argmax(1)


@torch.no_grad()
def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, payload = load_policy(args.checkpoint)
    profile = payload["config"]["profile"]
    opponent = None
    if args.opponent not in ("self", "greedy", "published_wp"):
        opponent, other = load_policy(args.opponent)
        if other["config"]["profile"] != profile:
            raise ValueError("Opponent must be adapted to the same rule")
    if args.opponent == "published_wp":
        from koikoi.env import BatchEnv
        from koikoi.upstream_batch import BatchPublishedAgent

        encoder = BatchEnv(1, "cuda", "paper8")
        published = BatchPublishedAgent("wp", "cuda", endgame=False)
    protocol = {
        "config": vars(args),
        "profile": profile,
        "checkpoint_sha256": sha(args.checkpoint),
        "opponent_sha256": sha(args.opponent) if opponent is not None else None,
        "purpose": "development; not independent selected-rule confirmation",
    }
    write(out / "protocol.json", protocol)
    backend = PublicStockBackend("cuda")
    rng = torch.Generator(device="cuda").manual_seed(args.search_seed)
    counts = {
        "matches": 0,
        "rounds": 0,
        "decisions": 0,
        "rollouts": 0,
        "transitions": 0,
        "roots": 0,
        "python_replay_decisions": 0,
    }
    scores, margins, seeds_all = [], [], []
    started = time.perf_counter()
    for offset in range(0, args.pairs, args.batch):
        size = min(args.batch, args.pairs - offset)
        seeds = list(range(args.deal_seed + offset, args.deal_seed + offset + size)) * 2
        batch = PublicStockBatch(seeds, profile, "cuda", backend)
        seats = torch.arange(2, device="cuda").repeat_interleave(size)
        trace = []
        for _ in range(700):
            alive = batch.state[:, PHASE] != TERMINAL
            if not alive.any():
                break
            obs, mask = batch.observe()
            learner = alive & (batch.state[:, PLAYER] == seats)
            actions = torch.zeros(len(seeds), device="cuda", dtype=torch.long)
            ids = learner.nonzero().flatten()
            if len(ids):
                if args.particles:
                    chosen, work = rollout_actions(
                        view(batch.state[ids], batch),
                        model,
                        args.particles,
                        args.horizon,
                        rng,
                        args.confidence,
                    )
                    for key, value in work.items():
                        counts[key] += value
                else:
                    chosen = model(obs[ids], mask[ids])[0].argmax(1)
                actions[ids] = chosen
            ids = (alive & ~learner).nonzero().flatten()
            if len(ids):
                if args.opponent == "greedy":
                    actions[ids] = greedy_actions(obs[ids], mask[ids])
                elif args.opponent == "published_wp":
                    ordinary = ids[batch.state[ids, PHASE] != 5]
                    if len(ordinary):
                        encoder.state = batch.state[ordinary].contiguous()
                        encoder.count = len(ordinary)
                        actions[ordinary] = published.act(encoder).long()
                    stock = ids[batch.state[ids, PHASE] == 5]
                    if len(stock):
                        actions[stock] = greedy_actions(obs[stock], mask[stock])
                else:
                    actions[ids] = (model if opponent is None else opponent)(
                        obs[ids], mask[ids]
                    )[0].argmax(1)
            trace.append(torch.where(alive, actions, -1).cpu().numpy().astype(np.int16))
            counts["decisions"] += int(alive.sum())
            result = batch.step(actions)
            counts["rounds"] += int(result[:, 1].sum())
            counts["matches"] += int(result[:, 2].sum())
        else:
            raise RuntimeError("Match failed to terminate")
        final = batch.state.cpu().numpy()
        trajectory = np.stack(trace)
        # Independent Python replay validates all scored trajectories, not samples.
        for j, seed in enumerate(seeds):
            ref = PublicStockEnv(seed, profile)
            for action in trajectory[:, j]:
                if action >= 0:
                    ref.step(int(action))
                    counts["python_replay_decisions"] += 1
            np.testing.assert_array_equal(ref.state, final[j])
        margin = (final[:, POINTS] - final[:, POINTS + 1]) * np.repeat([1, -1], size)
        scores.extend(((np.sign(margin) + 1) / 2).reshape(2, size).mean(0).tolist())
        margins.extend(margin.reshape(2, size).mean(0).tolist())
        seeds_all.extend(seeds[:size])
        np.savez_compressed(
            out / f"trajectories_{offset:05d}.npz",
            actions=trajectory,
            final=final,
            seeds=seeds,
        )
        print(
            json.dumps(
                dict(
                    pairs_done=offset + size,
                    seconds=time.perf_counter() - started,
                    win_rate=float(np.mean(scores)),
                    **counts,
                )
            ),
            flush=True,
        )
    result = dict(
        **protocol,
        **counts,
        seconds=time.perf_counter() - started,
        seeds=seeds_all,
        scores=scores,
        margins=margins,
        win_rate=float(np.mean(scores)),
    )
    write(out / "result.json", result)


def train_matrix(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        "profiles": list(PROFILES),
        "training_seeds": [911, 912],
        "updates": args.updates,
        "decisions_per_model": 512 * 64 * args.updates,
        "deal_seed_bases": [814000000, 815000000],
        "qualification": "Evaluate control against greedy and published WP before claiming adequate policy quality.",
        "source_hashes": {
            p: sha(p)
            for p in [
                "scripts/train_public_stock.py",
                "koikoi/public_stock_policy.py",
                "koikoi/public_stock.py",
                "koikoi/public_stock_native.py",
                "koikoi/native/public_stock.h",
            ]
        },
    }
    write(out / "protocol.json", plan)
    for profile in PROFILES:
        for i, seed in enumerate(plan["training_seeds"]):
            cmd = [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "scripts.train_public_stock",
                "--profile",
                profile,
                "--out",
                str(out / f"{profile}_seed{seed}"),
                "--updates",
                str(args.updates),
                "--seed",
                str(seed),
                "--deal-seed",
                str(plan["deal_seed_bases"][i]),
                "--save-every",
                str(args.updates),
            ]
            print(json.dumps({"command": cmd}), flush=True)
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--out", required=True)
    p.add_argument("--updates", type=int, default=1024)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="self")
    p.add_argument("--out", required=True)
    p.add_argument("--deal-seed", type=int, default=816000000)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--particles", type=int, default=0)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--confidence", type=float, default=1.0)
    p.add_argument("--search-seed", type=int, default=921)
    args = parser.parse_args()
    (train_matrix if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
