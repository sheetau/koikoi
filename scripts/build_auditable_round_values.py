"""Backward sampled policy evaluation on the finite public round-boundary state."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.public_stock_native import PublicStockBackend, PublicStockBatch
from koikoi.public_stock_residual import load_residual
from koikoi.public_stock_round_value import boundary_value
from koikoi.public_stock_search import view


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=840000000)
    args = p.parse_args()
    if not 2 <= args.samples <= 1024:
        raise ValueError("Samples must fit the nonoverlapping round seed blocks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, payload = load_residual(args.checkpoint)
    profile = payload["config"]["profile"]
    protocol = {
        "config": vars(args),
        "profile": profile,
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "meaning": "Value for player 0 before the fresh deal, indexed by absolute dealer and player-0 points; same fixed policy on both sides",
        "method": "Backward sampled single-round transitions, with independent initial deals per round/point cell",
        "sources": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [
                __file__,
                "koikoi/public_stock_round_value.py",
                "koikoi/public_stock_residual.py",
            ]
        },
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    values = torch.zeros((8, 2, 61), device="cuda")
    values[:, :, 0], values[:, :, 60] = -1, 1
    targets, standard_errors = [], []
    counts = {
        "conditional_round_rollouts": 0,
        "terminal_round_rollouts": 0,
        "transitions": 0,
    }
    started = time.perf_counter()
    n = 118 * args.samples
    backend = PublicStockBackend("cuda")
    for round_index in reversed(range(8)):
        batch = PublicStockBatch(
            range(
                args.seed + round_index * 131072, args.seed + round_index * 131072 + n
            ),
            profile,
            "cuda",
            backend,
        )
        # Native initial deals are valid uniform proposals. Round metadata and
        # public points are set to the conditional state before any action;
        # future deals generated after settlement are never inspected here.
        batch.state[:, ROUND] = round_index
        dealer = torch.arange(2, device="cuda").repeat_interleave(59 * args.samples)
        batch.state[:, PLAYER] = batch.state[:, DEALER] = dealer
        points = (
            torch.arange(1, 60, device="cuda").repeat_interleave(args.samples).repeat(2)
        )
        batch.state[:, POINTS], batch.state[:, POINTS + 1] = points, 60 - points
        finished = torch.zeros(n, dtype=torch.bool, device="cuda")
        action_trace = []
        for _ in range(80):
            alive = (~finished).nonzero().flatten()
            if not len(alive):
                break
            current = view(batch.state[alive], batch)
            inputs, mask = model.encode(current)
            scores, _ = model(inputs, mask)
            chosen = scores.argmax(1)
            action_row = torch.full((n,), -1, device="cuda", dtype=torch.int16)
            action_row[alive] = chosen.to(torch.int16)
            action_trace.append(action_row.cpu().numpy())
            result = current.step(chosen)
            batch.state[alive] = current.state
            finished[alive] = result[:, 1] > 0
            counts["transitions"] += len(alive)
        if not finished.all():
            raise RuntimeError("A round exceeded the explicit 80-decision bound")
        target = boundary_value(
            batch.state, torch.zeros(n, device="cuda"), values
        ).reshape(2, 59, args.samples)
        values[round_index, :, 1:60] = target.mean(2)
        np.savez_compressed(
            out / f"round_{round_index}.npz",
            actions=np.stack(action_trace),
            final_public=batch.state[:, [PHASE, ROUND, DEALER, POINTS, POINTS + 1, 114]]
            .cpu()
            .numpy(),
        )
        targets.append(target.cpu().numpy())
        standard_errors.append((target.std(2) / args.samples**0.5).cpu().numpy())
        counts["conditional_round_rollouts"] += n
        counts["terminal_round_rollouts"] += int(
            (batch.state[:, PHASE] == TERMINAL).sum()
        )
        print(
            json.dumps(
                {
                    "round_index": round_index,
                    **counts,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        out / "table.npz",
        value=values.cpu().numpy(),
        targets=np.stack(targets[::-1]),
        conditional_standard_error=np.stack(standard_errors[::-1]),
    )
    summary = {
        **protocol,
        **counts,
        "seconds": time.perf_counter() - started,
        "table_sha256": hashlib.sha256((out / "table.npz").read_bytes()).hexdigest(),
        "traces": {
            f"round_{r}.npz": hashlib.sha256(
                (out / f"round_{r}.npz").read_bytes()
            ).hexdigest()
            for r in range(8)
        },
        "limits": "Sampled values; conditional SE excludes propagated uncertainty from later-round tables. Not strength evidence.",
    }
    (out / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
