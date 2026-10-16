"""Device-batched deterministic checkpoint duels with duplicate seat-swapped deals."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agents import NeuralAgent
from .env import BatchEnv
from .evaluate import paired_interval, validate_split
from .layout import PHASE, PLAYER, POINTS, TERMINAL
from .train import provenance


@torch.no_grad()
def paired_batch(first, second, seed, pairs, profile, device="cpu", endgame=False):
    """One complete pair per seed; terminal rows are frozen, never redealt."""
    env = BatchEnv(2 * pairs, device=device, profile=profile, seed=seed, debug=True)
    env.state[pairs:].copy_(env.state[:pairs])
    candidate_seat = torch.cat(
        [torch.zeros(pairs, device=device), torch.ones(pairs, device=device)]
    ).int()
    totals = torch.zeros(3, dtype=torch.int64, device=device)
    for _ in range(1000):  # At most four decisions per turn, 16 turns, 12 rounds.
        alive = env.state[:, PHASE] != TERMINAL
        if not bool(alive.any()):
            break
        obs, mask = env.observe()
        # Dummy logits for already-finished rows; step_frozen ignores these actions.
        mask[:, 0] |= ~alive
        candidate_turn = env.state[:, PLAYER] == candidate_seat
        action = torch.where(
            candidate_turn,
            first(obs, mask)[0].argmax(-1),
            second(obs, mask)[0].argmax(-1),
        )
        if endgame:
            solved = env.solved_actions()
            action = torch.where(solved >= 0, solved, action)
        result = env.step(action, freeze_done=True)
        totals[0] += alive.sum()
        totals[1:] += result[:, 1:3].sum(0)
    else:
        raise RuntimeError("Match exceeded the finite-game decision bound")
    difference = (env.state[:, POINTS] - env.state[:, POINTS + 1]) * (
        1 - 2 * candidate_seat
    )
    difference = difference.cpu().numpy()
    score = (np.sign(difference) + 1) / 2
    records = [
        {
            "seed": seed + i,
            "seat0_score": float(score[i]),
            "seat1_score": float(score[pairs + i]),
            "seat0_point_difference": int(difference[i]),
            "seat1_point_difference": int(difference[pairs + i]),
        }
        for i in range(pairs)
    ]
    return records, dict(zip(("decisions", "rounds", "matches"), totals.cpu().tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--opponent", required=True, type=Path)
    parser.add_argument(
        "--profile", choices=("paper8", "research12"), default="research12"
    )
    parser.add_argument("--pairs", type=int, default=10000)
    parser.add_argument("--batch-pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    parser.add_argument(
        "--endgame", action="store_true", help="Apply the guard to BOTH agents"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/batch_evaluation.json")
    )
    args = parser.parse_args()
    if args.batch_pairs < 1:
        parser.error("batch-pairs must be positive")
    try:
        validate_split(args.seed, args.pairs, args.split)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(2)
    run_provenance = provenance()
    agents = [
        NeuralAgent(p, device=args.device, deterministic=True)
        for p in (args.agent, args.opponent)
    ]
    if any(agent.profile != args.profile for agent in agents):
        parser.error("Checkpoint rule profiles must match the evaluation profile")
    records, totals = [], {"decisions": 0, "rounds": 0, "matches": 0}
    start = time.perf_counter()
    for offset in range(0, args.pairs, args.batch_pairs):
        n = min(args.batch_pairs, args.pairs - offset)
        rows, counts = paired_batch(
            agents[0].model,
            agents[1].model,
            args.seed + offset,
            n,
            args.profile,
            args.device,
            args.endgame,
        )
        records.extend(rows)
        for key in totals:
            totals[key] += counts[key]
        print(
            json.dumps(
                {"pairs": len(records), "elapsed_seconds": time.perf_counter() - start}
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - start
    scores = [(r["seat0_score"] + r["seat1_score"]) / 2 for r in records]
    differences = [
        (r["seat0_point_difference"] + r["seat1_point_difference"]) / 2 for r in records
    ]
    report = {
        "protocol_version": 2,
        "execution": "batched_deterministic",
        "agent": str(args.agent),
        "opponent": str(args.opponent),
        "profile": args.profile,
        "split": args.split,
        "seed_start": args.seed,
        "pairs": args.pairs,
        **totals,
        "draw_adjusted_win_rate": float(np.mean(scores)),
        "win_rate_ci95": paired_interval(scores),
        "mean_point_difference": float(np.mean(differences)),
        "point_difference_ci95": paired_interval(differences),
        "elapsed_seconds": elapsed,
        "matches_per_second": totals["matches"] / elapsed,
        "device": args.device,
        "batch_pairs": args.batch_pairs,
        "deterministic": True,
        "endgame": args.endgame,
        "opponent_endgame": args.endgame,
        "provenance": run_provenance,
        "checkpoints": {
            str(path): {
                "sha256": agent.checkpoint_sha256,
                "training_counters": agent.training_counters,
                "training_provenance": agent.training_provenance,
            }
            for path, agent in zip((args.agent, args.opponent), agents)
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
