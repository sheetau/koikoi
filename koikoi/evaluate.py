"""Paired, reproducible complete-match evaluation with pair-level bootstrap CIs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .agents import EndgameAgent, GreedyAgent, NeuralAgent, RandomAgent
from .layout import CONTINUE, KOI, PHASE, PLAYER, POINTS, TERMINAL
from .reference import ReferenceEnv
from .search import SearchAgent
from .train import provenance
from .upstream import UPSTREAM_REV, PublishedAgent


def make_agent(name, seed, deterministic=False, device="cpu"):
    if name == "random":
        return RandomAgent(seed)
    if name == "greedy":
        return GreedyAgent()
    if name in ("rl-wp", "rl-point"):
        return PublishedAgent("wp" if name == "rl-wp" else "point")
    if name in ("rl-wp-assisted", "rl-point-assisted"):
        return EndgameAgent(
            PublishedAgent("wp" if name == "rl-wp-assisted" else "point")
        )
    return NeuralAgent(name, device=device, deterministic=deterministic)


def paired_interval(samples, seed=99, replicates=5000):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 100):
        size = min(100, replicates - start)
        indices = rng.integers(len(samples), size=(size, len(samples)))
        means[start : start + size] = samples[indices].mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def duel(a, b, seed, profile):
    env = ReferenceEnv(seed, profile)
    agents, timings = (a, b), [[], []]
    stats = [
        {
            "discard": 0,
            "pick_hand": 0,
            "pick_draw": 0,
            "koi": 0,
            "continued": 0,
            "stopped": 0,
        }
        for _ in range(2)
    ]
    decisions = 0
    while env.state[PHASE] != TERMINAL:
        player = int(env.state[PLAYER])
        phase = int(env.state[PHASE])
        start = time.perf_counter()
        action = agents[player].act(env)
        timings[player].append(time.perf_counter() - start)
        stats[player][("discard", "pick_hand", "pick_draw", "koi")[phase]] += 1
        if phase == KOI:
            stats[player]["continued" if action == CONTINUE else "stopped"] += 1
        env.step(action)
        decisions += 1
    difference = int(env.state[POINTS] - env.state[POINTS + 1])
    result = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
    return result, difference, timings, decisions, stats


def seed_agents(a, b, seed):
    """Each duplicate deal is replayable independently of earlier pair lengths."""
    torch.manual_seed(seed)
    for offset, agent in enumerate((a, b)):
        while isinstance(agent, EndgameAgent):
            agent = agent.agent
        if hasattr(agent, "rng"):
            agent.rng = np.random.default_rng(seed + 101 * (offset + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        required=True,
        help="random, greedy, rl-wp, rl-point, or local checkpoint",
    )
    parser.add_argument("--opponent", default="rl-wp")
    parser.add_argument("--profile", choices=("paper8", "research12"), default="paper8")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--endgame",
        action="store_true",
        help="Apply exact outcome-dominance rules to the candidate",
    )
    parser.add_argument(
        "--search", choices=("none", "uniform", "belief"), default="none"
    )
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--search-horizon", type=int, default=32)
    parser.add_argument(
        "--search-confidence",
        type=float,
        default=2.5,
        help="Paired standard-error margin before changing the base action; not a strength guarantee",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for candidate/checkpoint inference; published models stay on CPU",
    )
    parser.add_argument(
        "--search-all",
        action="store_true",
        help="Search every nontrivial decision, not only koi/stop",
    )
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    if args.profile != "paper8" and any(
        x in ("rl-wp", "rl-point", "rl-wp-assisted", "rl-point-assisted")
        for x in (args.agent, args.opponent)
    ):
        parser.error("Published baselines require paper8")
    if args.split == "test" and args.seed < 100_000_000:
        parser.error("Final test seeds start at 100000000; do not tune on that split")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    evaluation_provenance = provenance()
    a = make_agent(args.agent, args.seed + 101, args.deterministic, args.device)
    b = make_agent(args.opponent, args.seed + 202, args.deterministic, args.device)
    checkpoint_metadata = {
        name: {
            "sha256": agent.checkpoint_sha256,
            "training_counters": agent.training_counters,
            "training_provenance": agent.training_provenance,
        }
        for name, agent in ((args.agent, a), (args.opponent, b))
        if isinstance(agent, NeuralAgent)
    }
    for agent in (a, b):
        if isinstance(agent, NeuralAgent) and agent.profile != args.profile:
            parser.error(
                "Checkpoint rule profile differs; transfer evaluations need a separate protocol"
            )
    if args.search != "none":
        if not isinstance(a, NeuralAgent):
            parser.error("Search requires a neural checkpoint")
        a = SearchAgent(
            a,
            args.particles,
            args.search_horizon,
            args.seed + 303,
            args.search == "belief",
            koi_only=not args.search_all,
            confidence=args.search_confidence,
        )
    search_agent = a
    if args.endgame:
        a = EndgameAgent(a)
    pairs, differences, records, timings = [], [], [], []
    total_decisions = 0
    agent_stats, opponent_stats = {}, {}
    for pair in range(args.pairs):
        seed = args.seed + pair
        seed_agents(a, b, seed)
        first = duel(a, b, seed, args.profile)
        seed_agents(a, b, seed)
        second = duel(b, a, seed, args.profile)
        pair_score = (first[0] + 1 - second[0]) / 2
        pair_diff = (first[1] - second[1]) / 2
        pairs.append(pair_score)
        differences.append(pair_diff)
        records.append(
            {
                "seed": seed,
                "seat0_score": first[0],
                "seat1_score": 1 - second[0],
                "seat0_point_difference": first[1],
                "seat1_point_difference": -second[1],
            }
        )
        timings.extend(first[2][0] + second[2][1])
        total_decisions += first[3] + second[3]
        for target, lhs, rhs in (
            (agent_stats, first[4][0], second[4][1]),
            (opponent_stats, first[4][1], second[4][0]),
        ):
            for key in lhs:
                target[key] = target.get(key, 0) + lhs[key] + rhs[key]
        if (pair + 1) % 10 == 0:
            print(
                json.dumps({"pairs": pair + 1, "score": float(np.mean(pairs))}),
                flush=True,
            )
    report = {
        "protocol_version": 2,
        "agent": args.agent,
        "opponent": args.opponent,
        "profile": args.profile,
        "split": args.split,
        "seed_start": args.seed,
        "pairs": args.pairs,
        "matches": 2 * args.pairs,
        "draw_adjusted_win_rate": float(np.mean(pairs)),
        "win_rate_ci95": paired_interval(pairs),
        "mean_point_difference": float(np.mean(differences)),
        "point_difference_ci95": paired_interval(differences),
        "mean_decision_ms": 1000 * float(np.mean(timings)),
        "p95_decision_ms": 1000 * float(np.quantile(timings, 0.95)),
        "decisions": total_decisions,
        "search_simulated_decisions": getattr(search_agent, "simulated_decisions", 0),
        "searched_decisions": getattr(search_agent, "searched_decisions", 0),
        "changed_decisions": getattr(search_agent, "changed_decisions", 0),
        "search_confidence": args.search_confidence if args.search != "none" else None,
        "endgame": args.endgame,
        "solved_decisions": getattr(a, "solved_decisions", 0),
        "agent_decisions": agent_stats,
        "opponent_decisions": opponent_stats,
        "search": args.search,
        "search_all": args.search_all,
        "device": args.device,
        "search_particles": args.particles if args.search != "none" else 0,
        "search_horizon": args.search_horizon if args.search != "none" else 0,
        "deterministic": args.deterministic,
        "upstream_revision": UPSTREAM_REV,
        "provenance": evaluation_provenance,
        "records": records,
    }
    report["checkpoints"] = checkpoint_metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
