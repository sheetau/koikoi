"""Paired, reproducible complete-match evaluation with pair-level bootstrap CIs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .agents import GreedyAgent, NeuralAgent, RandomAgent
from .layout import PHASE, PLAYER, POINTS, TERMINAL
from .reference import ReferenceEnv
from .search import SearchAgent
from .train import provenance
from .upstream import PublishedAgent, UPSTREAM_REV


def make_agent(name, seed, deterministic=False):
    if name == "random":
        return RandomAgent(seed)
    if name == "greedy":
        return GreedyAgent()
    if name in ("rl-wp", "rl-point"):
        return PublishedAgent("wp" if name == "rl-wp" else "point")
    return NeuralAgent(name, deterministic=deterministic)


def paired_interval(samples, seed=99, replicates=5000):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 100):
        size = min(100, replicates-start)
        indices = rng.integers(len(samples), size=(size, len(samples)))
        means[start:start+size] = samples[indices].mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def duel(a, b, seed, profile):
    env = ReferenceEnv(seed, profile)
    agents, timings = (a, b), [[], []]
    decisions = 0
    while env.state[PHASE] != TERMINAL:
        player = int(env.state[PLAYER])
        start = time.perf_counter()
        action = agents[player].act(env)
        timings[player].append(time.perf_counter()-start)
        env.step(action)
        decisions += 1
    difference = int(env.state[POINTS]-env.state[POINTS+1])
    result = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
    return result, difference, timings, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="random, greedy, rl-wp, rl-point, or local checkpoint")
    parser.add_argument("--opponent", default="rl-wp")
    parser.add_argument("--profile", choices=("paper8", "research12"), default="paper8")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--search", choices=("none", "uniform", "belief"), default="none")
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--search-horizon", type=int, default=32)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")
    if args.profile != "paper8" and any(x in ("rl-wp", "rl-point") for x in (args.agent, args.opponent)):
        parser.error("Published baselines require paper8")
    if args.split == "test" and args.seed < 100_000_000:
        parser.error("Final test seeds start at 100000000; do not tune on that split")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    a = make_agent(args.agent, args.seed+101, args.deterministic)
    b = make_agent(args.opponent, args.seed+202, args.deterministic)
    for agent in (a, b):
        if isinstance(agent, NeuralAgent) and agent.profile != args.profile:
            parser.error("Checkpoint rule profile differs; transfer evaluations need a separate protocol")
    if args.search != "none":
        if not isinstance(a, NeuralAgent):
            parser.error("Search requires a neural checkpoint")
        a = SearchAgent(a, args.particles, args.search_horizon, args.seed+303, args.search == "belief")
    pairs, differences, records, timings = [], [], [], []
    total_decisions = 0
    for pair in range(args.pairs):
        seed = args.seed+pair
        first = duel(a, b, seed, args.profile)
        second = duel(b, a, seed, args.profile)
        pair_score = (first[0]+1-second[0])/2
        pair_diff = (first[1]-second[1])/2
        pairs.append(pair_score)
        differences.append(pair_diff)
        records.append({"seed": seed, "seat0_score": first[0], "seat1_score": 1-second[0],
                        "seat0_point_difference": first[1], "seat1_point_difference": -second[1]})
        timings.extend(first[2][0]+second[2][1])
        total_decisions += first[3]+second[3]
        if (pair+1) % 10 == 0:
            print(json.dumps({"pairs": pair+1, "score": float(np.mean(pairs))}), flush=True)
    report = {"agent": args.agent, "opponent": args.opponent, "profile": args.profile,
              "split": args.split, "seed_start": args.seed, "pairs": args.pairs, "matches": 2*args.pairs,
              "draw_adjusted_win_rate": float(np.mean(pairs)), "win_rate_ci95": paired_interval(pairs),
              "mean_point_difference": float(np.mean(differences)), "point_difference_ci95": paired_interval(differences),
              "mean_decision_ms": 1000*float(np.mean(timings)), "p95_decision_ms": 1000*float(np.quantile(timings, .95)),
              "decisions": total_decisions, "search_simulated_decisions": getattr(a, "simulated_decisions", 0),
              "search": args.search, "deterministic": args.deterministic,
              "upstream_revision": UPSTREAM_REV, "provenance": provenance(), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k,v in report.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
