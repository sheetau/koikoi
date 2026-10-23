"""Prospective paired rule screen and untouched confirmation of transferred AI."""

import argparse
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.rule_variants import AUTO_ROUNDS, COMPLETED_ROUNDS, VARIANTS, VariantEnv
from koikoi.upstream import UPSTREAM_PATH
from koikoi.upstream_batch import BatchPublishedAgent
from koikoi.variant_native import VariantBackend, VariantBatch
from scripts.run_arbitration_study import PARENTS, read, sha, write


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    files = [str(p) for p in Path("koikoi").glob("*.py")]
    files += [str(p) for p in Path("koikoi/native").glob("*.*")]
    files += [__file__, "tests/test_rule_variants.py", "docs/RULE_VARIANTS.rst"]
    files += [p for p, _ in PARENTS]
    files += [
        str(UPSTREAM_PATH / f"model_agent/{phase}_rl_{op}.pt")
        for op in ("wp", "point")
        for phase in ("discard", "pick", "koikoi")
    ]
    p = {
        "scope": "transferred weights, public payoff-correct assistance, GPU batched arithmetic",
        "variants": {name: asdict(v) for name, v in VARIANTS.items()},
        "parents": PARENTS,
        "opponents": ["wp", "point"],
        "development": {"seed": 240100000, "pairs": 256},
        "confirmation": {"seed": 240200000, "pairs": 2048},
        "batch_pairs": 64,
        "device": "cuda",
        "cpu_threads": 2,
        "selection": "highest minimum of the two parent equal-opponent mean win scores; fixed insertion-order ties",
        "primary_confirmation": "selected-minus-control gain has positive simultaneous lower bounds in both parent equal-opponent mixtures",
        "interval": "paired-seed bootstrap 20000 resamples, one-sided 97.5% per parent (Bonferroni family 95%)",
        "frozen_weights": True,
        "training": "none; transfer screen, not variant training or paper8 reproduction",
        "inputs": {f: sha(f) for f in files},
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Rule study input changed: {path}")
    return p


@torch.no_grad()
def batch(parent, opponent, variant, seed, pairs, backend, encoder):
    seeds = list(range(seed, seed + pairs)) * 2
    env = VariantBatch(seeds, variant, "cuda", backend)
    seats = torch.tensor([0] * pairs + [1] * pairs, device="cuda")
    actions_saved, counts = [], Counter()
    counts["initial_automatic_rounds"] = int(env.state[:, AUTO_ROUNDS].sum())
    counts["rounds"] = int(env.state[:, COMPLETED_ROUNDS].sum())
    counts["matches"] = int((env.state[:, PHASE] == TERMINAL).sum())
    for _ in range(600):
        alive = env.state[:, PHASE] != TERMINAL
        if not bool(alive.any()):
            break
        candidate = env.state[:, PLAYER] == seats
        actions = torch.zeros(2 * pairs, dtype=torch.int32, device="cuda")
        for own, agent in ((True, parent), (False, opponent)):
            indices = torch.where(alive & (candidate == own))[0]
            if not len(indices):
                continue
            encoder.state = env.state[indices].contiguous()
            encoder.count = len(indices)
            if own:
                features, legal = encoder.published_observe()
                logits = agent.model.actor(features, encoder.state[:, PHASE], legal)
                actions[indices] = logits.argmax(-1).int()
            else:
                actions[indices] = agent.act(encoder)
        solved = env.solved_actions()
        actions = torch.where(solved >= 0, solved, actions)
        actions_saved.append(actions.cpu().numpy().astype(np.int16))
        counts["decisions"] += int(alive.sum())
        result = env.step(actions)
        counts["rounds"] += int(result[:, 1].sum())
        counts["matches"] += int(result[:, 2].sum())
    else:
        raise ValueError("Variant match exceeded decision bound")
    final = env.state.cpu().numpy()
    if counts["matches"] != 2 * pairs or counts["rounds"] != int(
        final[:, COMPLETED_ROUNDS].sum()
    ):
        raise ValueError("Initial and decision-triggered accounting differs")
    counts["automatic_rounds"] = int(final[:, AUTO_ROUNDS].sum())
    actions = (
        np.stack(actions_saved)
        if actions_saved
        else np.empty((0, 2 * pairs), dtype=np.int16)
    )
    return actions, final, dict(counts)


def audit_batch(variant, seed, pairs, actions, final, counts):
    """Replay every recorded action with Python; no native transition reuse."""
    refs = [VariantEnv(s, variant) for s in list(range(seed, seed + pairs)) * 2]
    work = Counter(
        rounds=sum(int(e.state[COMPLETED_ROUNDS]) for e in refs),
        matches=sum(int(e.state[PHASE] == TERMINAL) for e in refs),
    )
    for step in actions:
        for env, action in zip(refs, step, strict=True):
            if env.state[PHASE] == TERMINAL:
                continue
            result = env.step(int(action))
            work["decisions"] += 1
            work["rounds"] += int(result[1])
            work["matches"] += int(result[2])
    np.testing.assert_array_equal([e.state for e in refs], final)
    if any(work[k] != counts[k] for k in ("decisions", "rounds", "matches")):
        raise ValueError("Python replay work differs")
    return dict(work)


def run(base, stage):
    p = checked(base)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parents = [NeuralAgent(path, "cuda", deterministic=True) for path, _ in PARENTS]
    if any(
        a.checkpoint_sha256 != digest
        for a, (_, digest) in zip(parents, PARENTS, strict=True)
    ):
        raise ValueError("Frozen parent checkpoint differs")
    opponents = {
        k: BatchPublishedAgent(k, "cuda", endgame=False) for k in p["opponents"]
    }
    backend, encoder = VariantBackend("cuda"), BatchEnv(1, "cuda", "paper8")
    names = (
        list(VARIANTS)
        if stage == "development"
        else ["paper_control", read(base / "development_summary.json")["selected"]]
    )
    names = list(dict.fromkeys(names))
    design = p[stage]
    for name in names:
        for parent_index, parent in enumerate(parents):
            for opponent_name, opponent in opponents.items():
                for offset in range(0, design["pairs"], p["batch_pairs"]):
                    folder = (
                        base
                        / stage
                        / name
                        / f"p{parent_index}_{opponent_name}_{offset:05}"
                    )
                    if (folder / "result.json").exists():
                        continue
                    folder.mkdir(parents=True)
                    size = min(p["batch_pairs"], design["pairs"] - offset)
                    seed = design["seed"] + offset
                    started = time.perf_counter()
                    actions, final, work = batch(
                        parent, opponent, name, seed, size, backend, encoder
                    )
                    seconds = time.perf_counter() - started
                    np.savez_compressed(
                        folder / "trace.npz", actions=actions, final=final
                    )
                    replay = audit_batch(name, seed, size, actions, final, work)
                    diff = (final[:, POINTS] - final[:, POINTS + 1]) * np.array(
                        [1] * size + [-1] * size
                    )
                    scores = (np.sign(diff) + 1) / 2
                    paired = (scores[:size] + scores[size:]) / 2
                    write(
                        folder / "result.json",
                        {
                            "status": "passed",
                            "variant": name,
                            "parent": parent_index,
                            "opponent": opponent_name,
                            "seed": seed,
                            "pairs": size,
                            "paired_scores": paired.tolist(),
                            "work": work,
                            "python_replay_work": replay,
                            "seconds": seconds,
                            "trace_sha256": sha(folder / "trace.npz"),
                            "protocol_sha256": sha(base / "protocol.json"),
                        },
                    )
                print(
                    {
                        "stage": stage,
                        "variant": name,
                        "parent": parent_index,
                        "opponent": opponent_name,
                    },
                    flush=True,
                )
    summarize(base, stage)


def summarize(base, stage):
    p = checked(base)
    names = (
        list(VARIANTS)
        if stage == "development"
        else list(
            dict.fromkeys(
                ["paper_control", read(base / "development_summary.json")["selected"]]
            )
        )
    )
    values, totals, bindings = {}, Counter(), {}
    for name in names:
        values[name] = {}
        for parent in (0, 1):
            values[name][str(parent)] = {}
            for opponent in p["opponents"]:
                scores = []
                for offset in range(0, p[stage]["pairs"], p["batch_pairs"]):
                    folder = base / stage / name / f"p{parent}_{opponent}_{offset:05}"
                    row = read(folder / "result.json")
                    if (
                        row["status"] != "passed"
                        or row["trace_sha256"] != sha(folder / "trace.npz")
                        or row["protocol_sha256"] != sha(base / "protocol.json")
                    ):
                        raise ValueError("Unverified rule result")
                    scores.extend(row["paired_scores"])
                    totals.update(row["work"])
                    bindings[str(folder / "result.json")] = sha(folder / "result.json")
                values[name][str(parent)][opponent] = scores
    means = {
        name: {
            parent: {op: float(np.mean(v)) for op, v in opp.items()}
            for parent, opp in pop.items()
        }
        for name, pop in values.items()
    }
    rank = {
        name: min(np.mean(list(opp.values())) for opp in pop.values())
        for name, pop in means.items()
    }
    report = {
        "scope": p["scope"],
        "means": means,
        "minimum_parent_mixture": rank,
        "work": dict(totals),
        "result_bindings": bindings,
        "unique_deal_streams": p[stage]["pairs"],
        "protocol_sha256": sha(base / "protocol.json"),
    }
    if stage == "development":
        report["selected"] = max(rank, key=rank.get)
    else:
        selected = read(base / "development_summary.json")["selected"]
        report["selected"], report["comparisons"] = selected, {}
        rng = np.random.default_rng(40821)
        for parent in ("0", "1"):
            delta = np.mean(
                [
                    np.array(values[selected][parent][op])
                    - np.array(values["paper_control"][parent][op])
                    for op in p["opponents"]
                ],
                axis=0,
            )
            bootstrap = np.concatenate(
                [
                    delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(
                        axis=1
                    )
                    for _ in range(20)
                ]
            )
            report["comparisons"][parent] = {
                "mean_gain": float(delta.mean()),
                "lower": float(np.quantile(bootstrap, 0.025)),
                "pairs": len(delta),
            }
        report["criterion_met"] = all(
            c["lower"] > 0 for c in report["comparisons"].values()
        )
    write(base / f"{stage}_summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--stage", choices=("development", "confirmation"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.summary:
        summarize(args.base, args.stage)
    else:
        run(args.base, args.stage)
