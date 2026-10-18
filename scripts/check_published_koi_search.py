"""Bounded capability/latency check, not a match-strength experiment."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import GreedyAgent, solved_koi_action
from koikoi.layout import KOI, PHASE, PLAYER, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.published_koi_search import PublishedKoiSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def corpus(count=16):
    states = []
    work = {"decisions": 0, "round_endings": 0, "completed_matches": 0}
    for seed in range(25910001, 25910101):
        env = ReferenceEnv(seed, "paper8")
        while env.state[PHASE] != TERMINAL:
            if env.state[PHASE] == KOI and solved_koi_action(env) is None:
                states.append(env.state.copy())
                if len(states) == count:
                    return np.stack(states), work
            result = env.step(GreedyAgent().act(env))
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
    raise RuntimeError("Insufficient unguarded roots")


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh capability output directory")
    if not torch.cuda.is_available():
        raise ValueError("This capability check requires CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = {
        str(path): sha(path)
        for path in [
            Path(__file__),
            Path("scripts/published_koi_search.py"),
            Path("scripts/collect_reusable_paired_targets.py"),
            Path("scripts/paired_rollout_targets.py"),
            Path("tests/test_published_koi_search.py"),
        ]
    }
    for path, expected in PARENTS:
        assert sha(path) == expected
        inputs[path] = expected
    source = provenance()
    folder.mkdir(parents=True)
    states, work = corpus()
    torch.save(torch.from_numpy(states), folder / "roots.pt")
    cases = {}
    for index, (checkpoint, _) in enumerate(PARENTS, 1):
        search = PublishedKoiSearch(checkpoint, particles=32, confirmation_particles=32)
        rows = []
        for offset in range(-2, len(states)):
            root_index = offset if offset >= 0 else offset + 2
            env = ReferenceEnv(1, "paper8")
            env.state[:] = states[root_index]
            search.reset_seed(25920001 + root_index)
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = search.act(env)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            np.testing.assert_array_equal(env.state, states[root_index])
            assert env.legal()[action]
            rows.append(
                {
                    "root": root_index,
                    "warmup": offset < 0,
                    "seconds": seconds,
                    "action": action,
                    "decision": search.last_decision,
                    "cumulative_branch_work": asdict(search.work),
                    "cumulative_confirmation_work": asdict(search.confirmation_work),
                }
            )
            print(
                json.dumps({"parent": index, "root": offset, "seconds": seconds}),
                flush=True,
            )
        warmup = rows[1]["cumulative_branch_work"]
        timed_work = {k: v - warmup[k] for k, v in asdict(search.work).items()}
        durations = [r["seconds"] for r in rows if not r["warmup"]]
        cases[f"parent{index}"] = {
            "rows": rows,
            "warmup_branch_work": warmup,
            "timed_branch_work": timed_work,
            "timed_mean_seconds": float(np.mean(durations)),
            "timed_median_seconds": float(np.median(durations)),
            "timed_max_seconds": float(np.max(durations)),
            "changed_timed_roots": sum(
                r["decision"]["changed"] for r in rows if not r["warmup"]
            ),
        }
        del search
    assert provenance()["source_sha256"] == source["source_sha256"]
    assert all(sha(path) == digest for path, digest in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "provenance": source,
        "roots_sha256": sha(folder / "roots.pt"),
        "roots": len(states),
        "root_rounds": states[:, ROUND].tolist(),
        "root_players": states[:, PLAYER].tolist(),
        "source_work": work,
        "cases": cases,
        "settings": {
            "particles": 32,
            "confirmation_particles": 32,
            "confidence": 2.5,
            "root_device": "cpu",
            "root_threads": 2,
            "continuation_device": "cuda",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "scope": "Fixed greedy-play unguarded koi roots, two warmup calls and sixteen timed calls per parent. Includes hidden sampling, singleton CPU fallback, batched CUDA terminal continuations and confirmation when proposed. Other CPU evaluation may be active; this is not isolated deployment timing or evidence of stronger match play. No fitting or final-test use.",
    }
    write(folder / "audit.json", result)
    print(
        json.dumps(
            {"output": str(folder / "audit.json"), "sha256": sha(folder / "audit.json")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
