"""Fixed on-policy root/cost diagnostic; no full-match strength inference.

Both retained parents supply32 unsolved own koi roots against an alternating
WP/Point whole-match schedule. At every root compare32 and256 particles with
equal-size independent confirmation and confidence2.5. Retain every outcome;
do not select roots by estimated gains. This does not change the live study.
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent, solved_koi_action
from koikoi.evaluate import make_agent
from koikoi.layout import KOI, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.published_koi_search import PublishedKoiSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def collect(checkpoint, first_seed, count=32):
    parent = NeuralAgent(checkpoint, device="cpu", deterministic=True)
    opponents = [
        make_agent(name, 1, deterministic=True)
        for name in ("rl-wp-assisted", "rl-point-assisted")
    ]
    states, metadata = [], []
    work = {"decisions": 0, "round_endings": 0, "completed_matches": 0}
    for index in range(1000):
        seed = first_seed + index
        env = ReferenceEnv(seed, "paper8")
        seat, opponent = (index // 2) % 2, index % 2
        while env.state[PHASE] != TERMINAL:
            own = int(env.state[PLAYER]) == seat
            if own and env.state[PHASE] == KOI and solved_koi_action(env) is None:
                states.append(env.state.copy())
                metadata.append(
                    {
                        "seed": seed,
                        "seat": seat,
                        "opponent": opponent,
                        "round": int(env.state[ROUND]),
                    }
                )
                if len(states) == count:
                    return np.stack(states), metadata, work
            result = env.step((parent if own else opponents[opponent]).act(env))
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_matches"] += int(result[2])
    raise RuntimeError("Insufficient fixed-protocol source roots")


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh diagnostic directory")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/published_koi_search.py"),
            Path("scripts/collect_reusable_paired_targets.py"),
            Path("scripts/paired_rollout_targets.py"),
        )
    }
    for path, digest in PARENTS:
        assert sha(path) == digest
        inputs[path] = digest
    source = provenance()
    folder.mkdir(parents=True)
    cases = {}
    for parent_index, (checkpoint, _) in enumerate(PARENTS, 1):
        states, metadata, source_work = collect(
            checkpoint, 25930001 + (parent_index - 1) * 10000
        )
        roots_path = folder / f"parent{parent_index}_roots.pt"
        torch.save(
            {"states": torch.from_numpy(states), "metadata": metadata}, roots_path
        )
        budgets = {}
        for particles in (32, 256):
            search = PublishedKoiSearch(
                checkpoint, particles=particles, confirmation_particles=particles
            )
            original_compare = search._compare
            trace = []

            def record(
                reference,
                base,
                alternative,
                count,
                rng,
                confirmation=False,
                original_compare=original_compare,
                search=search,
                trace=trace,
            ):
                before = asdict(search.work)
                result = original_compare(
                    reference, base, alternative, count, rng, confirmation
                )
                terminal = search.workspaces[count].env.state
                assert (terminal[:, PHASE] == TERMINAL).all()
                points = (
                    (
                        (terminal[:, POINTS] - terminal[:, POINTS + 1])
                        * (1 - 2 * int(reference.state[PLAYER]))
                    )
                    .cpu()
                    .numpy()
                    .reshape(count, 2)
                )
                changes = np.diff(np.sign(points).astype(float), axis=1).ravel() / 2
                expected = float(
                    changes.mean() - 2.5 * changes.std(ddof=1) / math.sqrt(count)
                )
                assert abs(result["lower_margin"] - expected) < 1e-12
                assert result["point_change"] == float(np.diff(points, axis=1).mean())
                trace.append(
                    {
                        "confirmation": confirmation,
                        "points": points.tolist(),
                        "summary": result,
                        "work": {
                            k: v - before[k] for k, v in asdict(search.work).items()
                        },
                    }
                )
                return result

            search._compare = record
            rows = []
            for root_index, state in enumerate(states):
                env = ReferenceEnv(1, "paper8")
                env.state[:] = state
                # Same proposal/confirmation seed across budgets; smaller samples
                # are prefixes, not independent replications of the larger sample.
                search.reset_seed(25950001 + parent_index * 1000 + root_index)
                trace.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()
                action = search.act(env)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                np.testing.assert_array_equal(env.state, state)
                assert env.legal()[action]
                rows.append(
                    {
                        "root": root_index,
                        "metadata": metadata[root_index],
                        "seconds": seconds,
                        "action": action,
                        "decision": search.last_decision,
                        "branches": list(trace),
                    }
                )
                if (root_index + 1) % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "parent": parent_index,
                                "particles": particles,
                                "roots": root_index + 1,
                            }
                        ),
                        flush=True,
                    )
            budgets[str(particles)] = {
                "rows": rows,
                "work": asdict(search.work),
                "confirmation_work": asdict(search.confirmation_work),
                "proposed_roots": search.proposed_decisions,
                "changed_roots": search.changed_decisions,
                "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
                "median_seconds": float(np.median([r["seconds"] for r in rows])),
            }
            del search
        cases[f"parent{parent_index}"] = {
            "source_work": source_work,
            "roots_sha256": sha(roots_path),
            "budgets": budgets,
        }
    assert all(sha(p) == digest for p, digest in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "provenance": source,
        "cases": cases,
        "roots_per_parent": 32,
        "confidence": 2.5,
        "settings": {
            "root_threads": 2,
            "root_device": "cpu",
            "continuation_device": "cuda",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        "scope": "Fixed on-policy root diagnostic, not repeated-search match evaluation. Root selection precedes branch outcomes. Smaller proposal/confirmation samples share prefixes with the larger samples. Each baseline source match may supply multiple roots. Branch terminal outcomes retain modeling error from uniform hidden worlds and parent continuation at both seats. Timings include cold first calls and concurrent CPU activity. No training, strength qualification, final-test use or deployment promotion.",
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
