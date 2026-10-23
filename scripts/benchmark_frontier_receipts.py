"""Counterbalanced serial timing of unchanged versus reused interval receipts."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from koikoi.layout import TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.continuation_frontier import (
    ContinuationActors,
    solve_continuation_frontier,
)
from scripts.prospective_incumbent import compressed_save
from scripts.resumable_frontier import solve_resumable_frontier
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.uniform_controller_study import actors, load


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    original = read("runs/resumable_frontier_dev_v1/protocol.json")
    seen, fixtures = set(), []
    for i, fixture in enumerate(original["fixtures"]):
        parent = read(fixture["source"])["parent"]
        stratum = (fixture["draws"], parent)
        if stratum not in seen:
            fixtures.append({**fixture, "original_index": i, "parent": parent})
            seen.add(stratum)
    if seen != {(5, 0), (5, 1), (6, 0), (6, 1)}:
        raise ValueError("Four development strata required")
    paths = set(read("runs/continuation_frontier_dev_v1/protocol.json")["inputs"])
    paths.add("scripts/benchmark_frontier_receipts.py")
    base.mkdir(parents=True)
    write(
        base / "protocol.json",
        {
            "scope": "Serial development latency benchmark; four first-in-order parent/draw strata; no new source matches or strength inference",
            "fixtures": fixtures,
            "repetitions": 2,
            "order": "For each fixture run old/new then new/old; fresh dynamics and empty actor cache each arm; model loading and serialization excluded",
            "shared_transitions": 2000000,
            "local_transitions": 100000,
            "prefix_transitions": 250000,
            "capacity": 4096,
            "inputs": {p: sha(p) for p in sorted(paths)},
            "criterion": "All comparison receipts, selections, prefix masses and physical work identical; report every timing and total paired runtime ratio. No universal speed claim.",
        },
    )


def equivalent(old, new):
    if (
        old.comparisons != new.comparisons
        or old.gain_lower != new.gain_lower
        or old.decisions != new.decisions
    ):
        raise ValueError("Programs, receipts or planning work changed")
    a, b = old.frontier, new.frontier
    for name in ("parent_program", "work", "complete", "refusal", "total_mass"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"Prefix field changed: {name}")
    if len(a.nodes) != len(b.nodes):
        raise ValueError("Prefix node count changed")
    for (key1, world1), (key2, world2) in zip(a.nodes, b.nodes, strict=True):
        if key1 != key2 or world1.model_names != world2.model_names:
            raise ValueError("Prefix identity changed")
        for name in ("states", "weights", "models"):
            if not np.array_equal(getattr(world1, name), getattr(world2, name)):
                raise ValueError(f"Prefix world changed: {name}")


def run(base):
    protocol = read(base / "protocol.json")
    for path, digest in protocol["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    parents, opponents, _ = actors()
    opponent_ids = read("runs/uniform_controller_evaluation/protocol.json")[
        "opponent_identities"
    ]
    rows = []
    for i, fixture in enumerate(protocol["fixtures"]):
        source = read(fixture["source"])
        worlds = load(fixture["program"])["worlds"]
        parent, opponent = parents[source["parent"]], source["opponent"]
        identity = PARENTS[source["parent"]][1]
        first = None
        for repetition in range(protocol["repetitions"]):
            order = ("old", "reused") if repetition == 0 else ("reused", "old")
            for arm in order:
                folder = base / f"fixture_{i:02}_repeat_{repetition}_{arm}"
                folder.mkdir()
                dynamics = KoiTailDynamics(
                    protocol["capacity"],
                    worlds.model_names,
                    {opponent: opponents[opponent]},
                    parent,
                )
                registry = ContinuationActors(identity, parent, {})
                keywords = {
                    "draws": 14 - int(worlds.states[0, TURN]),
                    "max_decisions": protocol["shared_transitions"],
                    "prefix_decisions": protocol["prefix_transitions"],
                    "local_decisions": protocol["local_transitions"],
                }
                gc.collect()
                start = time.perf_counter()
                if arm == "old":
                    patch = solve_resumable_frontier(
                        worlds,
                        dynamics,
                        identity,
                        {opponent: opponent_ids[opponent]},
                        **keywords,
                    )
                else:
                    patch = solve_continuation_frontier(
                        worlds,
                        dynamics,
                        registry,
                        {opponent: opponent_ids[opponent]},
                        **keywords,
                    )
                seconds = time.perf_counter() - start
                compressed_save(folder / "program.pkl.gz", patch)
                if first is None:
                    first = patch
                else:
                    equivalent(first, patch)
                row = {
                    "fixture": i,
                    "original_fixture": fixture["original_index"],
                    "repetition": repetition,
                    "arm": arm,
                    "seconds": seconds,
                    "decisions": patch.decisions,
                    "gain_lower": str(patch.gain_lower),
                    "program_sha256": sha(folder / "program.pkl.gz"),
                    "equality": "passed",
                }
                write(folder / "result.json", row)
                rows.append(row)
                print(row, flush=True)
                del patch, dynamics, registry
    totals = {
        arm: sum(r["seconds"] for r in rows if r["arm"] == arm)
        for arm in ("old", "reused")
    }
    write(
        base / "summary.json",
        {
            "scope": protocol["scope"],
            "protocol_sha256": sha(base / "protocol.json"),
            "rows": rows,
            "seconds": totals,
            "runtime_ratio": totals["reused"] / totals["old"],
            "all_receipts_equal": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    else:
        run(args.base)
