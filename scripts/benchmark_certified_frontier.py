"""Focused serial timing of public-certificate pruning before proposal creation."""

import argparse
import gc
import time
from pathlib import Path

from koikoi.layout import TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.certified_frontier import solve_certified_frontier
from scripts.continuation_frontier import (
    ContinuationActors,
    solve_continuation_frontier,
    verify_continuation_patch,
)
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.study_continuation_frontier import assess
from scripts.uniform_controller_study import actors, load


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    previous = read("runs/continuation_frontier_dev_v1/protocol.json")
    paths = set(previous["inputs"])
    paths.update(
        (
            "scripts/certified_frontier.py",
            "tests/test_certified_frontier.py",
            "scripts/benchmark_certified_frontier.py",
            "docs/CERTIFIED_FRONTIER_SKIP.rst",
        )
    )
    base.mkdir(parents=True)
    write(
        base / "protocol.json",
        {
            "scope": "Special-case serial development timing at original fixture 2, all 1304 frontier nodes publicly certified; no population latency or strength inference",
            "fixture": previous["fixtures"][2],
            "families": ["parent_default", "other_parent", "policy_pool"],
            "repetitions": 2,
            "order": "old/new then new/old per family; fresh dynamics and action caches; no concurrent research worker",
            "shared_transitions": 2000000,
            "local_transitions": 100000,
            "prefix_transitions": 250000,
            "capacity": 4096,
            "inputs": {p: sha(p) for p in sorted(paths)},
        },
    )


def check_pruning(old, result):
    new = result.patch
    if (
        old.decisions != new.decisions
        or old.gain_lower != new.gain_lower
        or old.frontier.parent_program != new.frontier.parent_program
    ):
        raise ValueError("Reference, physical work or certified gain changed")
    omitted = {key for key, _, _ in result.skipped}
    if len(omitted) != len(result.skipped):
        raise ValueError("Repeated skipped history")
    if new.comparisons != tuple(
        (key, r) for key, r in old.comparisons if key not in omitted
    ):
        raise ValueError("Uncertified comparison changed")
    for key, r in old.comparisons:
        if key in omitted:
            receipts = [r["parent_receipt"], *r["candidate_receipts"]]
            if (
                r["decisions"]
                or r["selection"].candidate is not None
                or any(
                    p.lower != p.upper or p.lower != receipts[0].lower for p in receipts
                )
            ):
                raise ValueError("Omitted comparison was not value-identical")


def run(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    source = read(p["fixture"]["source"])
    worlds = load(p["fixture"]["program"])["worlds"]
    parents, opponents, _ = actors()
    parent_index, opponent = source["parent"], source["opponent"]
    parent, identity = parents[parent_index], PARENTS[parent_index][1]
    identities = read("runs/uniform_controller_evaluation/protocol.json")[
        "opponent_identities"
    ]
    rows = []
    for family in p["families"]:
        alternatives = {}
        if family != "parent_default":
            alternatives[PARENTS[1 - parent_index][1]] = parents[1 - parent_index]
        if family == "policy_pool":
            alternatives.update((identities[k], opponents[k]) for k in ("wp", "point"))
        for repetition in range(p["repetitions"]):
            results = {}
            for arm in ("old", "pruned") if repetition == 0 else ("pruned", "old"):
                folder = base / f"{family}_repeat_{repetition}_{arm}"
                folder.mkdir()
                registry = ContinuationActors(identity, parent, alternatives)
                dynamics = KoiTailDynamics(
                    p["capacity"],
                    worlds.model_names,
                    {opponent: opponents[opponent]},
                    parent,
                )
                kwargs = {
                    "draws": 14 - int(worlds.states[0, TURN]),
                    "max_decisions": p["shared_transitions"],
                    "prefix_decisions": p["prefix_transitions"],
                    "local_decisions": p["local_transitions"],
                }
                gc.collect()
                start = time.perf_counter()
                result = (
                    solve_continuation_frontier
                    if arm == "old"
                    else solve_certified_frontier
                )(
                    worlds,
                    dynamics,
                    registry,
                    {opponent: identities[opponent]},
                    **kwargs,
                )
                seconds = time.perf_counter() - start
                patch = result if arm == "old" else result.patch
                compressed_save(folder / "program.pkl.gz", result)
                row = {
                    "family": family,
                    "repetition": repetition,
                    "arm": arm,
                    "seconds": seconds,
                    "decisions": patch.decisions,
                    "gain_lower": str(patch.gain_lower),
                    "frontier_nodes": len(patch.frontier.nodes),
                    "comparisons": len(patch.comparisons),
                    "continuation_calls": dict(registry.calls),
                    "program_sha256": sha(folder / "program.pkl.gz"),
                }
                write(folder / "selection.json", row)
                print(row, flush=True)
                results[arm] = result
                rows.append(row)
                if arm == "pruned":
                    if len(result.skipped) != 1304 or patch.comparisons:
                        raise ValueError("Unexpected special-case frontier")
                    oracle = verify_continuation_patch(
                        worlds, patch, registry, {opponent: opponents[opponent]}
                    )
                    execution = assess(source, patch, registry, opponents[opponent])
                    write(
                        folder / "audit.json",
                        {
                            "status": "passed",
                            "oracle": oracle,
                            "execution": execution,
                            "selection_sha256": sha(folder / "selection.json"),
                        },
                    )
                del dynamics, registry
            check_pruning(results["old"], results["pruned"])
    totals = {
        family: {
            arm: sum(
                r["seconds"] for r in rows if r["family"] == family and r["arm"] == arm
            )
            for arm in ("old", "pruned")
        }
        for family in p["families"]
    }
    bindings = {
        str(path): sha(path)
        for path in sorted(base.glob("*/*"))
        if path.suffix in (".json", ".gz")
    }
    write(
        base / "summary.json",
        {
            "scope": p["scope"],
            "protocol_sha256": sha(base / "protocol.json"),
            "rows": rows,
            "seconds": totals,
            "runtime_ratios": {
                family: v["pruned"] / v["old"] for family, v in totals.items()
            },
            "all_unchanged_comparisons_equal": True,
            "result_bindings": bindings,
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
