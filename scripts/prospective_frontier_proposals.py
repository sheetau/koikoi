"""Fresh matched-budget comparison of two local proposal mechanisms."""

import argparse
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from koikoi.env import BatchEnv
from koikoi.layout import TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.frontier_policy_patch import solve_frontier_patch
from scripts.logged_public_source import audit
from scripts.prospective_incumbent import compressed_save
from scripts.resumable_frontier import solve_resumable_frontier
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_next_discard_frontier import assess
from scripts.uniform_controller_design import POPULATIONS, specification
from scripts.uniform_controller_study import (
    actors,
    build_support,
    collect_case,
    source_files,
)
from scripts.verify_frontier_patch import verify_frontier_patch

ARMS = ("optimized_proposal", "parent_default_proposal")


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    paths = set(read("runs/resumable_frontier_dev_v1/protocol.json")["inputs"])
    paths.add(__file__)
    design = specification()
    design.update(
        sources_per_population=64, source_matches=256, workers=2, chunk_size=8
    )
    p = {
        "scope": "fresh direct-frontier controllers versus the same parent; no symbolic first phase and no old parent-refusal filter",
        "design": design,
        "arms": ARMS,
        "max_decisions": 2000000,
        "local_decisions": 100000,
        "prefix_decisions": 250000,
        "primary": "paired mean expected-gain difference parent-default minus optimized proposal; both equal-opponent parent mixtures have positive simultaneous 95% bootstrap lower bounds",
        "secondary": "adoption frequency, expected parent gains, actual paired scores, physical work and operational elapsed time",
        "interval": "20000 within-population source bootstrap replicates; one-sided 97.5% bound per parent mixture",
        "cases": [
            {"index": 4 * rep + i, "population": pop, "replicate": rep}
            for rep in range(64)
            for i, pop in enumerate(POPULATIONS)
        ],
        "opponent_identities": read("runs/uniform_controller_evaluation/protocol.json")[
            "opponent_identities"
        ],
        "inputs": {path: sha(path) for path in sorted(paths)},
        "extension": "none; all missing roots, support refusals and zero gains retained; no sample replacement",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input differs: {path}")
    return p


def worker(base, chunk):
    p = checked(base)
    config = p["design"]
    parents, opponents, predictors = actors()
    native = BatchEnv(1, debug=True)
    folder = base / f"chunk_{chunk:03}"
    folder.mkdir()
    cases = p["cases"][8 * chunk : 8 * (chunk + 1)]
    for case in cases:
        index = case["index"]
        parent_index, opponent = case["population"].split(":")
        parent = parents[int(parent_index)]
        source_dir = folder / f"source_{index:04}"
        source = collect_case(
            source_dir, case, config, parent, opponents[opponent], predictors, native
        )
        source_files(source_dir, case, source, config)
        source_audit = audit(source, predictors, parent, opponents[opponent])
        record = {
            "case": case,
            "source": str(source_dir / "record.json"),
            "source_sha256": sha(source_dir / "record.json"),
            "source_work": source["work"],
            "source_audit": source_audit,
            "status": "missing_root",
            "arms": {},
        }
        if source["selected_root"] is not None:
            support, worlds, identity = build_support(source, opponents, config)
            record["support"] = asdict(support)
            record["status"] = "support_refused" if worlds is None else "evaluated"
            if worlds is not None:
                parent_id = PARENTS[int(parent_index)][1]
                identities = {identity: p["opponent_identities"][identity]}
                # Reconstruct support twice independently of stored plan/cache.
                support_check, worlds_check, _ = build_support(
                    source, opponents, config
                )
                if asdict(support_check) != record["support"]:
                    raise ValueError("Support reconstruction mismatch")
                for name in ("states", "models", "weights"):
                    np.testing.assert_array_equal(
                        getattr(worlds, name), getattr(worlds_check, name)
                    )
                for arm in ARMS:
                    dyn = KoiTailDynamics(
                        config["capacity"],
                        worlds.model_names,
                        {identity: opponents[identity]},
                        parent,
                    )
                    common = {
                        "draws": 14 - int(worlds.states[0, TURN]),
                        "max_decisions": p["max_decisions"],
                        "prefix_decisions": p["prefix_decisions"],
                    }
                    started = time.perf_counter()
                    patch = (
                        solve_frontier_patch(
                            worlds, dyn, parent_id, identities, **common
                        )
                        if arm == "optimized_proposal"
                        else solve_resumable_frontier(
                            worlds,
                            dyn,
                            parent_id,
                            identities,
                            **common,
                            local_decisions=p["local_decisions"],
                        )
                    )
                    seconds = time.perf_counter() - started
                    path = folder / f"program_{index:04}_{arm}.pkl.gz"
                    compressed_save(path, {"worlds": worlds, "patch": patch})
                    if patch.frontier.complete:
                        oracle = verify_frontier_patch(
                            worlds,
                            patch,
                            parent,
                            {identity: opponents[identity]},
                            parent_id,
                        )
                    else:
                        if patch.comparisons or patch.gain_lower:
                            raise ValueError("Refused prefix changed the parent")
                        oracle = {
                            "exact_increment": "0",
                            "refusal": patch.frontier.refusal,
                        }
                    execution = assess(
                        source, patch, parent, opponents[identity], parent_id
                    )
                    record["arms"][arm] = {
                        "decisions": patch.decisions,
                        "seconds": seconds,
                        "gain_lower": str(patch.gain_lower),
                        "oracle": oracle,
                        "execution": execution,
                        "program": str(path),
                        "program_sha256": sha(path),
                        "frontier_nodes": len(patch.frontier.nodes),
                        "attempted_nodes": len(patch.comparisons),
                        "adopted_nodes": sum(
                            r["selection"].candidate is not None
                            for _, r in patch.comparisons
                        ),
                    }
        write(folder / f"result_{index:04}.json", record)
        print(
            {
                "source": index,
                "status": record["status"],
                "gains": {
                    a: v["oracle"]["exact_increment"] for a, v in record["arms"].items()
                },
            },
            flush=True,
        )


def summarize(base):
    p = checked(base)
    values = {a: {pop: [] for pop in POPULATIONS} for a in ARMS}
    counts, work, bindings = Counter(), {a: Counter() for a in ARMS}, {}
    for case in p["cases"]:
        i = case["index"]
        path = base / f"chunk_{i // 8:03}" / f"result_{i:04}.json"
        row = read(path)
        if row["case"] != case or sha(row["source"]) != row["source_sha256"]:
            raise ValueError("Source/case binding mismatch")
        bindings[str(path)] = sha(path)
        counts[row["status"]] += 1
        for arm in ARMS:
            result = row["arms"].get(arm)
            gain = (
                Fraction(0)
                if result is None
                else Fraction(result["oracle"]["exact_increment"])
            )
            if result is not None:
                if sha(result["program"]) != result["program_sha256"]:
                    raise ValueError("Program binding changed")
                work[arm].update(
                    {k: result[k] for k in ("decisions", "seconds", "adopted_nodes")}
                )
            values[arm][case["population"]].append(float(gain))
    report = {
        "scope": p["scope"],
        "sources": len(p["cases"]),
        "counts": dict(counts),
        "work": {a: dict(w) for a, w in work.items()},
        "result_bindings": bindings,
        "means": {
            a: {pop: float(np.mean(v)) for pop, v in group.items()}
            for a, group in values.items()
        },
        "positive_sources": {
            a: sum(x > 0 for v in group.values() for x in v)
            for a, group in values.items()
        },
        "comparisons": {},
        "protocol_sha256": sha(base / "protocol.json"),
    }
    rng = np.random.default_rng(470310)
    for parent in (0, 1):
        deltas = [
            np.array(values[ARMS[1]][f"{parent}:{op}"])
            - np.array(values[ARMS[0]][f"{parent}:{op}"])
            for op in ("wp", "point")
        ]
        bootstrap = (
            sum(
                np.concatenate(
                    [
                        d[rng.integers(len(d), size=(1000, len(d)))].mean(axis=1)
                        for _ in range(20)
                    ]
                )
                for d in deltas
            )
            / 2
        )
        report["comparisons"][str(parent)] = {
            "mean_gain": float(np.mean(deltas)),
            "lower": float(np.quantile(bootstrap, 0.025)),
        }
    report["criterion_met"] = all(
        c["lower"] > 0 for c in report["comparisons"].values()
    )
    write(base / "summary.json", report)


def run(base):
    p = checked(base)

    def launch(chunk):
        with (base / f"worker_{chunk:03}.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scripts.prospective_frontier_proposals",
                    "--base",
                    str(base),
                    "--worker",
                    str(chunk),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=p["design"]["workers"]) as pool:
        list(pool.map(launch, range(32)))
    summarize(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    elif args.summary:
        summarize(args.base)
    else:
        worker(args.base, args.worker)
