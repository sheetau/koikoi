"""Frozen development test of pruning before optimized proposal construction."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.next_discard_frontier import next_discard_frontier
from scripts.prospective_incumbent import compressed_save
from scripts.pruned_optimized_frontier import solve_pruned_optimized_frontier
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_next_discard_frontier import assess
from scripts.uniform_controller_study import actors, load
from scripts.verify_frontier_patch import verify_frontier_patch


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    previous = read("runs/continuation_frontier_dev_v1/protocol.json")
    fixtures = [dict(f) for f in previous["fixtures"][:8]]
    baseline = [
        r
        for r in read("runs/reusable_frontier_dev_v2/summary.json")["rows"]
        if r["arm"] == "fixed"
    ]
    if [r["fixture"] for r in baseline] != list(range(8)):
        raise ValueError("All eight fixed baseline sources required")
    paths = set(previous["inputs"])
    paths.update(
        (
            "scripts/pruned_optimized_frontier.py",
            "tests/test_pruned_optimized_frontier.py",
            "scripts/certified_frontier.py",
            "scripts/study_pruned_optimized_frontier.py",
            "docs/PRUNED_OPTIMIZED_FRONTIER.rst",
            "runs/reusable_frontier_dev_v2/summary.json",
        )
    )
    for i, fixture in enumerate(fixtures):
        fixture["baseline"] = f"runs/reusable_frontier_dev_v2/fixture_{i:02}/fixed"
        paths.update(
            fixture["baseline"] + "/" + name
            for name in ("program.pkl.gz", "selection.json", "audit.json")
        )
    base.mkdir(parents=True)
    write(
        base / "protocol.json",
        {
            "scope": "All eight original five-draw fixtures, reused development only; imported optimized fixed baseline",
            "fixtures": fixtures,
            "baseline": baseline,
            "shared_transitions": 2000000,
            "prefix_transitions": 250000,
            "per_phase_transitions": 50000,
            "capacity": 4096,
            "workers": 2,
            "criterion": "Total exact gain no smaller, positive-source count no smaller, total physical transitions strictly smaller than imported fixed baseline; no population strength claim",
            "inputs": {path: sha(path) for path in sorted(paths)},
        },
    )


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    return p


def worker(base, index):
    p = checked(base)
    fixture = p["fixtures"][index]
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, evidence = read(fixture["source"]), load(fixture["program"])
    worlds = evidence["worlds"]
    parents, opponents, _ = actors()
    parent, opponent = parents[source["parent"]], source["opponent"]
    identity = PARENTS[source["parent"]][1]
    identities = {
        opponent: read("runs/uniform_controller_evaluation/protocol.json")[
            "opponent_identities"
        ][opponent]
    }
    dynamics = KoiTailDynamics(
        p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
    )
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    started = time.perf_counter()
    result = solve_pruned_optimized_frontier(
        worlds,
        dynamics,
        identity,
        identities,
        draws=bound.draws,
        max_decisions=p["shared_transitions"],
        prefix_decisions=p["prefix_transitions"],
        per_phase_decisions=p["per_phase_transitions"],
    )
    seconds = time.perf_counter() - started
    patch = result.patch
    compressed_save(folder / "program.pkl.gz", result)
    row = {
        "fixture": index,
        "decisions": patch.decisions,
        "seconds": seconds,
        "gain_lower": str(patch.gain_lower),
        "frontier_complete": patch.frontier.complete,
        "refusal": patch.frontier.refusal,
        "frontier_nodes": len(patch.frontier.nodes),
        "skipped_nodes": len(result.skipped),
        "attempted_nodes": len(patch.comparisons),
        "adopted_nodes": sum(
            r["selection"].candidate is not None for _, r in patch.comparisons
        ),
        "program_sha256": sha(folder / "program.pkl.gz"),
    }
    write(folder / "selection.json", row)
    print({"planned": row}, flush=True)
    if patch.frontier.complete:
        oracle = verify_frontier_patch(
            worlds, patch, parent, {opponent: opponents[opponent]}, identity
        )
    else:
        if patch.comparisons or patch.gain_lower or result.skipped:
            raise ValueError("Refused prefix cannot expose comparisons")
        oracle = {"exact_increment": "0", "refusal": patch.frontier.refusal}
    execution = assess(source, patch, parent, opponents[opponent], identity)
    write(
        folder / "audit.json",
        {
            "status": "passed",
            "oracle": oracle,
            "execution": execution,
            "selection_sha256": sha(folder / "selection.json"),
        },
    )
    print({"verified": index, "gain": oracle["exact_increment"]}, flush=True)


def run(base):
    p = checked(base)

    def launch(i):
        with (base / f"worker_{i:02}.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scripts.study_pruned_optimized_frontier",
                    "--base",
                    str(base),
                    "--worker",
                    str(i),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        list(pool.map(launch, range(len(p["fixtures"]))))
    rows = []
    for i in range(len(p["fixtures"])):
        folder = base / f"fixture_{i:02}"
        row, audit = read(folder / "selection.json"), read(folder / "audit.json")
        if (
            audit["status"] != "passed"
            or audit["selection_sha256"] != sha(folder / "selection.json")
            or row["program_sha256"] != sha(folder / "program.pkl.gz")
        ):
            raise ValueError("Result identity changed")
        rows.append(
            {
                **row,
                "exact_increment": audit["oracle"]["exact_increment"],
                "audit_sha256": sha(folder / "audit.json"),
            }
        )
    old, new = {}, {}
    for target, group in ((old, p["baseline"]), (new, rows)):
        target.update(
            exact_gain_sum=str(
                sum((Fraction(r["exact_increment"]) for r in group), Fraction(0))
            ),
            transitions=sum(r["decisions"] for r in group),
            positive_sources=sum(Fraction(r["exact_increment"]) > 0 for r in group),
            seconds=sum(r["seconds"] for r in group),
        )
    eligible = (
        Fraction(new["exact_gain_sum"]) >= Fraction(old["exact_gain_sum"])
        and new["positive_sources"] >= old["positive_sources"]
        and new["transitions"] < old["transitions"]
    )
    write(
        base / "summary.json",
        {
            "scope": p["scope"],
            "protocol_sha256": sha(base / "protocol.json"),
            "rows": rows,
            "baseline": old,
            "pruned": new,
            "criterion_met": eligible,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    else:
        worker(args.base, args.worker)
