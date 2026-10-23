"""Frozen eight-fixture development comparison; never a population strength test."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.frontier_policy_patch import solve_frontier_patch
from scripts.next_discard_frontier import next_discard_frontier
from scripts.prospective_incumbent import compressed_save
from scripts.reusable_frontier import solve_reusable_frontier
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_next_discard_frontier import assess
from scripts.uniform_controller_study import actors, load
from scripts.verify_frontier_patch import verify_frontier_patch

ARMS = ("fixed", "adaptive_uncached", "adaptive_cached")
INPUTS = (
    "scripts/reusable_frontier.py",
    "scripts/study_reusable_frontier.py",
    "tests/test_reusable_frontier.py",
)


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    fixtures = read("runs/five_draw_parent_gap_fixtures.json")["fixtures"]
    paths = set(INPUTS)
    for f in fixtures:
        for path, digest in f["inputs"].items():
            if sha(path) != digest:
                raise ValueError("Original fixture changed")
            paths.add(path)
    # Bind all Python dependencies in the search package: bounded directory,
    # no old reports or run inventory is needed and no imported claim is rerun.
    paths.update(str(p) for p in Path("scripts").glob("*.py"))
    paths.update(str(p) for p in Path("koikoi").glob("*.py"))
    paths.update(str(p) for p in Path("koikoi/native").glob("*.*"))
    paths.add("runs/uniform_controller_evaluation/protocol.json")
    paths.update(p for p, _ in PARENTS)
    protocol = {
        "scope": "development only; all eight earlier fixtures retained",
        "arms": ARMS,
        "fixtures": fixtures,
        "inputs": {p: sha(p) for p in sorted(paths)},
        "shared_transitions": 2000000,
        "prefix_transitions": 250000,
        "pilot_transitions": 2000,
        "local_transitions": 100000,
        "cache_payload_bytes": 67108864,
        "capacity": 4096,
        "criterion": "paired exact gains, coverage, physical work and elapsed time; no population inference",
        "selection": "choose adaptive only if summed exact gain exceeds fixed with no fewer positive fixtures; otherwise preserve fixed",
        "oracle": "independent Python prefix and all adopted local pairs; actual suffix replay",
        "workers": 2,
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return p


def worker(base, index, arm):
    p = checked(base)
    f = p["fixtures"][index]
    folder = base / f"fixture_{index:02}" / arm
    folder.mkdir(parents=True)
    source, old = read(f["source"]), load(f["program"])
    worlds = old["worlds"]
    parents, opponents, _ = actors()
    parent, opponent = parents[source["parent"]], source["opponent"]
    parent_id = PARENTS[source["parent"]][1]
    identities = {
        opponent: read("runs/uniform_controller_evaluation/protocol.json")[
            "opponent_identities"
        ][opponent]
    }
    dyn = KoiTailDynamics(
        p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
    )
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    common = {
        "draws": bound.draws,
        "max_decisions": p["shared_transitions"],
        "prefix_decisions": p["prefix_transitions"],
    }
    started = time.perf_counter()
    patch = (
        solve_frontier_patch(worlds, dyn, parent_id, identities, **common)
        if arm == "fixed"
        else solve_reusable_frontier(
            worlds,
            dyn,
            parent_id,
            identities,
            **common,
            pilot_decisions=p["pilot_transitions"],
            local_decisions=p["local_transitions"],
            cache_bytes=p["cache_payload_bytes"] if arm == "adaptive_cached" else 0,
        )
    )
    seconds = time.perf_counter() - started
    compressed_save(folder / "program.pkl.gz", patch)
    row = {
        "fixture": index,
        "source": f["case"],
        "arm": arm,
        "gain_lower": str(patch.gain_lower),
        "decisions": patch.decisions,
        "seconds": seconds,
        "frontier_nodes": len(patch.frontier.nodes),
        "attempted_nodes": len(patch.comparisons),
        "adopted_nodes": sum(
            r["selection"].candidate is not None for _, r in patch.comparisons
        ),
        "cache_hits": sum(
            r.get("proposal_work", {}).get("completed_choice_hits", 0)
            for _, r in patch.comparisons
        ),
        "reused_transitions": sum(
            r.get("proposal_work", {}).get("reused_choice_transitions", 0)
            for _, r in patch.comparisons
        ),
        "cache_payload_peak": getattr(patch, "cache_peak_bytes", 0),
        "program_sha256": sha(folder / "program.pkl.gz"),
    }
    write(folder / "selection.json", row)
    print({"planned": row}, flush=True)
    oracle = verify_frontier_patch(
        worlds, patch, parent, {opponent: opponents[opponent]}, parent_id
    )
    execution = assess(source, patch, parent, opponents[opponent], parent_id)
    write(
        folder / "audit.json",
        {
            "status": "passed",
            "oracle": oracle,
            "execution": execution,
            "selection_sha256": sha(folder / "selection.json"),
        },
    )
    print(
        {"verified": index, "arm": arm, "exact_increment": oracle["exact_increment"]},
        flush=True,
    )


def run(base):
    p = checked(base)

    def launch(task):
        i, arm = task
        folder = base / f"fixture_{i:02}" / arm
        if (folder / "audit.json").exists():
            return
        log = base / f"worker_{i:02}_{arm}.log"
        with log.open("x", encoding="utf-8", newline="\n") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scripts.study_reusable_frontier",
                    "--base",
                    str(base),
                    "--worker",
                    str(i),
                    "--arm",
                    arm,
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        list(pool.map(launch, [(i, arm) for i in range(8) for arm in ARMS]))
    summarize(base)


def summarize(base):
    checked(base)
    rows, totals = [], {}
    for arm in ARMS:
        group = []
        for i in range(8):
            folder = base / f"fixture_{i:02}" / arm
            row, audit = read(folder / "selection.json"), read(folder / "audit.json")
            if (
                audit["status"] != "passed"
                or audit["selection_sha256"] != sha(folder / "selection.json")
                or row["program_sha256"] != sha(folder / "program.pkl.gz")
            ):
                raise ValueError("Study result binding differs")
            row["exact_increment"] = audit["oracle"]["exact_increment"]
            group.append(row)
        rows.extend(group)
        totals[arm] = {
            k: sum(r[k] for r in group)
            for k in (
                "decisions",
                "seconds",
                "adopted_nodes",
                "cache_hits",
                "reused_transitions",
            )
        }
        totals[arm]["sum_exact_increment"] = str(
            sum((Fraction(r["exact_increment"]) for r in group), Fraction(0))
        )
        totals[arm]["positive_fixtures"] = sum(
            Fraction(r["exact_increment"]) > 0 for r in group
        )
    baseline = totals["fixed"]
    selected = [
        a
        for a in ARMS[1:]
        if Fraction(totals[a]["sum_exact_increment"])
        > Fraction(baseline["sum_exact_increment"])
        and totals[a]["positive_fixtures"] >= baseline["positive_fixtures"]
    ]
    write(
        base / "summary.json",
        {
            "scope": "development fixtures only",
            "rows": rows,
            "totals": totals,
            "eligible_adaptive": selected,
            "protocol_sha256": sha(base / "protocol.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    mode.add_argument("--worker", type=int, choices=range(8))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    elif args.summary:
        summarize(args.base)
    else:
        worker(args.base, args.worker, args.arm)
