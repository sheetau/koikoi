"""Execute changed boundaries and explicitly reuse identical integrated controls."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.check_frontier_hybrid_integration import assess, compare_patch
from scripts.check_frontier_hybrid_integration import checked as check_integrated
from scripts.frontier_hybrid_controller import FrontierHybridExecutor
from scripts.next_discard_frontier import next_discard_frontier
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_next_discard_frontier import checked as check_screen
from scripts.select_parent_gap_fixtures import ORIGINAL
from scripts.selected_frontier_controller import (
    configuration,
    evaluate_selected_frontier,
)
from scripts.uniform_controller_study import actors, audit_plan, load

INTEGRATED = Path("runs/frontier_hybrid_integration")
SCREEN = Path("runs/next_discard_frontier_screen")
REVIEWS = (
    Path("runs/frontier_hybrid_integration_review.json"),
    Path("runs/next_discard_frontier_screen_review.json"),
)
INPUTS = (
    "scripts/selected_frontier_controller.py",
    "scripts/check_selected_frontier_integration.py",
    "docs/SELECTED_FRONTIER_INTEGRATION.rst",
)


def bindings():
    check_integrated(INTEGRATED)
    check_screen(SCREEN)
    for path in REVIEWS:
        review = read(path)
        if review["status"] != "passed":
            raise ValueError("Complete predecessor reviews required")
        for name, digest in review["bindings"].items():
            if sha(name) != digest:
                raise ValueError("Predecessor evidence changed")
    return {str(p): sha(p) for p in [*REVIEWS, *map(Path, INPUTS)]}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    fixtures = []
    old_protocol = read(INTEGRATED / "protocol.json")
    if {
        k: v for k, v in old_protocol["design"].items() if k != "patch_draws"
    } != configuration():
        raise ValueError("Selected controller changes other planning settings")
    for i, old_fixture in enumerate(old_protocol["fixtures"]):
        worlds = load(old_fixture["program"])["worlds"]
        bound = next_discard_frontier(
            worlds.states[0], worlds.viewer, len(worlds.model_names)
        )
        old = load(INTEGRATED / f"fixture_{i:02}" / "program.pkl.gz")
        changed = old.patch is not None and old.patch.frontier.draws != bound.draws
        fixtures.append(
            {
                **old_fixture,
                "mode": "execute" if changed else "reuse",
                "entry_bound": asdict(bound),
            }
        )
    if [i for i, f in enumerate(fixtures) if f["mode"] == "execute"] != [0, 1]:
        raise ValueError("Unexpected changed integration cases")
    p = {
        "scope": "Two changed executions and ten explicit identical-program reuses",
        "inputs": bindings(),
        "design": configuration(),
        "fixtures": fixtures,
        "capacity": 4096,
        "workers": 2,
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    if (
        p["inputs"] != bindings()
        or p["design"] != configuration()
        or len(p["fixtures"]) != 12
    ):
        raise ValueError("Selected integration inputs changed")
    return p


def worker(base, index):
    p = checked(base)
    fixture = p["fixtures"][index]
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, old_row, original = (
        read(fixture["source"]),
        read(fixture["selection"]),
        load(fixture["program"]),
    )
    worlds = original["worlds"]
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    if worlds.draws != 5 or asdict(bound) != fixture["entry_bound"]:
        raise ValueError("Selected public configuration differs")
    old_folder = INTEGRATED / folder.name
    previous = load(old_folder / "program.pkl.gz")
    changed = (
        previous.patch is not None and previous.patch.frontier.draws != bound.draws
    )
    if fixture["mode"] != ("execute" if changed else "reuse"):
        raise ValueError("Execution/reuse classification changed")
    support_work, seconds = {}, 0.0
    if changed:
        parents, opponents, _ = actors()
        _, support_work = audit_plan(
            old_row, original, source, opponents, read(ORIGINAL / "protocol.json")
        )
        parent, opponent = parents[source["parent"]], source["opponent"]
        parent_id = PARENTS[source["parent"]][1]
        identities = dict(previous.fixed.binding.opponent_identities)
        dyn = KoiTailDynamics(
            p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
        )
        started = time.perf_counter()
        result = evaluate_selected_frontier(worlds, dyn, parent_id, identities)
        seconds = time.perf_counter() - started
    else:
        result = previous
    if (result.fixed, result.optimized, result.original) != (
        previous.fixed,
        previous.optimized,
        previous.original,
    ):
        raise ValueError("Original phases changed in selected wrapper")
    FrontierHybridExecutor(result, None, result.fixed.binding.parent_identity)
    if result.patch is not None:
        compare_patch(result.patch, load(SCREEN / folder.name / "program.pkl.gz"))
    compressed_save(folder / "program.pkl.gz", result)
    if changed:
        assessment = assess(source, result, parent, opponents[opponent], parent_id)
        if assessment != read(SCREEN / folder.name / "result.json")["assessment"]:
            raise ValueError("Selected actual suffix differs")
    else:
        assessment = read(old_folder / "result.json")["assessment"]
    row = {
        "status": "passed",
        "fixture": index,
        "case": fixture["case"],
        "mode": fixture["mode"],
        "entry_bound": asdict(bound),
        "protocol_sha256": sha(base / "protocol.json"),
        "gain_lower": str(result.gain_lower),
        "semantic_decisions": result.decisions,
        "new_planning_decisions": result.decisions if changed else 0,
        "new_support_work": support_work,
        "seconds": seconds,
        "new_actual_work": assessment["actual_work"] if changed else {},
        "new_replay_work": assessment["replay_work"] if changed else {},
        "assessment": assessment,
        "evidence_mode": "new execution; identical values reused"
        if changed
        else "identical integrated program and execution reused",
        "fresh_chance_draws": 0,
        "new_value_oracle_work": 0,
        "artifacts": {"program.pkl.gz": sha(folder / "program.pkl.gz")},
    }
    write(folder / "result.json", row)
    print({"fixture": index, "mode": fixture["mode"], "stage": "complete"}, flush=True)


def run(base):
    p = checked(base)
    if (base / "queue.json").exists():
        raise FileExistsError(base / "queue.json")
    write(base / "queue.json", {"status": "running", "fixtures": 12, "workers": 2})

    def launch(i):
        with (base / f"worker_{i:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.check_selected_frontier_integration",
                    "--base",
                    str(base),
                    "--worker",
                    str(i),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return i

    try:
        with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
            for job in as_completed([pool.submit(launch, i) for i in range(12)]):
                print({"complete_fixture": job.result()}, flush=True)
        paths = [base / f"fixture_{i:02}" / "result.json" for i in range(12)]
        write(
            base / "summary.json",
            {"status": "passed", "results": {str(f): sha(f) for f in paths}},
        )
        write(base / "queue.json", {"status": "complete", "fixtures": 12})
    except Exception as error:
        write(base / "queue.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/selected_frontier_integration")
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", type=int, choices=range(12))
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    else:
        worker(args.base, args.worker)
