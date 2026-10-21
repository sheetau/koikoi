"""Reexecute original phases and validate the integrated frontier fallback."""

import argparse
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.frontier_hybrid_controller import (
    FrontierHybridExecutor,
    evaluate_frontier_hybrid,
)
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_frontier_patch import checked as check_patch_screen
from scripts.select_parent_gap_fixtures import ORIGINAL
from scripts.select_parent_gap_fixtures import checked as check_fixtures
from scripts.uniform_controller_study import actors, audit_plan, load

FIXTURES = Path("runs/five_draw_parent_gap_fixtures.json")
PATCH_SCREEN = Path("runs/frontier_patch_screen")
REVIEW = Path("runs/frontier_patch_screen_review.json")
INPUTS = (
    "scripts/frontier_hybrid_controller.py",
    "tests/test_frontier_hybrid_controller.py",
    "scripts/check_frontier_hybrid_integration.py",
    "docs/FRONTIER_HYBRID_INTEGRATION.rst",
)


def bindings():
    check_patch_screen(PATCH_SCREEN)
    check_fixtures(FIXTURES)
    review = read(REVIEW)
    if (
        read(PATCH_SCREEN / "queue.json") != {"status": "complete", "fixtures": 8}
        or review["status"] != "passed"
    ):
        raise ValueError("Completed independent frontier review required")
    for path, digest in review["bindings"].items():
        if sha(path) != digest:
            raise ValueError("Frontier evidence changed")
    return {str(p): sha(p) for p in [FIXTURES, REVIEW, *map(Path, INPUTS)]}


def preservation_controls():
    found = {}
    for chunk in range(512):
        folder = ORIGINAL / f"chunk_{chunk:04}"
        records = folder / "records.json"
        for row in read(records):
            if row["status"] != "evaluated" or row["selection"]["reason"] not in (
                "strict_exact_gain",
                "no_strict_gain",
            ):
                continue
            parent = int(row["case"]["population"].split(":")[0])
            key = (parent, row["selection"]["reason"])
            if key in found:
                continue
            index = row["case"]["index"]
            source = folder / f"source_{index:05}" / "record.json"
            selection = folder / f"selection_{index:05}.json"
            program = folder / f"program_{index:05}.pkl.gz"
            if (
                sha(selection) != row["selection_sha256"]
                or read(source) != row["source"]
            ):
                raise ValueError("Preservation source binding differs")
            found[key] = {
                "case": row["case"],
                "source": str(source),
                "selection": str(selection),
                "program": str(program),
                "reason": key[1],
                "inputs": {
                    str(p): sha(p) for p in (records, source, selection, program)
                },
                "patch_folder": None,
            }
            if len(found) == 4:
                return list(found.values())
    raise ValueError("Missing preservation control")


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    fixtures = [
        {**f, "patch_folder": str(PATCH_SCREEN / f"fixture_{i:02}")}
        for i, f in enumerate(read(FIXTURES)["fixtures"])
    ] + preservation_controls()
    p = {
        "scope": "Twelve fixed integration/preservation cases; no fresh strength inference",
        "inputs": bindings(),
        "fixtures": fixtures,
        "workers": 2,
        "capacity": 4096,
        "design": {
            "phase_decisions": 2000000,
            "phase_nodes": 500000,
            "patch_draws": 3,
            "prefix_decisions": 250000,
            "local_phase_decisions": 50000,
            "local_nodes": 25000,
            "max_rows": 65536,
        },
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    if p["inputs"] != bindings() or len(p["fixtures"]) != 12:
        raise ValueError("Integration inputs changed")
    for fixture in p["fixtures"]:
        for path, digest in fixture["inputs"].items():
            if sha(path) != digest:
                raise ValueError("Frozen integration fixture changed")
    return p


def compare_patch(new, old):
    for name in (
        "parent_program",
        "total_mass",
        "draws",
        "terminal_mass",
        "pending_mass",
        "work",
        "refusal",
    ):
        if getattr(new.frontier, name) != getattr(old.frontier, name):
            raise ValueError("Integrated prefix differs: " + name)
    if len(new.frontier.nodes) != len(old.frontier.nodes):
        raise ValueError("Integrated prefix node count differs")
    for (key, node), (old_key, old_node) in zip(
        new.frontier.nodes, old.frontier.nodes, strict=True
    ):
        if key != old_key or (node.viewer, node.draws, node.model_names) != (
            old_node.viewer,
            old_node.draws,
            old_node.model_names,
        ):
            raise ValueError("Integrated frontier identity differs")
        for name in ("states", "models", "weights"):
            np.testing.assert_array_equal(getattr(node, name), getattr(old_node, name))
    if (new.comparisons, new.gain_lower, new.decisions) != (
        old.comparisons,
        old.gain_lower,
        old.decisions,
    ):
        raise ValueError("Integrated local programs, values or work differ")


def assess(source, patch, parent, opponent, parent_id):
    outcomes, actual, replay = {}, Counter(), Counter()
    for label in ("parent", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = FrontierHybridExecutor(patch, parent, parent_id)
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] != source["seat"]:
                action = int(opponent.act(env))
            else:
                action = (
                    int(parent.act(env)) if label == "parent" else executor.act(env)
                )
            actions.append(action)
            add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL:
            raise ValueError("Actual tail did not terminate")
        final = env.state.tolist()
        checked_work = replay_actions(
            reference(source["selected_root"]["state"]), actions, final
        )
        if checked_work != work:
            raise ValueError("Actual replay differs")
        actual.update(work)
        replay.update(checked_work)
        outcomes[label] = {"actions": actions, "final": final, "work": work}
    if (
        outcomes["parent"]["final"] != source["final"]
        or outcomes["parent"]["actions"]
        != source["actions"][source["selected_root"]["decision"] :]
    ):
        raise ValueError("Parent no longer reproduces the source suffix")
    if patch.gain_lower == 0 and outcomes["parent"] != outcomes["controller"]:
        raise ValueError("Fallback differs from parent")
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
    }


def worker(base, index):
    p = checked(base)
    fixture = p["fixtures"][index]
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, old_row, old = (
        read(fixture["source"]),
        read(fixture["selection"]),
        load(fixture["program"]),
    )
    parents, opponents, _ = actors()
    original_protocol = read(ORIGINAL / "protocol.json")
    _, support_work = audit_plan(old_row, old, source, opponents, original_protocol)
    worlds, parent = old["worlds"], parents[source["parent"]]
    opponent, parent_id = source["opponent"], PARENTS[source["parent"]][1]
    identities = {opponent: original_protocol["opponent_identities"][opponent]}
    dyn = KoiTailDynamics(
        p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
    )
    started = time.perf_counter()
    hybrid = evaluate_frontier_hybrid(worlds, dyn, parent_id, identities, **p["design"])
    seconds = time.perf_counter() - started
    if (
        hybrid.fixed != old["phases"]["fixed"]
        or hybrid.optimized != old["phases"].get("optimized")
        or hybrid.original != old["selection"]
    ):
        raise ValueError("Original phases or selection changed")
    previous = None
    if fixture["patch_folder"] is not None:
        previous = Path(fixture["patch_folder"])
        if hybrid.patch is None:
            raise ValueError("Expected the parent-refusal patch")
        compare_patch(hybrid.patch, load(previous / "program.pkl.gz"))
    elif hybrid.patch is not None:
        raise ValueError("A completed original decision unexpectedly used a patch")
    compressed_save(folder / "program.pkl.gz", hybrid)
    assessment = assess(source, hybrid, parent, opponents[opponent], parent_id)
    if (
        previous is not None
        and assessment != read(previous / "result.json")["assessment"]
    ):
        raise ValueError("Integrated actual patch suffix differs")
    write(
        folder / "result.json",
        {
            "status": "passed",
            "fixture": index,
            "case": fixture["case"],
            "protocol_sha256": sha(base / "protocol.json"),
            "old_reason": hybrid.original.reason,
            "patched": hybrid.patch is not None,
            "gain_lower": str(hybrid.gain_lower),
            "decisions": hybrid.decisions,
            "fixed_work": hybrid.fixed.result.work,
            "optimized_work": None
            if hybrid.optimized is None
            else hybrid.optimized.result.work,
            "patch_decisions": 0 if hybrid.patch is None else hybrid.patch.decisions,
            "seconds": seconds,
            "support_audit_work": support_work,
            "assessment": assessment,
            "fresh_chance_draws": 0,
            "new_value_oracle_work": 0,
            "reused_value_evidence": fixture["patch_folder"] or fixture["selection"],
            "artifacts": {f.name: sha(f) for f in folder.iterdir() if f.is_file()},
        },
    )
    print(
        {"fixture": index, "stage": "complete", "gain_lower": str(hybrid.gain_lower)},
        flush=True,
    )


def run(base):
    checked(base)
    queue = base / "queue.json"
    if queue.exists():
        raise FileExistsError(queue)
    write(queue, {"status": "running", "fixtures": 12, "workers": 2})

    def launch(i):
        with (base / f"worker_{i:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.check_frontier_hybrid_integration",
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
        with ThreadPoolExecutor(max_workers=2) as pool:
            for f in as_completed([pool.submit(launch, i) for i in range(12)]):
                print({"complete_fixture": f.result()}, flush=True)
        paths = [base / f"fixture_{i:02}" / "result.json" for i in range(12)]
        write(
            base / "summary.json",
            {"status": "passed", "results": {str(f): sha(f) for f in paths}},
        )
        write(queue, {"status": "complete", "fixtures": 12})
    except Exception as error:
        write(queue, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/frontier_hybrid_integration")
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
