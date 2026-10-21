"""Paired development screen of public score bounds on all eight closed fixtures."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.prospective_incumbent import (
    compressed_save,
    describe_selection,
    evaluate_arm,
)
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.score_bound_assessment import assess_pair, verify_pair
from scripts.score_bounded_tail import ScoreBoundPlanner
from scripts.screen_six_draw_gate import checked as check_previous
from scripts.uniform_controller_study import actors, audit_plan, load

PREVIOUS = Path("runs/six_draw_gate_screen")
INPUTS = (
    "scripts/public_score_bounds.py",
    "scripts/score_bounded_tail.py",
    "scripts/score_bound_assessment.py",
    "scripts/verify_score_bounded_tail.py",
    "scripts/screen_score_bounds.py",
    "tests/test_public_score_bounds.py",
    "tests/test_score_bounded_tail.py",
    "docs/PUBLIC_SCORE_BOUNDS.rst",
)


def bindings():
    check_previous(PREVIOUS)
    if read(PREVIOUS / "queue.json")["status"] != "complete":
        raise ValueError("Previous eight-fixture screen must be closed")
    paths = [
        PREVIOUS / name for name in ("protocol.json", "summary.json", "queue.json")
    ]
    summary = read(PREVIOUS / "summary.json")
    for i in range(8):
        folder = PREVIOUS / f"fixture_{i:02}"
        result = folder / "result.json"
        if sha(result) != summary["results"][str(result)]:
            raise ValueError("Previous result changed")
        paths.append(result)
        for name, digest in read(result)["artifacts"].items():
            path = folder / name
            if sha(path) != digest:
                raise ValueError("Previous artifact changed")
            paths.append(path)
    return {str(p): sha(p) for p in paths + list(map(Path, INPUTS))}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    protocol = {
        "scope": "Paired development only; all eight earlier-gate fixtures, no replacements.",
        "previous": str(PREVIOUS),
        "inputs": bindings(),
        "design": read(PREVIOUS / "protocol.json")["design"],
        "criterion": "Describe completed parent values, adoptions and physical work at unchanged caps.",
        "oracle": "Full independent parent/selected replay for any adoption or newly completed parent value.",
        "workers": 2,
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)


def checked(base):
    p = read(base / "protocol.json")
    if p["inputs"] != bindings():
        raise ValueError("Frozen score-bound screen inputs changed")
    if p["design"] != read(PREVIOUS / "protocol.json")["design"]:
        raise ValueError("Screen budgets changed")
    return p


def worker(base, index):
    p = checked(base)
    previous = PREVIOUS / f"fixture_{index:02}"
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source = read(previous / "source.json")
    old_row = read(previous / "selection.json")
    old = load(previous / "program.pkl.gz")
    parents, opponents, _ = actors()
    # Independently reconstruct posterior, compare every world/mass, and recheck
    # the old saved selection before reusing its worlds. No new chance draws.
    _, support_work = audit_plan(
        old_row, old, source, opponents, read(PREVIOUS / "protocol.json")
    )
    worlds = old["worlds"]
    parent = parents[source["parent"]]
    opponent = source["opponent"]
    parent_id = PARENTS[source["parent"]][1]
    identities = read(PREVIOUS / "protocol.json")["opponent_identities"]
    dyn = KoiTailDynamics(
        p["design"]["capacity"],
        worlds.model_names,
        {opponent: opponents[opponent]},
        parent,
    )
    phases, selection, seconds = evaluate_arm(
        worlds,
        dyn,
        parent_id,
        {opponent: identities[opponent]},
        "controller",
        p["design"],
        planner_type=ScoreBoundPlanner,
    )
    common_values = 0
    for name in old["phases"].keys() & phases.keys():
        before, after = old["phases"][name].result.plan, phases[name].result.plan
        if before is not None and after is not None:
            for action in before.values.keys() & after.values.keys():
                if before.values[action] != after.values[action]:
                    raise ValueError("Common complete root value changed")
                common_values += 1
    row = {
        "selection": describe_selection(selection),
        "seconds": seconds,
        "phases": {
            k: {"work": v.result.work, "refusal": v.result.refusal}
            for k, v in phases.items()
        },
        "parent_adapter_calls": dyn.baseline_calls,
        "opponent_adapter_calls": dyn.policy_calls,
        "common_exact_values_checked": common_values,
    }
    write(folder / "selection.json", row)
    compressed_save(
        folder / "program.pkl.gz", {"phases": phases, "selection": selection}
    )
    print(
        {"fixture": index, "stage": "planned", "selection": row["selection"]},
        flush=True,
    )
    newly_valued = (
        old_row["selection"]["parent_q"] is None and selection.parent_q is not None
    )
    check_selection = selection
    oracle_scope = "adopted" if selection.plan is not None else "none"
    if newly_valued and selection.plan is None:
        fixed = phases["fixed"].result.plan
        check_selection = replace(
            selection, plan=fixed, selected_q=fixed.values[fixed.action]
        )
        oracle_scope = "new_parent_and_fixed_candidate"
    oracle = verify_pair(
        worlds, check_selection, parent, {opponent: opponents[opponent]}
    )
    actual = assess_pair(source, selection, parent, opponents[opponent], parent_id)
    write(
        folder / "result.json",
        {
            "status": "passed",
            "fixture": index,
            "source_key": source["source_key"],
            "protocol_sha256": sha(base / "protocol.json"),
            "selection": row,
            "newly_valued_parent": newly_valued,
            "support_audit_work": support_work,
            "oracle_scope": oracle_scope,
            "oracle": oracle,
            "assessment": actual,
            "fresh_chance_draws": 0,
            "artifacts": {f.name: sha(f) for f in folder.iterdir() if f.is_file()},
        },
    )
    print({"fixture": index, "stage": "complete"}, flush=True)


def run(base):
    checked(base)
    queue = base / "queue.json"
    if queue.exists():
        raise FileExistsError(queue)
    write(queue, {"status": "running", "fixtures": 8, "workers": 2})

    def launch(index):
        with (base / f"worker_{index:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.screen_score_bounds",
                    "--base",
                    str(base),
                    "--worker",
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return index

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, i) for i in range(8)]
            for f in as_completed(futures):
                print({"complete_fixture": f.result()}, flush=True)
        results = [base / f"fixture_{i:02}" / "result.json" for i in range(8)]
        write(
            base / "summary.json",
            {
                "status": "passed",
                "fixtures": 8,
                "results": {str(f): sha(f) for f in results},
            },
        )
        write(queue, {"status": "complete", "fixtures": 8})
    except Exception as error:
        write(queue, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/score_bound_screen"))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", type=int, choices=range(8))
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    else:
        worker(args.base, args.worker)
