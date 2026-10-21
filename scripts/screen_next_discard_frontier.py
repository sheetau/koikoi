"""Fixed public next-discard frontier comparison under unchanged budgets."""

import argparse
import hashlib
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from koikoi.layout import DISCARD, PHASE, PLAYER, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.check_frontier_hybrid_integration import compare_patch
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.frontier_policy_patch import FrontierExecutor, solve_frontier_patch
from scripts.next_discard_frontier import next_discard_frontier
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_frontier_patch import checked as check_parent_screen
from scripts.screen_proposal_interval import describe_result
from scripts.select_parent_gap_fixtures import ORIGINAL
from scripts.select_parent_gap_fixtures import checked as check_fixtures
from scripts.uniform_controller_study import actors, audit_plan, load
from scripts.verify_frontier_patch import verify_frontier_patch

FIXTURES = Path("runs/five_draw_parent_gap_fixtures.json")
PARENT_SCREEN = Path("runs/frontier_patch_screen")
REVIEW = Path("runs/frontier_patch_screen_review.json")
DESIGN = {
    "max_decisions": 2000000,
    "prefix_decisions": 250000,
    "per_phase_decisions": 50000,
    "max_nodes_per_phase": 25000,
    "max_rows": 65536,
}
INPUTS = (
    "scripts/next_discard_frontier.py",
    "tests/test_next_discard_frontier.py",
    "scripts/screen_next_discard_frontier.py",
    "docs/NEXT_DISCARD_FRONTIER_SCREEN.rst",
    "scripts/check_frontier_hybrid_integration.py",
)


def bindings():
    check_parent_screen(PARENT_SCREEN)
    check_fixtures(FIXTURES)
    review = read(REVIEW)
    if (
        read(PARENT_SCREEN / "queue.json") != {"status": "complete", "fixtures": 8}
        or review["status"] != "passed"
    ):
        raise ValueError("Complete predecessor required")
    for path, digest in review["bindings"].items():
        if sha(path) != digest:
            raise ValueError("Predecessor evidence changed")
    return {str(p): sha(p) for p in [FIXTURES, REVIEW, *map(Path, INPUTS)]}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    p = {
        "scope": "All eight fixed sources; public next-discard threshold, unchanged budgets.",
        "frontier_rule": "14 minus public root turn at identified-model five-draw gate",
        "inputs": bindings(),
        "fixtures": read(FIXTURES)["fixtures"],
        "design": DESIGN,
        "capacity": 4096,
        "workers": 1,
        "accounting": "Import original fixed work separately; prefix and all local comparisons share 2M new transitions.",
        "oracle": "Independent complete Python prefix; both local values for every adopted patch. Exact root increment by decomposition, not absolute root values.",
        "criterion": "Fixed-fixture feasibility and certified gains; no new population inference.",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    if (
        p["inputs"] != bindings()
        or p["fixtures"] != read(FIXTURES)["fixtures"]
        or p["design"] != DESIGN
        or p["capacity"] != 4096
        or p["workers"] != 1
    ):
        raise ValueError("Frozen next-discard screen changed")
    return p


def assess(source, patch, parent, opponent, parent_id):
    outcomes, actual, replay = {}, Counter(), Counter()
    for label in ("parent", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = FrontierExecutor(patch, parent, parent_id)
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
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    started = time.perf_counter()
    patch = solve_frontier_patch(
        worlds, dyn, parent_id, identities, draws=bound.draws, **p["design"]
    )
    seconds = time.perf_counter() - started
    if (
        not patch.frontier.complete
        or patch.frontier.work["peak_rows"] > bound.maximum_rows
        or patch.frontier.work["decisions"] > bound.maximum_transitions
    ):
        raise ValueError("Derived prefix completion bound failed")
    for _, node in patch.frontier.nodes:
        if not np.all(node.states[:, TURN] == bound.target_turn) or not np.all(
            node.states[:, PHASE] == DISCARD
        ):
            raise ValueError("Frontier is not the next learner discard")
    previous = load(PARENT_SCREEN / folder.name / "program.pkl.gz")
    changed_gate = previous.frontier.draws != bound.draws
    if not changed_gate:
        compare_patch(patch, previous)
    imported = fixture["original_fixed_work"]["decisions"]
    if imported > 2000000 or imported + patch.decisions > 4000000:
        raise ValueError("Nominal integrated work exceeds the original maximum")
    row = {
        "entry_bound": asdict(bound),
        "changed_gate": changed_gate,
        "prefix_complete": patch.frontier.complete,
        "prefix_refusal": patch.frontier.refusal,
        "prefix_work": patch.frontier.work,
        "frontier_nodes": len(patch.frontier.nodes),
        "attempted_nodes": len(patch.comparisons),
        "adopted_nodes": sum(
            r["selection"].candidate is not None for _, r in patch.comparisons
        ),
        "gain_lower": str(patch.gain_lower),
        "decisions": patch.decisions,
        "imported_first_phase_decisions": imported,
        "nominal_online_decisions": imported + patch.decisions,
        "seconds": seconds,
        "parent_adapter_calls": dyn.baseline_calls,
        "opponent_adapter_calls": dyn.policy_calls,
        "local_comparisons": [
            {"key_sha256": hashlib.sha256(key).hexdigest(), **describe_result(result)}
            for key, result in patch.comparisons
        ],
    }
    write(folder / "selection.json", row)
    compressed_save(folder / "program.pkl.gz", patch)
    print(
        {
            "fixture": index,
            "stage": "planned",
            "nodes": row["frontier_nodes"],
            "adopted": row["adopted_nodes"],
            "gain_lower": row["gain_lower"],
            "decisions": patch.decisions,
        },
        flush=True,
    )
    started = time.perf_counter()
    oracle = (
        verify_frontier_patch(
            worlds, patch, parent, {opponent: opponents[opponent]}, parent_id
        )
        if patch.frontier.complete
        else None
    )
    oracle_seconds = time.perf_counter() - started
    assessment = assess(source, patch, parent, opponents[opponent], parent_id)
    write(
        folder / "result.json",
        {
            "status": "passed",
            "fixture": index,
            "case": fixture["case"],
            "protocol_sha256": sha(base / "protocol.json"),
            "selection": row,
            "support_audit_work": support_work,
            "oracle": oracle,
            "oracle_seconds": oracle_seconds,
            "assessment": assessment,
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
    write(queue, {"status": "running", "fixtures": 8, "workers": 1})

    def launch(i):
        with (base / f"worker_{i:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.screen_next_discard_frontier",
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
        with ThreadPoolExecutor(max_workers=1) as pool:
            for f in as_completed([pool.submit(launch, i) for i in range(8)]):
                print({"complete_fixture": f.result()}, flush=True)
        paths = [base / f"fixture_{i:02}" / "result.json" for i in range(8)]
        write(
            base / "summary.json",
            {"status": "passed", "results": {str(f): sha(f) for f in paths}},
        )
        write(queue, {"status": "complete", "fixtures": 8})
    except Exception as error:
        write(queue, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/next_discard_frontier_screen")
    )
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
