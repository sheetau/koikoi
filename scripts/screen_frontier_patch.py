"""Fixed original-gate fixtures for public-frontier local policy patches."""

import argparse
import hashlib
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.frontier_policy_patch import FrontierExecutor, solve_frontier_patch
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_five_draw_extension import checked as check_parent_screen
from scripts.screen_proposal_interval import describe_result
from scripts.select_parent_gap_fixtures import ORIGINAL
from scripts.select_parent_gap_fixtures import checked as check_fixtures
from scripts.uniform_controller_study import actors, audit_plan, load
from scripts.verify_frontier_patch import verify_frontier_patch

FIXTURES = Path("runs/five_draw_parent_gap_fixtures.json")
PARENT_SCREEN = Path("runs/five_draw_extension_screen")
DESIGN = {
    "draws": 3,
    "max_decisions": 2000000,
    "prefix_decisions": 250000,
    "per_phase_decisions": 50000,
    "max_nodes_per_phase": 25000,
    "max_rows": 65536,
}
INPUTS = (
    "scripts/frontier_policy_patch.py",
    "scripts/verify_frontier_patch.py",
    "scripts/screen_frontier_patch.py",
    "tests/test_frontier_policy_patch.py",
    "docs/FRONTIER_POLICY_PATCH.rst",
    "docs/FRONTIER_PATCH_SCREEN.rst",
)


def bindings():
    check_parent_screen(PARENT_SCREEN)
    check_fixtures(FIXTURES)
    return {
        str(p): sha(p)
        for p in [FIXTURES, PARENT_SCREEN / "protocol.json", *map(Path, INPUTS)]
    }


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    p = {
        "scope": "Development only; all eight fixed five-draw parent-refusal sources.",
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
        raise ValueError("Frozen frontier-patch screen changed")
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
    started = time.perf_counter()
    patch = solve_frontier_patch(worlds, dyn, parent_id, identities, **p["design"])
    seconds = time.perf_counter() - started
    imported = fixture["original_fixed_work"]["decisions"]
    if imported > 2000000 or imported + patch.decisions > 4000000:
        raise ValueError("Nominal integrated work exceeds the original maximum")
    row = {
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
                    "scripts.screen_frontier_patch",
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
    parser.add_argument("--base", type=Path, default=Path("runs/frontier_patch_screen"))
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
