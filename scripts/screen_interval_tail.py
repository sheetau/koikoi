"""Fixed eight-source development screen of shared-budget interval comparisons."""

import argparse
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
from scripts.interval_tail import (
    IntervalExecutor,
    compare_programs,
    make_program,
    select_interval,
)
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_score_bounds import PREVIOUS
from scripts.screen_score_bounds import checked as check_score_screen
from scripts.uniform_controller_study import actors, audit_plan, load
from scripts.verify_interval_tail import verify_receipt

SCORE_SCREEN = Path("runs/score_bound_screen")
INPUTS = (
    "scripts/interval_tail.py",
    "scripts/verify_interval_tail.py",
    "scripts/screen_interval_tail.py",
    "tests/test_interval_tail.py",
    "docs/INTERVAL_TAIL.rst",
)


def bindings():
    check_score_screen(SCORE_SCREEN)
    if read(SCORE_SCREEN / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("Predecessor screen must be closed")
    review_path = Path("runs/score_bound_screen_review.json")
    review = read(review_path)
    if review["status"] != "passed" or len(review["rows"]) != 8:
        raise ValueError("Complete predecessor artifact review required")
    for path, digest in review["bindings"].items():
        if sha(path) != digest:
            raise ValueError("Predecessor binding changed")
    return {str(p): sha(p) for p in [review_path, *map(Path, INPUTS)]}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    p = {
        "scope": "All eight closed six-draw sources; development only, no replacements.",
        "inputs": bindings(),
        "previous": str(PREVIOUS),
        "design": {
            "max_decisions": 2000000,
            "max_nodes": 500000,
            "quantum": 4096,
            "max_frontier_rows": 65536,
            "capacity": 4096,
            "workers": 2,
        },
        "programs": "Unchanged parent plus each other legal root action, parent default thereafter.",
        "budget": "One shared two-million-transition budget for all programs, including parent.",
        "oracle": "Full independent reference replay for every adoption and newly exact parent.",
        "criterion": "Descriptive strict certified gains and mass/work; no strength inference.",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    if p["inputs"] != bindings():
        raise ValueError("Frozen interval-screen input changed")
    return p


def describe(receipt):
    return {
        "parent": receipt.program.is_parent,
        "root_action": dict(receipt.program.overrides)[
            receipt.program.binding.root_key
        ],
        "overrides": len(receipt.program.overrides),
        "lower": str(receipt.lower),
        "upper": str(receipt.upper),
        "total_mass": receipt.total_mass,
        "resolved_mass": receipt.resolved_mass,
        "resolved_signed": receipt.resolved_signed,
        "work": receipt.work,
        "refusal": receipt.refusal,
    }


def assess(source, selection, parent, opponent, parent_id):
    outcomes, actual, replay = {}, Counter(), Counter()
    for label in ("parent", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = IntervalExecutor(selection, parent, parent_id)
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
    if selection.candidate is None and outcomes["parent"] != outcomes["controller"]:
        raise ValueError("Fallback differs from parent")
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
    }


def worker(base, index):
    p = checked(base)
    old_folder = PREVIOUS / f"fixture_{index:02}"
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source = read(old_folder / "source.json")
    old_row, old = (
        read(old_folder / "selection.json"),
        load(old_folder / "program.pkl.gz"),
    )
    parents, opponents, _ = actors()
    _, support_work = audit_plan(
        old_row, old, source, opponents, read(PREVIOUS / "protocol.json")
    )
    worlds, parent = old["worlds"], parents[source["parent"]]
    opponent, parent_id = source["opponent"], PARENTS[source["parent"]][1]
    identities = read(PREVIOUS / "protocol.json")["opponent_identities"]
    dyn = KoiTailDynamics(
        p["design"]["capacity"],
        worlds.model_names,
        {opponent: opponents[opponent]},
        parent,
    )
    original = make_program(worlds, dyn, parent_id, {opponent: identities[opponent]})
    programs = [original] + [
        make_program(
            worlds,
            dyn,
            parent_id,
            {opponent: identities[opponent]},
            {original.binding.root_key: action},
        )
        for action in original.binding.legal
        if action != original.parent_root_action
    ]
    started = time.perf_counter()
    result = compare_programs(
        worlds,
        dyn,
        programs,
        **{
            k: p["design"][k]
            for k in ("max_decisions", "max_nodes", "quantum", "max_frontier_rows")
        },
    )
    seconds = time.perf_counter() - started
    selection, receipts = result["selection"], result["receipts"]
    if selection != select_interval(receipts[0], receipts[1:]):
        raise ValueError("Selection differs from interval receipts")
    row = {
        "receipts": list(map(describe, receipts)),
        "selected_action": None
        if selection.candidate is None
        else dict(selection.candidate.program.overrides)[original.binding.root_key],
        "gain_lower": str(selection.gain_lower),
        "reason": result["reason"],
        "first_adoption": result["first_adoption"],
        "decisions": result["decisions"],
        "expanded_tasks": result["expanded_tasks"],
        "seconds": seconds,
        "parent_adapter_calls": dyn.baseline_calls,
        "opponent_adapter_calls": dyn.policy_calls,
    }
    write(folder / "selection.json", row)
    compressed_save(folder / "program.pkl.gz", result)
    print(
        {
            "fixture": index,
            "stage": "planned",
            "gain_lower": row["gain_lower"],
            "decisions": row["decisions"],
        },
        flush=True,
    )
    newly_exact = (
        old_row["selection"]["parent_q"] is None
        and receipts[0].lower == receipts[0].upper
    )
    oracle = {}
    if selection.candidate is not None or newly_exact:
        oracle["parent"] = verify_receipt(
            worlds, receipts[0], parent, {opponent: opponents[opponent]}, parent_id
        )
    if selection.candidate is not None:
        oracle["candidate"] = verify_receipt(
            worlds,
            selection.candidate,
            parent,
            {opponent: opponents[opponent]},
            parent_id,
        )
    assessment = assess(source, selection, parent, opponents[opponent], parent_id)
    write(
        folder / "result.json",
        {
            "status": "passed",
            "fixture": index,
            "protocol_sha256": sha(base / "protocol.json"),
            "selection": row,
            "support_audit_work": support_work,
            "newly_exact_parent": newly_exact,
            "oracle": oracle,
            "assessment": assessment,
            "fresh_chance_draws": 0,
            "artifacts": {f.name: sha(f) for f in folder.iterdir() if f.is_file()},
        },
    )
    print({"fixture": index, "stage": "complete"}, flush=True)


def run(base):
    p = checked(base)
    queue = base / "queue.json"
    if queue.exists():
        raise FileExistsError(queue)
    write(
        queue, {"status": "running", "fixtures": 8, "workers": p["design"]["workers"]}
    )

    def launch(i):
        with (base / f"worker_{i:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.screen_interval_tail",
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
        with ThreadPoolExecutor(max_workers=p["design"]["workers"]) as pool:
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
    parser.add_argument("--base", type=Path, default=Path("runs/interval_tail_screen"))
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
