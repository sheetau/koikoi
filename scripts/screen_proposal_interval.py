"""Fixed development screen of contingent proposal-first interval adoption."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.interval_tail import select_interval
from scripts.proposal_interval_tail import compile_proposal, propose_then_compare
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_interval_tail import PREVIOUS, assess, describe
from scripts.screen_interval_tail import checked as check_interval
from scripts.uniform_controller_study import actors, audit_plan, load
from scripts.verify_interval_tail import verify_receipt

INTERVAL = Path("runs/interval_tail_screen")
DESIGN = {
    "proposal_decisions": 1000000,
    "reference_decisions": 1000000,
    "max_nodes_per_phase": 250000,
    "quantum": 4096,
}
INPUTS = (
    "scripts/proposal_interval_tail.py",
    "scripts/screen_proposal_interval.py",
    "tests/test_proposal_interval_tail.py",
    "docs/PROPOSAL_INTERVAL_SCREEN.rst",
)


def bindings():
    check_interval(INTERVAL)
    if read(INTERVAL / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("Interval predecessor must be complete")
    path = Path("runs/interval_tail_screen_review.json")
    review = read(path)
    if review["status"] != "passed" or len(review["rows"]) != 8:
        raise ValueError("Complete predecessor review required")
    for name, digest in review["bindings"].items():
        if sha(name) != digest:
            raise ValueError("Predecessor evidence changed")
    return {str(p): sha(p) for p in [path, *map(Path, INPUTS)]}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    p = {
        "scope": "All eight previously selected six-draw development fixtures; no replacements.",
        "inputs": bindings(),
        "design": DESIGN,
        "capacity": 4096,
        "workers": 2,
        "sources": str(PREVIOUS),
        "oracle": "Full Python replay for every complete compiled candidate; parent replay for every adoption or newly exact parent.",
        "criterion": "Descriptive completion, certified gains and work; no population-strength inference.",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", p)


def checked(base):
    p = read(base / "protocol.json")
    if (
        p["inputs"] != bindings()
        or p["design"] != DESIGN
        or p["capacity"] != 4096
        or p["workers"] != 2
    ):
        raise ValueError("Frozen proposal screen inputs or budgets changed")
    return p


def describe_result(result):
    candidate, parent = result["candidate_certificate"], result["parent_receipt"]
    phase = result["proposal_phase"]
    selection = select_interval(parent, [] if candidate is None else [candidate])
    if selection != result["selection"]:
        raise ValueError("Saved selection differs from its certificates")
    return {
        "candidate": None if candidate is None else describe(candidate),
        "parent": describe(parent),
        "adopted": selection.candidate is not None,
        "gain_lower": str(selection.gain_lower),
        "decisions": result["decisions"],
        "proposal_work": phase.result.work,
        "proposal_refusal": phase.result.refusal,
    }


def worker(base, index):
    p = checked(base)
    previous = PREVIOUS / f"fixture_{index:02}"
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, old_row = read(previous / "source.json"), read(previous / "selection.json")
    old = load(previous / "program.pkl.gz")
    parents, opponents, _ = actors()
    _, support_work = audit_plan(
        old_row, old, source, opponents, read(PREVIOUS / "protocol.json")
    )
    worlds, parent = old["worlds"], parents[source["parent"]]
    opponent, parent_id = source["opponent"], PARENTS[source["parent"]][1]
    identities = read(PREVIOUS / "protocol.json")["opponent_identities"]
    dyn = KoiTailDynamics(
        p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
    )
    started = time.perf_counter()
    result = propose_then_compare(
        worlds, dyn, parent_id, {opponent: identities[opponent]}, **p["design"]
    )
    seconds = time.perf_counter() - started
    # Recompile from the saved proposal phase, with its full belief binding.
    if (
        compile_proposal(
            result["proposal_phase"],
            worlds,
            dyn,
            parent_id,
            {opponent: identities[opponent]},
        )
        != result["candidate_certificate"]
    ):
        raise ValueError("Compiled candidate differs from the completed proposal")
    row = {
        **describe_result(result),
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
            "candidate": None
            if row["candidate"] is None
            else row["candidate"]["lower"],
            "gain_lower": row["gain_lower"],
            "decisions": row["decisions"],
        },
        flush=True,
    )
    candidate, reference = result["candidate_certificate"], result["parent_receipt"]
    newly_exact = (
        old_row["selection"]["parent_q"] is None and reference.lower == reference.upper
    )
    oracle = {}
    oracle_started = time.perf_counter()
    if candidate is not None:
        oracle["candidate"] = verify_receipt(
            worlds, candidate, parent, {opponent: opponents[opponent]}, parent_id
        )
    if row["adopted"] or newly_exact:
        oracle["parent"] = verify_receipt(
            worlds, reference, parent, {opponent: opponents[opponent]}, parent_id
        )
    oracle_seconds = time.perf_counter() - oracle_started
    assessment = assess(
        source, result["selection"], parent, opponents[opponent], parent_id
    )
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
    write(queue, {"status": "running", "fixtures": 8, "workers": 2})

    def launch(i):
        with (base / f"worker_{i:02}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.screen_proposal_interval",
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
            for f in as_completed([pool.submit(launch, i) for i in range(8)]):
                print({"complete_fixture": f.result()}, flush=True)
        results = [base / f"fixture_{i:02}" / "result.json" for i in range(8)]
        write(
            base / "summary.json",
            {"status": "passed", "results": {str(f): sha(f) for f in results}},
        )
        write(queue, {"status": "complete", "fixtures": 8})
    except Exception as error:
        write(queue, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/proposal_interval_screen")
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
