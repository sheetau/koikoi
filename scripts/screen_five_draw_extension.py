"""Fixed five-draw parent-refusal fixtures for a contingent fallback extension."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.belief_tail import KoiTailDynamics
from scripts.proposal_interval_tail import compile_proposal, propose_then_compare
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_interval_tail import assess
from scripts.screen_proposal_interval import DESIGN, describe_result
from scripts.screen_proposal_interval import checked as check_six
from scripts.select_parent_gap_fixtures import ORIGINAL
from scripts.select_parent_gap_fixtures import checked as check_fixtures
from scripts.uniform_controller_study import actors, audit_plan, load
from scripts.verify_interval_tail import verify_receipt

FIXTURES = Path("runs/five_draw_parent_gap_fixtures.json")
SIX = Path("runs/proposal_interval_screen")
INPUTS = (
    "scripts/screen_five_draw_extension.py",
    "scripts/select_parent_gap_fixtures.py",
    "docs/FIVE_DRAW_FALLBACK_SCREEN.rst",
)


def bindings():
    check_six(SIX)
    if read(SIX / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("Six-draw screen must close first")
    check_fixtures(FIXTURES)
    review_path = Path("runs/proposal_interval_screen_review.json")
    review = read(review_path)
    if review["status"] != "passed" or len(review["rows"]) != 8:
        raise ValueError("Complete predecessor review required")
    for path, digest in review["bindings"].items():
        if sha(path) != digest:
            raise ValueError("Predecessor evidence changed")
    return {str(p): sha(p) for p in [FIXTURES, review_path, *map(Path, INPUTS)]}


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    p = {
        "scope": "Development only; first five-draw parent refusal per population/seat.",
        "inputs": bindings(),
        "fixtures": read(FIXTURES)["fixtures"],
        "design": DESIGN,
        "capacity": 4096,
        "workers": 2,
        "accounting": "Import original fixed-phase work separately; new 1M+1M stages fit the formerly unused 2M phase allowance.",
        "oracle": "Full Python replay for every complete compiled candidate and every adopted/newly exact parent.",
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
        or p["workers"] != 2
    ):
        raise ValueError("Frozen five-draw screen changed")
    return p


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
    result = propose_then_compare(worlds, dyn, parent_id, identities, **p["design"])
    seconds = time.perf_counter() - started
    if (
        compile_proposal(result["proposal_phase"], worlds, dyn, parent_id, identities)
        != result["candidate_certificate"]
    ):
        raise ValueError("Compiled candidate differs")
    imported = fixture["original_fixed_work"]["decisions"]
    if imported > 2000000 or imported + result["decisions"] > 4000000:
        raise ValueError("Extension exceeds the original nominal two-phase allowance")
    row = {
        **describe_result(result),
        "seconds": seconds,
        "imported_first_phase_decisions": imported,
        "nominal_online_decisions": imported + result["decisions"],
        "parent_adapter_calls": dyn.baseline_calls,
        "opponent_adapter_calls": dyn.policy_calls,
    }
    write(folder / "selection.json", row)
    compressed_save(folder / "program.pkl.gz", result)
    print(
        {
            "fixture": index,
            "source": fixture["case"]["index"],
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
    newly_exact = reference.lower == reference.upper
    oracle = {}
    started = time.perf_counter()
    if candidate is not None:
        oracle["candidate"] = verify_receipt(
            worlds, candidate, parent, {opponent: opponents[opponent]}, parent_id
        )
    if row["adopted"] or newly_exact:
        oracle["parent"] = verify_receipt(
            worlds, reference, parent, {opponent: opponents[opponent]}, parent_id
        )
    oracle_seconds = time.perf_counter() - started
    assessment = assess(
        source, result["selection"], parent, opponents[opponent], parent_id
    )
    write(
        folder / "result.json",
        {
            "status": "passed",
            "fixture": index,
            "case": fixture["case"],
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
                    "scripts.screen_five_draw_extension",
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
        "--base", type=Path, default=Path("runs/five_draw_extension_screen")
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
