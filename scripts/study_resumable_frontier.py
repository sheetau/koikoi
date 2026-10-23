"""Evaluate parent-default proposals at five- and six-draw development roots."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from koikoi.layout import TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.prospective_incumbent import compressed_save
from scripts.resumable_frontier import solve_resumable_frontier
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_next_discard_frontier import assess
from scripts.uniform_controller_study import actors, load
from scripts.verify_frontier_patch import verify_frontier_patch


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    five = read("runs/five_draw_parent_gap_fixtures.json")["fixtures"]
    fixtures = [
        {
            "draws": 5,
            "source": f["source"],
            "program": f["program"],
            "baseline": f"runs/next_discard_frontier_screen/fixture_{i:02}/selection.json",
        }
        for i, f in enumerate(five)
    ]
    fixtures += [
        {
            "draws": 6,
            "source": f"runs/six_draw_gate_screen/fixture_{i:02}/source.json",
            "program": f"runs/six_draw_gate_screen/fixture_{i:02}/program.pkl.gz",
            "baseline": f"runs/six_draw_gate_screen/fixture_{i:02}/result.json",
        }
        for i in range(8)
    ]
    paths = set(read("runs/reusable_frontier_dev_v2/protocol.json")["inputs"])
    paths.update(
        (__file__, "scripts/resumable_frontier.py", "tests/test_resumable_frontier.py")
    )
    paths.update(f[k] for f in fixtures for k in ("source", "program", "baseline"))
    protocol = {
        "scope": "all eight original five-draw development fixtures and all eight earlier six-draw probes; no fresh strength evidence",
        "fixtures": fixtures,
        "inputs": {p: sha(p) for p in sorted(paths)},
        "boundary": "first learner decision at or below 14 minus public root turn remaining draws",
        "shared_transitions": 2000000,
        "prefix_transitions": 250000,
        "local_transitions": 100000,
        "capacity": 4096,
        "workers": 2,
        "criterion": "independent positive gains in previously uncovered six-draw sources, plus comparison with five-draw fixed search",
        "oracle": "complete Python prefix and adopted local values; actual suffix replay; refusals retained",
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    return p


def worker(base, index):
    p = checked(base)
    f = p["fixtures"][index]
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, evidence = read(f["source"]), load(f["program"])
    worlds = evidence["worlds"]
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
    started = time.perf_counter()
    patch = solve_resumable_frontier(
        worlds,
        dyn,
        parent_id,
        identities,
        draws=14 - int(worlds.states[0, TURN]),
        max_decisions=p["shared_transitions"],
        prefix_decisions=p["prefix_transitions"],
        local_decisions=p["local_transitions"],
    )
    seconds = time.perf_counter() - started
    compressed_save(folder / "program.pkl.gz", patch)
    row = {
        "fixture": index,
        "draws": f["draws"],
        "decisions": patch.decisions,
        "seconds": seconds,
        "gain_lower": str(patch.gain_lower),
        "frontier_complete": patch.frontier.complete,
        "refusal": patch.frontier.refusal,
        "frontier_nodes": len(patch.frontier.nodes),
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
            worlds, patch, parent, {opponent: opponents[opponent]}, parent_id
        )
    else:
        if patch.comparisons or patch.gain_lower:
            raise ValueError("Refused prefix cannot adopt a policy")
        oracle = {"exact_increment": "0", "refusal": patch.frontier.refusal}
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
    print({"verified": index, "gain": oracle["exact_increment"]}, flush=True)


def run(base):
    p = checked(base)

    def launch(index):
        with (base / f"worker_{index:02}.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scripts.study_resumable_frontier",
                    "--base",
                    str(base),
                    "--worker",
                    str(index),
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
        if audit["selection_sha256"] != sha(folder / "selection.json") or row[
            "program_sha256"
        ] != sha(folder / "program.pkl.gz"):
            raise ValueError("Result identity mismatch")
        row["exact_increment"] = audit["oracle"]["exact_increment"]
        rows.append(row)
    write(
        base / "summary.json",
        {
            "scope": p["scope"],
            "rows": rows,
            "protocol_sha256": sha(base / "protocol.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run(args.base)
    else:
        worker(args.base, args.worker)
