"""Bind serial timings to saved policies and independently recheck equality."""

import argparse
from pathlib import Path

from scripts.benchmark_certified_frontier import check_pruning
from scripts.benchmark_frontier_receipts import equivalent
from scripts.export_research_extension import read, require, sha
from scripts.run_arbitration_study import write
from scripts.uniform_controller_study import load


def validate_inputs(base):
    protocol, summary = read(base / "protocol.json"), read(base / "summary.json")
    require(
        summary["protocol_sha256"] == sha(base / "protocol.json"), "Protocol changed"
    )
    for path, digest in protocol["inputs"].items():
        require(sha(path) == digest, f"Frozen input changed: {path}")
    return protocol, summary


def receipts(base):
    p, s = validate_inputs(base)
    bindings = {}
    for fixture in range(len(p["fixtures"])):
        for repetition in range(p["repetitions"]):
            programs = {}
            for arm in ("old", "reused"):
                folder = base / f"fixture_{fixture:02}_repeat_{repetition}_{arm}"
                row = read(folder / "result.json")
                require(row in s["rows"], "Timing row differs")
                require(
                    row["program_sha256"] == sha(folder / "program.pkl.gz"),
                    "Program changed",
                )
                programs[arm] = load(folder / "program.pkl.gz")
                require(
                    row["decisions"] == programs[arm].decisions
                    and row["gain_lower"] == str(programs[arm].gain_lower),
                    "Timing policy differs",
                )
                for name in ("result.json", "program.pkl.gz"):
                    bindings[str(folder / name)] = sha(folder / name)
            equivalent(programs["old"], programs["reused"])
    totals = {
        arm: sum(r["seconds"] for r in s["rows"] if r["arm"] == arm)
        for arm in ("old", "reused")
    }
    require(
        totals == s["seconds"]
        and totals["reused"] / totals["old"] == s["runtime_ratio"],
        "Timing sum differs",
    )
    bindings.update(
        {
            str(base / "protocol.json"): sha(base / "protocol.json"),
            str(base / "summary.json"): sha(base / "summary.json"),
        }
    )
    return {
        "scope": p["scope"],
        "fixtures": [f["original_index"] for f in p["fixtures"]],
        "rows": s["rows"],
        "seconds": totals,
        "runtime_ratio": s["runtime_ratio"],
        "all_receipts_equal": True,
        "result_bindings": bindings,
        "limitation": "Net comparison includes the named-program adapter's identity checks as well as receipt reuse; not a pure cache microbenchmark.",
    }


def pruning(base):
    p, s = validate_inputs(base)
    for path, digest in s["result_bindings"].items():
        require(sha(path) == digest, f"Result changed: {path}")
    for family in p["families"]:
        for repetition in range(p["repetitions"]):
            programs = {}
            for arm in ("old", "pruned"):
                folder = base / f"{family}_repeat_{repetition}_{arm}"
                row = read(folder / "selection.json")
                require(
                    row in s["rows"]
                    and row["program_sha256"] == sha(folder / "program.pkl.gz"),
                    "Timing row/program differs",
                )
                programs[arm] = load(folder / "program.pkl.gz")
                patch = programs[arm] if arm == "old" else programs[arm].patch
                require(
                    row["decisions"] == patch.decisions
                    and row["gain_lower"] == str(patch.gain_lower),
                    "Timing policy metadata differs",
                )
                if arm == "pruned":
                    audit = read(folder / "audit.json")
                    require(
                        audit["status"] == "passed"
                        and audit["selection_sha256"] == sha(folder / "selection.json"),
                        "Audit differs",
                    )
                    require(
                        audit["oracle"]["exact_increment"] == "0"
                        and not patch.comparisons,
                        "Unexpected changed policy",
                    )
                    execution = audit["execution"]
                    require(
                        execution["outcomes"]["parent"]
                        == execution["outcomes"]["controller"]
                        and execution["actual_work"] == execution["replay_work"],
                        "Actual policy changed",
                    )
                    require(
                        len(programs[arm].skipped) == len(patch.frontier.nodes) == 1304,
                        "Wrong pruning scope",
                    )
            check_pruning(programs["old"], programs["pruned"])
    totals = {
        family: {
            arm: sum(
                r["seconds"]
                for r in s["rows"]
                if r["family"] == family and r["arm"] == arm
            )
            for arm in ("old", "pruned")
        }
        for family in p["families"]
    }
    ratios = {family: v["pruned"] / v["old"] for family, v in totals.items()}
    require(
        totals == s["seconds"] and ratios == s["runtime_ratios"],
        "Pruning timing sums differ",
    )
    return {
        "scope": p["scope"],
        "rows": s["rows"],
        "seconds": totals,
        "runtime_ratios": ratios,
        "all_unchanged_comparisons_equal": True,
        "result_bindings": {
            **s["result_bindings"],
            str(base / "protocol.json"): sha(base / "protocol.json"),
            str(base / "summary.json"): sha(base / "summary.json"),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    report = {
        "receipt_reuse": receipts(Path("runs/frontier_receipt_benchmark_v1")),
        "certificate_pruning": pruning(Path("runs/certified_frontier_benchmark_v1")),
        "exporter_sha256": sha(__file__),
    }
    write(args.destination, report)
    print(
        {
            "receipt_rows": len(report["receipt_reuse"]["rows"]),
            "pruning_rows": len(report["certificate_pruning"]["rows"]),
            "equality": "passed",
        }
    )
