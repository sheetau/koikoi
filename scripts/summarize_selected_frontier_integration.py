"""Audit two new executions separately from ten unchanged evidence reuses."""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts.check_frontier_hybrid_integration import compare_patch
from scripts.check_selected_frontier_integration import INTEGRATED, SCREEN, checked
from scripts.frontier_hybrid_controller import FrontierHybridExecutor
from scripts.next_discard_frontier import next_discard_frontier
from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_study import load


def summarize(base, output):
    if output.exists():
        raise FileExistsError(output)
    p = checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 12}:
        raise ValueError("Complete selected integration required")
    summary = read(base / "summary.json")
    expected = {str(base / f"fixture_{i:02}" / "result.json") for i in range(12)}
    if summary["status"] != "passed" or set(summary["results"]) != expected:
        raise ValueError("Incomplete selected integration")
    totals = {k: Counter() for k in ("support", "actual", "replay")}
    rows, bindings, planning = [], {}, 0
    for i, fixture in enumerate(p["fixtures"]):
        folder = base / f"fixture_{i:02}"
        path = folder / "result.json"
        row = read(path)
        worlds = load(fixture["program"])["worlds"]
        bound = next_discard_frontier(
            worlds.states[0], worlds.viewer, len(worlds.model_names)
        )
        previous = load(INTEGRATED / folder.name / "program.pkl.gz")
        result = load(folder / "program.pkl.gz")
        changed = (
            previous.patch is not None and previous.patch.frontier.draws != bound.draws
        )
        if (
            summary["results"][str(path)] != sha(path)
            or row["status"] != "passed"
            or row["case"] != fixture["case"]
            or row["fixture"] != i
            or row["mode"] != fixture["mode"]
            or row["mode"] != ("execute" if changed else "reuse")
            or row["entry_bound"] != asdict(bound)
            or row["entry_bound"] != fixture["entry_bound"]
            or row["protocol_sha256"] != sha(base / "protocol.json")
            or row["artifacts"] != {"program.pkl.gz": sha(folder / "program.pkl.gz")}
            or row["fresh_chance_draws"] != 0
            or row["new_value_oracle_work"] != 0
            or row["semantic_decisions"] != result.decisions
            or row["new_planning_decisions"] != (result.decisions if changed else 0)
            or row["gain_lower"] != str(result.gain_lower)
            or (result.fixed, result.optimized, result.original)
            != (previous.fixed, previous.optimized, previous.original)
        ):
            raise ValueError("Selected integration binding or work differs")
        FrontierHybridExecutor(result, None, result.fixed.binding.parent_identity)
        if result.patch is not None:
            compare_patch(result.patch, load(SCREEN / folder.name / "program.pkl.gz"))
        evidence = SCREEN if changed else INTEGRATED
        if (
            row["assessment"]
            != read(evidence / folder.name / "result.json")["assessment"]
        ):
            raise ValueError("Execution differs from independently checked program")
        for label in ("actual", "replay"):
            work = row[f"new_{label}_work"]
            if work != (row["assessment"][f"{label}_work"] if changed else {}):
                raise ValueError("Reused execution counted as new work")
            totals[label].update(work)
        if not changed and row["new_support_work"]:
            raise ValueError("Reused support counted as new work")
        totals["support"].update(row["new_support_work"])
        planning += row["new_planning_decisions"]
        rows.append(
            {
                k: row[k]
                for k in (
                    "fixture",
                    "case",
                    "mode",
                    "gain_lower",
                    "new_planning_decisions",
                    "evidence_mode",
                )
            }
        )
        bindings[str(path)] = sha(path)
    if [r["fixture"] for r in rows if r["mode"] == "execute"] != [0, 1]:
        raise ValueError("New execution cohort differs")
    for name in ("protocol.json", "queue.json", "summary.json"):
        bindings[str(base / name)] = sha(base / name)
    result = {
        "status": "passed",
        "scope": "Two new executions; ten identical-program evidence reuses; no fresh sources",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "rows": rows,
        "new_planning_decisions": planning,
        "new_work": {k: dict(v) for k, v in totals.items()},
        "new_value_oracle_work": 0,
        "fresh_sources": 0,
    }
    write(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/selected_frontier_integration")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
