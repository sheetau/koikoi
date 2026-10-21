"""Complete-only artifact review and paired accounting for the score-bound screen."""

import argparse
from collections import Counter
from pathlib import Path

from scripts.incumbent_tail_controller import select_improvement
from scripts.prospective_incumbent import describe_selection
from scripts.run_arbitration_study import read, sha, write
from scripts.screen_score_bounds import PREVIOUS, checked
from scripts.uniform_controller_study import load


def summarize(base, output):
    if output.exists():
        raise FileExistsError(output)
    checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("All eight fixtures must finish")
    summary = read(base / "summary.json")
    if summary["status"] != "passed" or summary["fixtures"] != 8:
        raise ValueError("Complete successful screen required")
    totals = {
        k: Counter()
        for k in (
            "old_planning",
            "new_planning",
            "support_audit",
            "actual",
            "replay",
            "oracle_physical",
            "oracle_represented",
        )
    }
    peaks = {"old_planning": 0, "new_planning": 0}
    rows, bindings = [], {}
    for i in range(8):
        folder = base / f"fixture_{i:02}"
        result_path = folder / "result.json"
        if sha(result_path) != summary["results"][str(result_path)]:
            raise ValueError("Result binding differs")
        result = read(result_path)
        if (
            result["status"] != "passed"
            or result["fixture"] != i
            or result["fresh_chance_draws"] != 0
            or result["protocol_sha256"] != sha(base / "protocol.json")
        ):
            raise ValueError("Fixture identity or completion differs")
        if set(result["artifacts"]) != {"selection.json", "program.pkl.gz"}:
            raise ValueError("Missing or unexpected saved artifact")
        for name, digest in result["artifacts"].items():
            if sha(folder / name) != digest:
                raise ValueError("Saved artifact differs")
        stored = load(folder / "program.pkl.gz")
        phases = stored["phases"]
        selection = select_improvement(phases["fixed"], phases.get("optimized"))
        row = read(folder / "selection.json")
        if (
            selection != stored["selection"]
            or row != result["selection"]
            or describe_selection(selection) != row["selection"]
        ):
            raise ValueError("Saved selection differs")
        for name, phase in phases.items():
            if row["phases"][name] != {
                "work": phase.result.work,
                "refusal": phase.result.refusal,
            }:
                raise ValueError("Saved work differs")
        if selection.plan is not None or result["newly_valued_parent"]:
            if set(result["oracle"]) != {"parent", "selected"}:
                raise ValueError("Required independent oracle is missing")
            if result["oracle"]["parent"]["value"] != str(selection.parent_q):
                raise ValueError("Parent oracle differs")
        elif result["oracle"]:
            raise ValueError("Unexpected oracle receipt")
        old = read(PREVIOUS / folder.name / "selection.json")
        if result["newly_valued_parent"] != (
            old["selection"]["parent_q"] is None and selection.parent_q is not None
        ):
            raise ValueError("Parent completion classification differs")
        if result["oracle"]:
            plan = selection.plan or phases["fixed"].result.plan
            if result["oracle"]["selected"]["value"] != str(plan.values[plan.action]):
                raise ValueError("Selected oracle differs")
            worlds = load(PREVIOUS / folder.name / "program.pkl.gz")["worlds"]
            for replay in result["oracle"].values():
                if replay["represented_work"]["completions"] != sum(
                    map(int, worlds.weights)
                ):
                    raise ValueError("Incomplete oracle mass")
                totals["oracle_physical"].update(replay["physical_work"])
                totals["oracle_represented"].update(replay["represented_work"])
        for key, data in (("old_planning", old), ("new_planning", row)):
            for phase in data["phases"].values():
                work = phase["work"]
                peaks[key] = max(peaks[key], work.get("maximum_frontier_rows", 0))
                totals[key].update(
                    {k: v for k, v in work.items() if k != "maximum_frontier_rows"}
                )
        totals["support_audit"].update(result["support_audit_work"])
        totals["actual"].update(result["assessment"]["actual_work"])
        totals["replay"].update(result["assessment"]["replay_work"])
        outcomes = result["assessment"]["outcomes"]
        if selection.plan is None and outcomes["parent"] != outcomes["controller"]:
            raise ValueError("Fallback execution differs")
        rows.append(
            {
                "fixture": i,
                "source_key": result["source_key"],
                "old_selection": old["selection"],
                "new_selection": row["selection"],
                "newly_valued_parent": result["newly_valued_parent"],
                "common_exact_values_checked": row["common_exact_values_checked"],
                "old_seconds": old["seconds"],
                "new_seconds": row["seconds"],
                "old_phases": old["phases"],
                "new_phases": row["phases"],
            }
        )
        bindings[str(result_path)] = sha(result_path)
    for name in ("protocol.json", "queue.json", "summary.json"):
        bindings[str(base / name)] = sha(base / name)
    receipt = {
        "status": "passed",
        "scope": "Complete paired development screen; no population inference",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "rows": rows,
        "work": {k: dict(v) for k, v in totals.items()},
        "peak_frontier_rows": peaks,
    }
    write(output, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/score_bound_screen"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
