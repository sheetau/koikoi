"""Complete-only verification and accounting for the fixed interval screen."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.interval_tail import select_interval
from scripts.run_arbitration_study import read, sha, write
from scripts.screen_interval_tail import PREVIOUS, checked, describe
from scripts.uniform_controller_study import load


def summarize(base, destination):
    if destination.exists():
        raise FileExistsError(destination)
    protocol = checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("All eight fixtures must finish")
    summary = read(base / "summary.json")
    if summary["status"] != "passed" or len(summary["results"]) != 8:
        raise ValueError("Expected eight successful results")
    totals = {
        k: Counter()
        for k in (
            "planning",
            "support_audit",
            "actual",
            "replay",
            "oracle_physical",
            "oracle_represented",
        )
    }
    rows, bindings = [], {}
    common = 0
    for i in range(8):
        folder = base / f"fixture_{i:02}"
        path = folder / "result.json"
        if summary["results"][str(path)] != sha(path):
            raise ValueError("Result digest changed")
        result = read(path)
        if (
            result["status"] != "passed"
            or result["fixture"] != i
            or result["fresh_chance_draws"] != 0
            or result["protocol_sha256"] != sha(base / "protocol.json")
            or set(result["artifacts"]) != {"selection.json", "program.pkl.gz"}
        ):
            raise ValueError("Invalid fixture identity or artifact set")
        for name, digest in result["artifacts"].items():
            if sha(folder / name) != digest:
                raise ValueError("Saved artifact changed")
        stored = load(folder / "program.pkl.gz")
        receipts = stored["receipts"]
        selection = select_interval(receipts[0], receipts[1:])
        row = read(folder / "selection.json")
        action = (
            None
            if selection.candidate is None
            else dict(selection.candidate.program.overrides)[
                selection.parent.program.binding.root_key
            ]
        )
        if (
            selection != stored["selection"]
            or row != result["selection"]
            or row["receipts"] != list(map(describe, receipts))
            or row["gain_lower"] != str(selection.gain_lower)
            or row["selected_action"] != action
        ):
            raise ValueError("Saved interval selection differs")
        original = load(PREVIOUS / folder.name / "program.pkl.gz")
        old_row = read(PREVIOUS / folder.name / "selection.json")
        source = read(PREVIOUS / folder.name / "source.json")
        mass = sum(map(int, original["worlds"].weights))
        actions = [
            dict(r.program.overrides)[r.program.binding.root_key] for r in receipts
        ]
        if (
            len(set(actions)) != len(actions)
            or set(actions) != set(receipts[0].program.binding.legal)
            or any(
                len(r.program.overrides) != 1 or r.total_mass != mass for r in receipts
            )
            or receipts[0].program.parent_root_action
            != source["actions"][source["selected_root"]["decision"]]
        ):
            raise ValueError("Screen changed its root-only candidate family")
        fixed = original["phases"]["fixed"].result.plan
        if fixed is not None:
            for action, r in zip(actions, receipts, strict=True):
                if action in fixed.values:
                    if not r.lower <= fixed.values[action] <= r.upper:
                        raise ValueError(
                            "Interval excludes an independently completed predecessor value"
                        )
                    common += 1
        for r in receipts:
            w = r.work
            if (
                w["terminal_mass"] != w["represented_completions"]
                or w["terminal_mass"]
                + w["certified_win_mass"]
                + w["certified_loss_mass"]
                != r.resolved_mass
            ):
                raise ValueError("Resolved mass accounting differs")
            totals["planning"].update(
                {k: v for k, v in w.items() if k != "peak_frontier_rows"}
            )
        decisions = sum(r.work["decisions"] for r in receipts)
        nodes = sum(r.work["expanded_tasks"] for r in receipts)
        if (
            decisions != row["decisions"]
            or decisions != stored["decisions"]
            or nodes != row["expanded_tasks"]
            or nodes != stored["expanded_tasks"]
            or decisions > protocol["design"]["max_decisions"]
            or nodes > protocol["design"]["max_nodes"]
        ):
            raise ValueError("Shared budget accounting differs")
        newly_exact = (
            old_row["selection"]["parent_q"] is None
            and receipts[0].lower == receipts[0].upper
        )
        expected_oracles = (
            {"parent", "candidate"}
            if selection.candidate is not None
            else {"parent"}
            if newly_exact
            else set()
        )
        if (
            result["newly_exact_parent"] != newly_exact
            or set(result["oracle"]) != expected_oracles
        ):
            raise ValueError("Required independent value replay differs")
        for label, oracle in result["oracle"].items():
            r = receipts[0] if label == "parent" else selection.candidate
            if (
                not r.lower <= Fraction(oracle["value"]) <= r.upper
                or oracle["represented_work"]["completions"] != mass
            ):
                raise ValueError("Independent value or represented mass differs")
            totals["oracle_physical"].update(oracle["physical_work"])
            totals["oracle_represented"].update(oracle["represented_work"])
        outcomes = result["assessment"]["outcomes"]
        if outcomes["parent"]["final"] != source["final"]:
            raise ValueError("Parent source suffix differs")
        if selection.candidate is None and outcomes["parent"] != outcomes["controller"]:
            raise ValueError("Fallback execution differs")
        totals["support_audit"].update(result["support_audit_work"])
        for label in ("actual", "replay"):
            totals[label].update(result["assessment"][label + "_work"])
        rows.append(
            {
                "fixture": i,
                "source_key": source["source_key"],
                **row,
                "newly_exact_parent": newly_exact,
                "program_peak_rows": [r.work["peak_frontier_rows"] for r in receipts],
            }
        )
        bindings[str(path)] = sha(path)
    for name in ("protocol.json", "summary.json", "queue.json"):
        bindings[str(base / name)] = sha(base / name)
    report = {
        "status": "passed",
        "scope": "Complete development comparison; no fresh strength inference",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "rows": rows,
        "common_predecessor_values_checked": common,
        "work": {k: dict(v) for k, v in totals.items()},
    }
    write(destination, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/interval_tail_screen"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
