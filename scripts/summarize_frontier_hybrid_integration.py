"""Complete-only preservation and work audit for the twelve integrated cases."""

import argparse
from collections import Counter
from pathlib import Path

from scripts.check_frontier_hybrid_integration import checked, compare_patch
from scripts.frontier_hybrid_controller import FrontierHybridExecutor
from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_study import load


def summarize(base, output):
    if output.exists():
        raise FileExistsError(output)
    p = checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 12}:
        raise ValueError("All twelve integration cases must finish")
    summary = read(base / "summary.json")
    if summary["status"] != "passed" or len(summary["results"]) != 12:
        raise ValueError("Incomplete integration summary")
    totals = {
        k: Counter()
        for k in (
            "fixed",
            "optimized",
            "prefix",
            "proposal",
            "reference",
            "support_audit",
            "actual",
            "replay",
        )
    }

    def add(label, work):
        totals[label].update(
            {
                k: v
                for k, v in work.items()
                if k
                not in (
                    "maximum_frontier_rows",
                    "peak_frontier_rows",
                    "peak_rows",
                )
            }
        )

    rows, bindings = [], {}
    for i, fixture in enumerate(p["fixtures"]):
        folder = base / f"fixture_{i:02}"
        path = folder / "result.json"
        row = read(path)
        if (
            summary["results"].get(str(path)) != sha(path)
            or row["status"] != "passed"
            or row["fixture"] != i
            or row["case"] != fixture["case"]
            or row["protocol_sha256"] != sha(base / "protocol.json")
            or row["fresh_chance_draws"] != 0
            or row["new_value_oracle_work"] != 0
            or set(row["artifacts"]) != {"program.pkl.gz"}
            or row["artifacts"]["program.pkl.gz"] != sha(folder / "program.pkl.gz")
        ):
            raise ValueError("Frozen integrated case differs")
        hybrid = load(folder / "program.pkl.gz")
        FrontierHybridExecutor(hybrid, None, hybrid.fixed.binding.parent_identity)
        old = load(fixture["program"])
        if (
            hybrid.fixed != old["phases"]["fixed"]
            or hybrid.optimized != old["phases"].get("optimized")
            or hybrid.original != old["selection"]
            or row["fixed_work"] != hybrid.fixed.result.work
            or row["optimized_work"]
            != (None if hybrid.optimized is None else hybrid.optimized.result.work)
            or row["patched"] != (hybrid.patch is not None)
            or row["decisions"] != hybrid.decisions
            or row["gain_lower"] != str(hybrid.gain_lower)
            or row["old_reason"] != hybrid.original.reason
            or hybrid.phase_decisions != p["design"]["phase_decisions"]
        ):
            raise ValueError("Original policy or integrated accounting changed")
        if fixture["patch_folder"] is not None:
            if hybrid.patch is None:
                raise ValueError("Refused-parent extension disappeared")
            previous = Path(fixture["patch_folder"])
            compare_patch(hybrid.patch, load(previous / "program.pkl.gz"))
            if row["assessment"] != read(previous / "result.json")["assessment"]:
                raise ValueError("Integrated actual patch execution changed")
            add("prefix", hybrid.patch.frontier.work)
            for _, local in hybrid.patch.comparisons:
                add("proposal", local["proposal_phase"].result.work)
                add("reference", local["parent_receipt"].work)
        elif hybrid.patch is not None:
            raise ValueError("Preservation control ran a different method")
        patch_work = 0 if hybrid.patch is None else hybrid.patch.decisions
        if row["patch_decisions"] != patch_work:
            raise ValueError("Patch work differs")
        add("fixed", hybrid.fixed.result.work)
        if hybrid.optimized is not None:
            add("optimized", hybrid.optimized.result.work)
        add("support_audit", row["support_audit_work"])
        source = read(fixture["source"])
        parent = row["assessment"]["outcomes"]["parent"]
        if (
            parent["final"] != source["final"]
            or parent["actions"]
            != source["actions"][source["selected_root"]["decision"] :]
        ):
            raise ValueError("Actual parent suffix changed")
        for name in ("actual", "replay"):
            add(name, row["assessment"][name + "_work"])
        rows.append(
            {
                k: row[k]
                for k in (
                    "fixture",
                    "case",
                    "old_reason",
                    "patched",
                    "gain_lower",
                    "decisions",
                    "patch_decisions",
                    "seconds",
                    "reused_value_evidence",
                )
            }
        )
        bindings[str(path)] = sha(path)
    work = {k: dict(v) for k, v in totals.items()}
    if sum(r["decisions"] for r in rows) != sum(
        work[k].get("decisions", 0)
        for k in (
            "fixed",
            "optimized",
            "prefix",
            "proposal",
            "reference",
        )
    ):
        raise ValueError("Separate phase work does not sum to integrated execution")
    for name in ("protocol.json", "queue.json", "summary.json"):
        bindings[str(base / name)] = sha(base / name)
    report = {
        "status": "passed",
        "scope": "Twelve fixed integration cases; old phases actually reexecuted; no fresh strength inference",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "rows": rows,
        "work": work,
    }
    write(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/frontier_hybrid_integration")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
