"""Export the complete explicit-versus-symbolic computational comparison."""

import argparse
from pathlib import Path

from scripts.export_certified_tail_replay import aggregate, compare_coverage
from scripts.export_full_source_results import table
from scripts.run_arbitration_study import read, sha, write
from scripts.symbolic_tail_replay import check, source_records


def export(base, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh report and receipt paths")
    protocol, _ = check(base)
    assert read(base / "queue.json")["status"] == "complete"
    summary = read(base / "summary.json")
    assert summary["status"] == "passed"
    assert summary["protocol_sha256"] == sha(base / "protocol.json")
    assert summary["reused_sources"] == 512 and summary["new_source_matches"] == 0
    source = Path(protocol["source"])
    old_rows, new_rows, bindings = [], [], {}
    for chunk in protocol["chunks"]:
        folder = base / f"chunk_{chunk:03}"
        path = folder / "complete.json"
        bindings[str(path)] = sha(path)
        assert summary["chunk_bindings"][str(path)] == bindings[str(path)]
        complete = read(path)
        assert complete["status"] == "complete"
        assert read(folder / "process.json")["status"] == "complete"
        assert complete["protocol_sha256"] == sha(base / "protocol.json")
        for name, digest in complete["artifacts"].items():
            assert sha(folder / name) == digest
        old_rows.extend(source_records(source / f"chunk_{chunk:03}"))
        new_rows.extend(source_records(folder))
    assert bindings == summary["chunk_bindings"]
    assert [r["index"] for r in new_rows] == list(range(512))
    coverage = compare_coverage(old_rows, new_rows)
    versions = {"opponent_only": aggregate(old_rows), "symbolic": aggregate(new_rows)}
    outcomes = []
    exact = {}
    for version, cells in versions.items():
        exact[version] = {}
        for field, key in (("gain", "conditional"), ("actual", "actual")):
            values = [
                cells[parent, "controller"][field]
                - (cells[parent, "root_only"][field] if control == "root_only" else 0)
                for parent in (0, 1)
                for control in ("parent", "root_only")
            ]
            assert all(
                abs(float(a) - b) < 1e-12
                for a, b in zip(values, summary[f"{version}_{key}_mean"], strict=True)
            )
            exact[version][key] = list(map(str, values))
        for (parent, arm), cell in cells.items():
            outcomes.append(
                [
                    version,
                    parent + 1,
                    arm,
                    cell["adoptions"],
                    str(cell["gain"]),
                    f"{100 * float(cell['gain']):+.4f}",
                    str(cell["actual"]),
                    f"{100 * float(cell['actual']):+.4f}",
                ]
            )
    original = Path(read(source / "protocol.json")["source"])
    original_gate = read(original / "summary.json")["development_strength_gate"]
    text = (
        "Executable symbolic endgame programs: complete-cohort comparison\n"
        "===============================================================\n\n"
        "The opponent-only and symbolic programs use the same512 saved sources "
        "and128 deal streams. No new source matches are added. Missing roots stay "
        "in each parent's256-source denominator. Both arms use the original "
        "maximum allowances and fresh caches. Actual win scores are1 for wins, "
        "1/2 for draws and0 for losses. Outcome gains below are versus the "
        "unchanged parent.\n\n"
        + table(
            "Policy outcomes",
            [
                "Version",
                "Parent",
                "Arm",
                "Adoptions",
                "Exact conditional mean",
                "Conditional pp",
                "Exact actual mean",
                "Actual pp",
            ],
            outcomes,
        )
        + table("Coverage changes", ["Arm", "Counts"], list(coverage.items()))
        + table(
            "Planning work and coverage",
            ["Version", "Arm", "Counts"],
            [
                [
                    version,
                    arm,
                    {k: v for k, v in counts.items() if k != "planning_seconds"},
                ]
                for version, arms in summary["costs"].items()
                for arm, counts in arms.items()
            ],
        )
        + "Physical transitions, represented simulated completions and certified "
        "rows/mass are separate measures. The opponent_only implementation calls "
        "its loss counts certified_opponent_rows/mass; the symbolic version calls "
        "them certified_opponent_loss_rows/mass and adds certified_own_win_rows/mass. "
        "Program-entry counts below count entries across root programs, not unique "
        "histories or simulated work.\n\n"
        + table(
            "Verification, conditional execution and program structure",
            ["Category", "Counts"],
            list(summary["work"].items()),
        )
        + "Symbolic parent calls are included in total parent adapter calls. "
        "Independent oracle calls are reported separately from planning and "
        "actual-tail calls. Every adopted program's parent and selected value "
        "was replayed through terminal scoring with Python rules, including "
        "omitted actions. Common completed root values and retained actions "
        "matched; omitted histories were checked against the public certificate.\n\n"
        + table(
            "Operational planning times",
            ["Version", "Arm", "Seconds"],
            [
                [v, a, f"{c.get('planning_seconds', 0):.3f}"]
                for v, arms in summary["costs"].items()
                for a, c in arms.items()
            ],
        )
        + "Saved controls and concurrent workers do not provide an isolated "
        "latency benchmark. Lower per-branch cost can enable additional phases, "
        "so it need not lower total scheduled work.\n\n"
        + f"The original prospective strength gate {'passed' if original_gate else 'failed'} "
        "and remains unchanged. These reused-cohort comparisons are descriptive, "
        "without a fresh strength gate or confidence intervals. Conditional "
        "improvements depend on the specified belief and opponent model; they "
        "are distinct from observed match-win changes.\n\n"
        "This export verifies completed audit bindings and exact aggregate "
        "arithmetic. It adds no simulation or inference.\n\n"
        f"Protocol SHA256: {sha(base / 'protocol.json')}.\n"
        f"Audit summary SHA256: {sha(base / 'summary.json')}.\n"
    )
    output.write_text(text, encoding="utf-8", newline="\n")
    write(
        receipt,
        {
            "status": "passed",
            "report_sha256": sha(output),
            "summary_sha256": sha(base / "summary.json"),
            "exporter_sha256": sha(__file__),
            "chunk_bindings": bindings,
            "coverage": coverage,
            "exact_contrasts": exact,
            "new_simulator_work": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/symbolic_tail_replay"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/SYMBOLIC_TAIL_REPLAY_RESULTS.rst")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("runs/symbolic_tail_replay_export.json")
    )
    args = parser.parse_args()
    export(args.base, args.output, args.receipt)
