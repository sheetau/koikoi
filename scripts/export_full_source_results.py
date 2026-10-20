"""Export all fixed full-source comparisons after both audits complete."""

import argparse
import json
from pathlib import Path

from scripts.full_source_study import check_plan, read
from scripts.train_branch_policy_gradient import sha, write


def table(title, headers, rows):
    lines = [f".. list-table:: {title}", "   :header-rows: 1", ""]
    for row in [headers, *rows]:
        for index, cell in enumerate(row):
            lines.append(f"   {'*' if index == 0 else ' '} - {cell}")
    return "\n".join(lines) + "\n\n"


def export(base, output, receipt):
    base, output, receipt = Path(base), Path(output), Path(receipt)
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh output and receipt paths")
    plan = check_plan(base)
    if plan["run_kind"] != "development":
        raise ValueError("Strength report requires the fixed development study")
    finished = read(base / "finished.json")
    complete = read(base / "complete.json")
    audit = read(base / "audit.json")
    summary = read(base / "evaluation/summary.json")
    evaluation_audit = read(base / "evaluation/audit.json")
    assert finished["status"] == complete["status"] == summary["status"] == "complete"
    assert audit["status"] == evaluation_audit["status"] == "passed"
    bindings = {}
    for key, relative in (
        ("protocol_sha256", "protocol.json"),
        ("training_audit_sha256", "audit.json"),
        ("evaluation_sha256", "evaluation/summary.json"),
        ("evaluation_audit_sha256", "evaluation/audit.json"),
    ):
        assert finished[key] == sha(base / relative)
        bindings[str(base / relative)] = finished[key]
    assert audit["complete_sha256"] == sha(base / "complete.json")
    assert (
        audit["protocol_sha256"]
        == complete["protocol_sha256"]
        == sha(base / "protocol.json")
    )
    assert evaluation_audit["summary_sha256"] == sha(base / "evaluation/summary.json")
    assert (
        evaluation_audit["protocol_sha256"]
        == summary["protocol_sha256"]
        == sha(base / "evaluation/protocol.json")
    )
    assert summary["points_sha256"] == sha(base / "evaluation/points.npy")
    assert [row["models"] for row in summary["primary"]] == plan["evaluation"][
        "primary"
    ]
    assert [row["models"] for row in summary["secondary_contrast_means"]] == plan[
        "evaluation"
    ]["secondary"]
    assert list(complete["cases"]) == list(plan["cases"])
    assert summary["model_order"] == list(plan["cases"]) + ["parent1", "parent2"]
    assert set(summary["secondary_matchup_means"]) == set(summary["model_order"])
    for relative in ("complete.json", "finished.json", "evaluation/points.npy"):
        bindings[str(base / relative)] = sha(base / relative)

    verdict = "passes" if summary["progression_criterion"] else "does not pass"
    text = (
        "Complete-source Monte Carlo comparison\n"
        "======================================\n\n"
        f"The completed twelve-arm study {verdict} its predefined six-contrast\n"
        "development criterion. Both training and evaluation audits pass.\n"
        "No canonical model is promoted by this screen.\n\n"
        "Both retained parents receive all-source Monte Carlo, one-root Monte\n"
        "Carlo and shared all-action branches, each at learning rates 1e-5 and\n"
        "1e-4. Every arm receives 128 updates and 8,192 source matches.\n"
        "Matching source matches does not match gradient or branch computation.\n"
        "All twelve endpoints and both parents face assisted WP and Point on\n"
        "1,024 duplicate pairs per opponent, using GPU batch-128 greedy play.\n"
        "The 57,344 evaluation matches share 2,048 distinct deal streams.\n"
        "The reserved 101M block is unused. See FULL_SOURCE_STUDY.rst for the\n"
        "fixed design and the distinction from the paper's CPU evaluation.\n\n"
        "Primary comparisons\n-------------------\n\n"
        "Win changes are percentage points in the equal-WP/Point mixture,\n"
        "counting draws as one half. All six lower bounds must exceed zero.\n"
        "The simultaneous 95% radius uses 10,000 maximum-absolute-centered\n"
        "bootstrap deviations over compound paired-deal blocks. These are\n"
        "development comparisons, not a universal strength ranking.\n\n"
    )
    text += table(
        "Every predefined primary contrast",
        ["Candidate", "Comparator", "Change (pp)", "Simultaneous 95% interval"],
        [
            [
                *row["models"],
                f"{100 * row['win_change']:+.4f}",
                "[{:+.4f}, {:+.4f}]".format(
                    *(100 * value for value in row["simultaneous_95"])
                ),
            ]
            for row in summary["primary"]
        ],
    )
    text += "Secondary comparisons\n---------------------\n\n"
    text += (
        "The following means are descriptive; no secondary interval is inferred.\n\n"
    )
    text += table(
        "Every predefined secondary contrast",
        ["Candidate", "Comparator", "Change (pp)"],
        [
            [*row["models"], f"{100 * row['win_change']:+.4f}"]
            for row in summary["secondary_contrast_means"]
        ],
    )
    text += table(
        "Every endpoint and parent matchup",
        ["Model", "Opponent", "Win score (%)", "Point margin"],
        [
            [
                name,
                kind.upper(),
                f"{100 * row['win']:.4f}",
                f"{row['point_margin']:+.4f}",
            ]
            for name in summary["model_order"]
            for kind, row in summary["secondary_matchup_means"][name].items()
        ],
    )
    totals = {
        kind: dict.fromkeys(("decisions", "round_endings", "completions"), 0)
        for kind in ("source", "branches")
    }
    rows, gradient_rows, updates = [], 0, 0
    for name, case in complete["cases"].items():
        path = base / name / "metrics.jsonl"
        assert sha(path) == case["metrics_sha256"]
        bindings[str(path)] = case["metrics_sha256"]
        metrics = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(metrics) == plan["updates"] == case["counters"]["updates"]
        count = sum(row["gradient_rows"] for row in metrics)
        gradient_rows += count
        updates += case["counters"]["updates"]
        row = [name]
        for kind, aggregate in totals.items():
            work = case["counters"][kind]
            for key in aggregate:
                aggregate[key] += work[key]
            row.append(
                f"{work['decisions']:,} / {work['round_endings']:,} / "
                f"{work['completions']:,}"
            )
        rows.append([*row, f"{count:,}"])
    assert updates == 1536 and totals["source"]["completions"] == 98304
    assert summary["work"]["completions"] == 57344
    text += (
        "Work and evidence\n-----------------\n\n"
        "Work triples below are decisions / round endings / completed matches\n"
        "or branches. Gradient rows count all source decisions for full-source\n"
        "learning and selected roots for the two root estimators. Branch action\n"
        "values contribute within each selected root's loss.\n\n"
    )
    text += table(
        "Actual training work",
        ["Condition", "Source work", "Additional branch work", "Gradient rows"],
        rows,
    )
    for kind, work in [*totals.items(), ("evaluation", summary["work"])]:
        text += (
            f"{kind.capitalize()}: {work['decisions']:,} decisions, "
            f"{work['round_endings']:,} round endings, "
            f"{work['completions']:,} completions.\n\n"
        )
    text += (
        f"Total: {updates:,} optimizer updates and {gradient_rows:,} gradient rows.\n"
        f"Training elapsed: {complete['elapsed_seconds']:.6f} seconds including I/O;\n"
        "this is not an isolated throughput benchmark.\n\n"
        "The training audit replays source and branch transitions, every stored\n"
        "full-source state/action/return, deals, selected roots and work counts.\n"
        "The evaluation audit replays trajectories and independently constructs\n"
        "the simultaneous bootstrap. Neither repeats all neural inference or\n"
        "optimizer updates. This exporter checks the completed receipt chain\n"
        "and metrics hashes; it does not rerun those audits.\n\n"
    )
    for path, digest in bindings.items():
        text += f"* ``{path}``: ``{digest}``.\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "bindings": bindings,
            "output": str(output),
            "output_sha256": sha(output),
            "primary_count": len(summary["primary"]),
            "secondary_count": len(summary["secondary_contrast_means"]),
            "model_count": len(summary["model_order"]),
            "totals": totals,
            "gradient_rows": gradient_rows,
            "optimizer_updates": updates,
            "progression_criterion": summary["progression_criterion"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/full_source_study")
    parser.add_argument("--output", default="docs/FULL_SOURCE_RESULTS.rst")
    parser.add_argument("--receipt", default="runs/full_source_results_export.json")
    args = parser.parse_args()
    export(args.base, args.output, args.receipt)
