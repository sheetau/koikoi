"""Export a compact paired-supervision table from verified raw match outcomes."""

import json
from pathlib import Path

import numpy as np

from scripts.plot_extension_results import load_changes
from scripts.run_arbitration_study import sha


def export():
    base = Path("runs/paired_arbitration_study")
    protocol, _, changes, inputs = load_changes(base, "paired", [1, 2])
    audit = json.loads((base / "analysis.json").read_text(encoding="utf-8"))
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Inputs & $\Delta$ wins vs WP (pp) & $\Delta$ wins vs Point (pp) \\",
        r"\midrule",
    ]
    for parent in (1, 2):
        for arm, label in (("full", "Full"), ("no_match", "No match context")):
            columns = []
            for opponent in ("rl-wp-assisted", "rl-point-assisted"):
                values = changes[parent, arm, opponent]["wins"]
                recorded = audit["paired_comparisons"][
                    f"parent{parent}_{arm}-parent{parent}/{opponent}"
                ]
                np.testing.assert_allclose(
                    values,
                    np.array([recorded["win_difference"], *recorded["win_ci95"]]) * 100,
                    atol=1e-12,
                    rtol=0,
                )
                columns.append(f"${values[0]:+.3f}$ [{values[1]:.3f}, {values[2]:.3f}]")
            rows.append(f"{parent} & {label} & " + " & ".join(columns) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    output = Path("docs/paper/generated")
    table = output / "paired_supervision_table.tex"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    extra = [Path(__file__), Path("scripts/plot_extension_results.py")]
    for parent in (1, 2):
        extra.append(base / f"parent{parent}" / "targets.pt")
        extra.append(base / f"parent{parent}" / "held_out_diagnostic.json")
        for arm in ("full", "no_match"):
            extra.append(base / f"parent{parent}_{arm}" / "final.pt")
    for path in (
        "runs/score_fit_screen/fit_audit.json",
        "runs/paired_difference_fit_screen/fit_audit.json",
        "runs/preference_fit_screen/fit_audit.json",
        "runs/score_update_budget_control/fit_audit.json",
        "runs/score_scaled_fit_screen/fit_audit.json",
        "runs/score_scale_followup/comparison.json",
        "runs/scaled_preference_collapse_diagnostic.json",
        "runs/score_zero_decay_control/fit_audit.json",
        "runs/score_zero_decay_control/comparison.json",
    ):
        extra.append(Path(path))
    for path in extra:
        inputs[str(path)] = sha(path)
        if path.name == "fit_audit.json":
            screen = json.loads(path.read_text(encoding="utf-8"))
            assert screen["status"] == "passed"
            plan_path = path.parent / "plan.json"
            assert sha(plan_path) == screen["plan_sha256"]
            inputs[str(plan_path)] = sha(plan_path)
            for case, report in screen["reports"].items():
                checkpoint = path.parent / case / "final.pt"
                assert sha(checkpoint) == report["checkpoint_sha256"]
                inputs[str(checkpoint)] = report["checkpoint_sha256"]
    for parent in (1, 2):
        path = (
            Path("runs/scaled_paired_targets_recovered/data")
            / f"parent{parent}"
            / "targets.pt"
        )
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Evidence changed: {path}")
    result = {
        "scope": "Completed paired full-match study, table recomputed from raw pair outcomes; completed score-fitting reports retained as historical diagnostic evidence, not new match-strength estimates. No pending opponent-target/static result included.",
        "inputs": inputs,
        "outputs": {str(table): sha(table)},
        "pair_count": protocol["evaluation"]["pairs"],
        "evaluation_matches": sum(row["matches"] for row in audit["outcomes"].values()),
        "evaluation_decisions": sum(
            row["decisions"] for row in audit["outcomes"].values()
        ),
        "changes": {"/".join(map(str, key)): value for key, value in changes.items()},
    }
    ledger = output / "paired_evidence.json"
    ledger.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        {
            "inputs": len(inputs),
            "ledger_sha256": sha(ledger),
            "table_sha256": sha(table),
        }
    )


if __name__ == "__main__":
    export()
