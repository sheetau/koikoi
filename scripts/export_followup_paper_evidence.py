"""Bind completed static and target controls; recompute static paired effects."""

import json
from pathlib import Path

import numpy as np

from scripts.analyze_arbitration_study import interval, outcome
from scripts.analyze_static_expert_controls import validate_report
from scripts.run_arbitration_study import read, sha


def export():
    static, paired = (
        Path("runs/static_expert_controls"),
        Path("runs/paired_arbitration_study"),
    )
    p, previous = read(static / "protocol.json"), read(paired / "protocol.json")
    saved = read(static / "parent_comparisons.json")
    inputs = saved["inputs"].copy()
    inputs[str(static / "parent_comparisons.json")] = sha(
        static / "parent_comparisons.json"
    )
    comparisons = {}
    assert read(static / "process.json")["status"] == "complete"
    assert p["evaluation"] == previous["evaluation"]
    for case, spec in p["cases"].items():
        for opponent in p["evaluation"]["opponents"]:
            path = static / case / (opponent + ".json")
            wins, points = validate_report(read(path), spec, opponent, p)
            parent = case.split("_", 1)[0]
            _, base_wins, base_points = outcome(paired, parent, opponent, previous)
            dw, dp = wins - base_wins, points - base_points
            result = {
                "win_score_gain": float(dw.mean()),
                "win_ci95": interval(dw),
                "point_margin_gain": float(dp.mean()),
                "points_ci95": interval(dp),
            }
            key = case + "/" + opponent
            for field, value in result.items():
                np.testing.assert_allclose(
                    value, saved["comparisons"][key][field], atol=1e-12, rtol=0
                )
            comparisons[key] = result
            inputs[str(path)] = sha(path)
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Opponent & $\Delta$ wins (pp) & $\Delta$ point margin \\",
        r"\midrule",
    ]
    for i in (1, 2):
        for opponent, label in (
            ("rl-wp-assisted", "WP"),
            ("rl-point-assisted", "Point"),
        ):
            row = comparisons[f"parent{i}_point_koi/{opponent}"]
            wins = np.array([row["win_score_gain"], *row["win_ci95"]]) * 100
            points = [row["point_margin_gain"], *row["points_ci95"]]
            rows.append(
                f"{i} & {label} & ${wins[0]:+.3f}$ [{wins[1]:.3f}, {wins[2]:.3f}] & ${points[0]:+.3f}$ [{points[1]:.3f}, {points[2]:.3f}]"
                + r" \\"
            )
    rows += [r"\bottomrule", r"\end{tabular}"]
    output = Path("docs/paper/generated")
    table = output / "static_koi_table.tex"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    target = Path("runs/opponent_target_study")
    results, protocol = read(target / "results.json"), read(target / "protocol.json")
    assert read(target / "process.json")["status"] == "complete"
    inputs.update(protocol["inputs"])
    for name in ("protocol.json", "process.json", "results.json"):
        inputs[str(target / name)] = sha(target / name)
    for index, row in results.items():
        audit = target / "data" / f"parent{index}" / "audit.json"
        assert read(audit)["status"] == "passed"
        inputs[str(audit)] = sha(audit)
        for condition, collection in row["collection"]["conditions"].items():
            folder = target / "data" / f"parent{index}" / condition
            for filename, key in (
                ("targets.pt", "dataset_sha256"),
                ("features.pt", "cache_sha256"),
            ):
                inputs[str(folder / filename)] = collection[key]
            checkpoint = target / "fits" / f"parent{index}" / condition / "final.pt"
            endpoint = row["analysis"]["endpoints"][
                condition + "/evaluated_on_opponent_preserving"
            ]
            inputs[str(checkpoint)] = endpoint["checkpoint_sha256"]
    for path in (Path(__file__), Path("scripts/analyze_static_expert_controls.py")):
        inputs[str(path)] = sha(path)
    for i in (1, 2):
        path = Path("runs/switch_budget_diagnostic") / f"parent{i}.json"
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Evidence changed: {path}")
    ledger = {
        "inputs": inputs,
        "outputs": {str(table): sha(table)},
        "static_comparisons": comparisons,
        "opponent_target_contrasts": {
            key: row["analysis"]["opponent_target_minus_parent_target"]
            for key, row in results.items()
        },
        "scope": "Static effects independently recomputed from raw matches; opponent-target/switch diagnostics bind completed audited results, not new match evaluations. Live direct-response study excluded.",
    }
    (output / "followup_evidence.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "inputs": len(inputs),
                "table_sha256": sha(table),
                "ledger_sha256": sha(output / "followup_evidence.json"),
            }
        )
    )


if __name__ == "__main__":
    export()
