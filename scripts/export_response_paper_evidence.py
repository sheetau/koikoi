"""Export the closed fixed-opponent response comparison and its verified table."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export():
    base = Path("runs/response_study")
    audit_path = base / "independent_audit.json"
    assert (
        sha(audit_path)
        == "ff6677bcb24ffac367b5003301be4a04236a913785d725cdd2672b2d09a3f79c"
    )
    audit = read(audit_path)
    assert audit["status"] == "passed"
    inputs = audit["inputs"].copy()
    assert all(sha(path) == digest for path, digest in inputs.items())
    inputs[str(audit_path)] = sha(audit_path)
    inputs[str(Path(__file__))] = sha(__file__)
    state, analysis = read(base / "process.json"), read(base / "analysis.json")
    assert state["status"] == "complete" and analysis["status"] == "passed"
    assert state["analysis_sha256"] == sha(base / "analysis.json")
    assert audit["evaluation_matches"] == 120000
    assert audit["shared_unique_deal_streams"] == 5000
    assert audit["replicated_response_gain"] is False
    assert not any(audit["primary_positive_lower_bounds"].values())
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Opponent & Response minus parent & Response minus self-play \\",
        r"\midrule",
    ]
    for index in (1, 2):
        for opponent, label in (
            ("rl-wp-assisted", "WP"),
            ("rl-point-assisted", "Point"),
        ):
            fields = []
            for suffix in ("", "_selfplay"):
                key = f"parent{index}_response-parent{index}{suffix}/{opponent}"
                saved, independent = (
                    analysis["contrasts"][key],
                    audit["recomputed"][key],
                )
                np.testing.assert_allclose(
                    [saved["win_change"], *saved["win_ci95"]],
                    [independent["win"], *independent["win_ci95"]],
                    rtol=0,
                    atol=1e-12,
                )
                values = [
                    100 * saved["win_change"],
                    *[100 * x for x in saved["win_ci95"]],
                ]
                fields.append(f"${values[0]:+.3f}$ [{values[1]:.3f}, {values[2]:.3f}]")
            rows.append(f"{index} & {label} & " + " & ".join(fields) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    folder = Path("docs/paper/generated")
    table = folder / "response_table.tex"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    result = {
        "status": "passed",
        "inputs": inputs,
        "outputs": {str(table): sha(table)},
        "training_totals": audit["training_totals"],
        "evaluation_matches": audit["evaluation_matches"],
        "evaluation_decisions": audit["evaluation_decisions"],
        "shared_unique_deal_streams": audit["shared_unique_deal_streams"],
        "outcomes": analysis["outcomes"],
        "contrasts": analysis["contrasts"],
        "primary_positive_lower_bounds": audit["primary_positive_lower_bounds"],
        "replicated_response_gain": audit["replicated_response_gain"],
        "scope": "Completed development comparison on shared 23M deal streams; exact artifact bindings and independent raw-pair audit. Equal simulator budgets, not equal learner samples. No fresh final confirmation, new simulation or training-seed uncertainty claim.",
    }
    ledger = folder / "response_evidence.json"
    ledger.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    assert all(sha(path) == digest for path, digest in inputs.items())
    print(
        json.dumps(
            {"output": str(ledger), "sha256": sha(ledger), "table_sha256": sha(table)}
        )
    )


if __name__ == "__main__":
    export()
