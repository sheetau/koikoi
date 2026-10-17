"""Export the completed learner-return comparison after its independent audit.

The caller pins the final audit hash explicitly. No interim results are exported.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

BASE = Path("runs/learner_return_study")
PROTOCOL_SHA = "675638eb8561f52e2d023c7f8fedc3006eac59a051d32e3b54e8e7cb78e10674"
OPPONENTS = (("rl-wp-assisted", "WP"), ("rl-point-assisted", "Point"))
COMPARISONS = (
    ("learner", ""),
    ("lambda1", ""),
    ("response", ""),
    ("learner", "_response"),
    ("learner", "_lambda1"),
    ("lambda1", "_response"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def win_table(contrasts, comparisons, headers):
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        "Parent & Opponent & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for index in (1, 2):
        for opponent, label in OPPONENTS:
            cells = []
            for arm, control in comparisons:
                key = f"parent{index}_{arm}-parent{index}{control}/{opponent}"
                row = contrasts[key]
                values = [100 * row["win_change"], *[100 * x for x in row["win_ci95"]]]
                cells.append(f"${values[0]:+.3f}$ [{values[1]:.3f}, {values[2]:.3f}]")
            lines.append(f"{index} & {label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def export(expected_audit_sha, folder=Path("docs/paper/generated")):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha):
        raise ValueError("Pin the independent audit with a lowercase SHA256")
    state = read(BASE / "process.json")
    if state["status"] != "complete":
        raise ValueError("The learner-return study is not complete")
    audit_path = BASE / "independent_audit.json"
    assert sha(audit_path) == expected_audit_sha
    audit, analysis = read(audit_path), read(BASE / "analysis.json")
    assert audit["status"] == analysis["status"] == "passed"
    assert analysis["kind"] == "learner_return_development"
    assert (
        sha(BASE / "protocol.json")
        == state["protocol_sha256"]
        == analysis["protocol_sha256"]
        == PROTOCOL_SHA
    )
    assert state["analysis_sha256"] == sha(BASE / "analysis.json")
    inputs = audit["inputs"].copy()
    for name in ("protocol.json", "process.json", "analysis.json"):
        path = BASE / name
        assert inputs[str(path)] == sha(path)
    assert all(sha(path) == digest for path, digest in inputs.items())
    inputs[str(audit_path)] = expected_audit_sha
    inputs[str(Path(__file__))] = sha(__file__)
    outcome_keys = {
        f"parent{i}{suffix}/{opponent}"
        for i in (1, 2)
        for suffix in ("", "_response", "_learner", "_lambda1")
        for opponent, _ in OPPONENTS
    }
    contrast_keys = {
        f"parent{i}_{arm}-parent{i}{control}/{opponent}"
        for i in (1, 2)
        for arm, control in COMPARISONS
        for opponent, _ in OPPONENTS
    }
    assert set(analysis["outcomes"]) == outcome_keys
    assert set(analysis["contrasts"]) == contrast_keys
    assert set(audit["recomputed"]) == outcome_keys | contrast_keys
    for key in outcome_keys | contrast_keys:
        independent = audit["recomputed"][key]
        outcome = key in outcome_keys
        saved = analysis["outcomes" if outcome else "contrasts"][key]
        names = (
            ("win_score", "point_difference", "points_ci95")
            if outcome
            else ("win_change", "point_change", "point_ci95")
        )
        values = [
            saved[names[0]],
            saved[names[1]],
            *saved["win_ci95"],
            *saved[names[2]],
        ]
        assert np.isfinite(values).all()
        np.testing.assert_allclose(
            values,
            [
                independent["win"],
                independent["points"],
                *independent["win_ci95"],
                *independent["points_ci95"],
            ],
            rtol=0,
            atol=1e-12,
        )
    primary = {
        f"parent{i}_learner-parent{i}{suffix}/{opponent}": audit["recomputed"][
            f"parent{i}_learner-parent{i}{suffix}/{opponent}"
        ]["win_ci95"][0]
        > 0
        for i in (1, 2)
        for suffix in ("", "_response")
        for opponent, _ in OPPONENTS
    }
    assert primary == audit["primary_positive_lower_bounds"]
    assert all(primary.values()) == audit["replicated_learner_gain"]
    assert all(primary.values()) == analysis["replicated_learner_gain"]
    assert audit["shared_unique_deal_streams"] == 5000
    assert sum(r["matches"] for r in analysis["outcomes"].values()) == 160000
    assert audit["evaluation_matches"] == 160000
    assert audit["evaluation_decisions"] == sum(
        r["decisions"] for r in analysis["outcomes"].values()
    )
    assert set(analysis["training"]) == {
        f"parent{i}_{arm}" for i in (1, 2) for arm in ("learner", "lambda1")
    }
    totals = {
        key: sum(r["counters"][key] for r in analysis["training"].values())
        for key in ("decisions", "rounds", "matches")
    }
    assert totals == audit["training_totals"] and totals["decisions"] == 34078720
    prior = []
    for index in (1, 2):
        path = Path(f"runs/response_study/parent{index}_response/training_audit.json")
        assert inputs[str(path)] == sha(path)
        prior.append(read(path)["counters"])
    reused = {key: sum(r[key] for r in prior) for key in totals}
    assert reused["decisions"] == 17039360
    tables = {
        folder / "learner_primary_table.tex": win_table(
            analysis["contrasts"],
            (("learner", ""), ("learner", "_response")),
            ("Learner minus parent", "Learner minus response"),
        ),
        folder / "learner_lambda1_table.tex": win_table(
            analysis["contrasts"],
            (("lambda1", ""), ("learner", "_lambda1")),
            (r"$\lambda=1$ minus parent", r"Learner minus $\lambda=1$"),
        ),
    }
    result = {
        "status": "passed",
        "inputs": inputs,
        "outputs": {
            str(path): hashlib.sha256(content.encode("utf-8")).hexdigest()
            for path, content in tables.items()
        },
        "expected_audit_sha256": expected_audit_sha,
        "training_totals": totals,
        "reused_response_training_totals": reused,
        "evaluation_matches": audit["evaluation_matches"],
        "evaluation_decisions": audit["evaluation_decisions"],
        "shared_unique_deal_streams": 5000,
        "outcomes": analysis["outcomes"],
        "contrasts": analysis["contrasts"],
        "primary_positive_lower_bounds": primary,
        "replicated_learner_gain": all(primary.values()),
        "scope": "Completed development comparison on shared 24M deal streams. New training and reused response dependencies are separate. All 16 outcomes and 24 paired contrasts retained. The intervention changes returns and optimization; lambda1 retains a finite-rollout tail bootstrap. No new simulation, independent final confirmation or training-seed uncertainty estimate.",
    }
    ledger = folder / "learner_evidence.json"
    outputs = {**tables, ledger: json.dumps(result, indent=2) + "\n"}
    assert all(sha(path) == digest for path, digest in inputs.items())
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Refusing to replace different evidence: {path}")
    folder.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    assert all(sha(path) == digest for path, digest in result["outputs"].items())
    print(json.dumps({"output": str(ledger), "sha256": sha(ledger)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/paper/generated"))
    args = parser.parse_args()
    export(args.audit_sha256, args.output_dir)
