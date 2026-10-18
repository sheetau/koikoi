"""Export the completed observer self-play comparison after its independent audit.

The caller pins the final audit hash explicitly. No interim results are exported.
"""

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

BASE = Path("runs/observer_selfplay_study")
PROTOCOL_SHA = "ed98449175207a42782d1083e9efa69f4c76c6d43af4464587f83c8b8963c3ef"
OPPONENTS = (("rl-wp-assisted", "WP"), ("rl-point-assisted", "Point"))
COMPARISONS = (
    ("observer", ""),
    ("signed", ""),
    ("selfplay", ""),
    ("observer", "_selfplay"),
    ("observer", "_signed"),
    ("signed", "_selfplay"),
)
TRACE = Path("runs/observer_trace_identity_cpu.json")
TRACE_SHA = "830420c471986c33b84bfd3a9a8d2b86f7aaef5bd343637856d4e682b4006d60"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison_table(contrasts, comparisons, headers, metric="win"):
    if metric not in ("win", "point"):
        raise ValueError("Expected win or point contrast")
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
                factor = 100 if metric == "win" else 1
                values = [
                    factor * row[metric + "_change"],
                    *[factor * x for x in row[metric + "_ci95"]],
                ]
                cells.append(f"${values[0]:+.3f}$ [{values[1]:.3f}, {values[2]:.3f}]")
            lines.append(f"{index} & {label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def export(expected_audit_sha, folder=Path("docs/paper/generated")):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha):
        raise ValueError("Pin the independent audit with a lowercase SHA256")
    state = read(BASE / "process.json")
    if state["status"] != "complete":
        raise ValueError("The observer self-play study is not complete")
    audit_path = BASE / "independent_audit.json"
    assert sha(audit_path) == expected_audit_sha
    audit, analysis = read(audit_path), read(BASE / "analysis.json")
    assert audit["status"] == analysis["status"] == "passed"
    assert analysis["kind"] == "observer_selfplay_development"
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
    assert sha(TRACE) == TRACE_SHA
    trace = read(TRACE)
    assert trace["status"] == "passed" and trace["pathwise_exact_cases"] == 6048
    assert trace["live_protocol_sha256"] == PROTOCOL_SHA
    assert len(trace["toy"]["cases"]) == 16
    truth = Fraction(trace["toy"]["true_gradient"])
    covariance = Fraction(trace["toy"]["cross_observer_score_covariance"])
    assert truth == Fraction(3, 800) and covariance == Fraction(-20301, 160000)
    for key, row in trace["toy"]["cases"].items():
        lam = Fraction(key.split("lambda")[1])
        expected_bias = (
            0
            if "_observer_" in key
            else covariance * (1 - lam if key.startswith("terminal_") else 1)
        )
        assert Fraction(row["exact_bias"]) == expected_bias
        assert Fraction(row["exact_gradient"]) == truth + expected_bias
        assert abs(row["cpu_gradient"] - float(truth + expected_bias)) < 3e-8
    inputs[str(TRACE)] = TRACE_SHA
    inputs.update(trace["inputs"])
    outcome_keys = {
        f"parent{i}{suffix}/{opponent}"
        for i in (1, 2)
        for suffix in ("", "_selfplay", "_observer", "_signed")
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
        f"parent{i}_observer-parent{i}{suffix}/{opponent}": audit["recomputed"][
            f"parent{i}_observer-parent{i}{suffix}/{opponent}"
        ]["win_ci95"][0]
        > 0
        for i in (1, 2)
        for suffix in ("", "_signed")
        for opponent, _ in OPPONENTS
    }
    assert primary == audit["primary_positive_lower_bounds"]
    assert all(primary.values()) == audit["replicated_observer_gain"]
    assert all(primary.values()) == analysis["replicated_observer_gain"]
    assert audit["shared_unique_deal_streams"] == 5000
    assert sum(r["matches"] for r in analysis["outcomes"].values()) == 160000
    assert audit["evaluation_matches"] == 160000
    assert audit["evaluation_decisions"] == sum(
        r["decisions"] for r in analysis["outcomes"].values()
    )
    assert set(analysis["training"]) == {
        f"parent{i}_{arm}" for i in (1, 2) for arm in ("observer", "signed")
    }
    totals = {
        key: sum(r["counters"][key] for r in analysis["training"].values())
        for key in ("decisions", "rounds", "matches", "value_rows")
    }
    assert totals == audit["training_totals"] and totals["decisions"] == 34078720
    prior = []
    for index in (1, 2):
        path = Path(f"runs/response_study/parent{index}_selfplay/training_audit.json")
        assert inputs[str(path)] == sha(path)
        prior.append(read(path)["counters"])
    reused = {
        key: sum(r[key] for r in prior) for key in ("decisions", "rounds", "matches")
    }
    assert reused["decisions"] == 17039360
    assert totals["value_rows"] == 68157440
    assert audit["matched_warmup"] == analysis["matched_warmup"]
    assert set(audit["matched_warmup"]) == {"parent1", "parent2"}
    for index in (1, 2):
        warmup = audit["matched_warmup"][f"parent{index}"]
        assert warmup["status"] == "passed"
        for arm in ("observer", "signed"):
            path = BASE / f"parent{index}_{arm}" / "warmup.pt"
            assert inputs[str(path)] == sha(path) == warmup[arm + "_sha256"]
    comparisons = (("observer", ""), ("observer", "_signed"))
    headers = ("Observer minus parent", "Observer minus signed")
    tables = {
        folder / "observer_selfplay_primary_table.tex": comparison_table(
            analysis["contrasts"], comparisons, headers
        ),
        folder / "observer_selfplay_points_table.tex": comparison_table(
            analysis["contrasts"], comparisons, headers, metric="point"
        ),
        folder / "observer_selfplay_historical_table.tex": comparison_table(
            analysis["contrasts"],
            (("selfplay", ""), ("observer", "_selfplay")),
            ("Self-play minus parent", "Observer minus self-play"),
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
        "matched_warmup": audit["matched_warmup"],
        "trace_identity": {
            "pathwise_exact_cases": trace["pathwise_exact_cases"],
            "toy": trace["toy"],
            "scope": trace["scope"],
        },
        "reused_selfplay_training_totals": reused,
        "evaluation_matches": audit["evaluation_matches"],
        "evaluation_decisions": audit["evaluation_decisions"],
        "shared_unique_deal_streams": 5000,
        "outcomes": analysis["outcomes"],
        "contrasts": analysis["contrasts"],
        "primary_positive_lower_bounds": primary,
        "replicated_observer_gain": all(primary.values()),
        "scope": "Completed development comparison on shared 27M deal streams. New training/value rows and reused ordinary-selfplay dependencies are separate. All 16 outcomes and 24 paired contrasts retained. Observer and signed arms share critic fitting and differ in policy returns; historical selfplay is a full-intervention control. No new simulation, independent final confirmation or training-seed uncertainty estimate.",
    }
    ledger = folder / "observer_selfplay_evidence.json"
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
