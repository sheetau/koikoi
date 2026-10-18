"""Export the complete public-seat comparison after its independent raw-pair audit."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts import export_observer_selfplay_paper_evidence as tables


def export(base, expected_protocol_sha, expected_audit_sha, folder):
    for digest in (expected_protocol_sha, expected_audit_sha):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                "Pin the protocol and independent audit with lowercase SHA256"
            )
    state = tables.read(base / "process.json")
    if state["status"] != "complete":
        raise ValueError("The public-seat study is not complete")
    protocol = tables.read(base / "protocol.json")
    if protocol["kind"] != "seat_observer_development":
        raise ValueError("Capability and smoke runs are not strength evidence")
    assert tables.sha(base / "protocol.json") == expected_protocol_sha
    audit_path = base / "independent_audit.json"
    assert tables.sha(audit_path) == expected_audit_sha
    audit, analysis = tables.read(audit_path), tables.read(base / "analysis.json")
    assert audit["status"] == analysis["status"] == "passed"
    assert audit["kind"] == analysis["kind"] == protocol["kind"]
    assert (
        state["protocol_sha256"] == analysis["protocol_sha256"] == expected_protocol_sha
    )
    assert state["analysis_sha256"] == tables.sha(base / "analysis.json")
    inputs = audit["inputs"].copy()
    for name in ("protocol.json", "process.json", "analysis.json"):
        path = base / name
        assert inputs[str(path)] == tables.sha(path)
    assert all(tables.sha(path) == digest for path, digest in inputs.items())
    for path in (audit_path, Path(__file__), Path(tables.__file__)):
        inputs[str(path)] = tables.sha(path)

    comparisons = (
        ("public", ""),
        ("masked", ""),
        ("observer", ""),
        ("public", "_observer"),
        ("public", "_masked"),
        ("masked", "_observer"),
    )
    outcomes = {
        f"parent{i}{suffix}/{opponent}"
        for i in (1, 2)
        for suffix in ("", "_observer", "_public", "_masked")
        for opponent, _ in tables.OPPONENTS
    }
    contrasts = {
        f"parent{i}_{arm}-parent{i}{control}/{opponent}"
        for i in (1, 2)
        for arm, control in comparisons
        for opponent, _ in tables.OPPONENTS
    }
    assert set(analysis["outcomes"]) == outcomes
    assert set(analysis["contrasts"]) == contrasts
    assert set(audit["recomputed"]) == outcomes | contrasts
    for key in outcomes | contrasts:
        absolute = key in outcomes
        row = analysis["outcomes" if absolute else "contrasts"][key]
        names = (
            ("win_score", "point_difference", "points_ci95")
            if absolute
            else ("win_change", "point_change", "point_ci95")
        )
        actual = [row[names[0]], row[names[1]], *row["win_ci95"], *row[names[2]]]
        independent = audit["recomputed"][key]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual,
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
        f"parent{i}_public-parent{i}{control}/{opponent}": audit["recomputed"][
            f"parent{i}_public-parent{i}{control}/{opponent}"
        ]["win_ci95"][0]
        > 0
        for i in (1, 2)
        for control in ("", "_masked")
        for opponent, _ in tables.OPPONENTS
    }
    assert primary == audit["primary_positive_lower_bounds"]
    qualified = all(primary.values())
    assert (
        qualified == audit["replicated_seat_gain"] == analysis["replicated_seat_gain"]
    )
    assert audit["shared_unique_deal_streams"] == 5000
    assert audit["evaluation_matches"] == 160000
    assert sum(row["matches"] for row in analysis["outcomes"].values()) == 160000
    assert audit["evaluation_decisions"] == sum(
        row["decisions"] for row in analysis["outcomes"].values()
    )
    assert set(analysis["training"]) == {
        f"parent{i}_{arm}" for i in (1, 2) for arm in ("public", "masked")
    }
    totals = {
        key: sum(row["counters"][key] for row in analysis["training"].values())
        for key in ("decisions", "rounds", "matches", "value_rows")
    }
    assert totals == audit["training_totals"]
    assert totals["decisions"] == 34078720 and totals["value_rows"] == 68157440
    assert set(audit["matched_warmup"]) == {"parent1", "parent2"}
    for parent, row in audit["matched_warmup"].items():
        assert row["status"] == "passed"
        for key, value in row.items():
            assert analysis["matched_warmup"][parent][key] == value
    source = Path(protocol["source_study"])
    reused = []
    for i in (1, 2):
        path = source / f"parent{i}_observer" / "training_audit.json"
        assert inputs[str(path)] == tables.sha(path)
        reused.append(tables.read(path)["counters"])
    reused_totals = {key: sum(row[key] for row in reused) for key in totals}
    assert reused_totals["decisions"] == 17039360
    assert reused_totals["value_rows"] == 34078720

    primary_comparisons = (("public", ""), ("public", "_masked"))
    headers = ("Public minus parent", "Public minus masked")
    outputs = {
        folder / "seat_observer_primary_table.tex": tables.comparison_table(
            analysis["contrasts"], primary_comparisons, headers
        ),
        folder / "seat_observer_points_table.tex": tables.comparison_table(
            analysis["contrasts"], primary_comparisons, headers, metric="point"
        ),
        folder / "seat_observer_historical_table.tex": tables.comparison_table(
            analysis["contrasts"],
            (("public", "_observer"), ("masked", "_observer")),
            ("Public minus archived observer", "Masked minus archived observer"),
        ),
    }
    result = {
        "status": "passed",
        "inputs": inputs,
        "outputs": {
            str(p): hashlib.sha256(s.encode("utf-8")).hexdigest()
            for p, s in outputs.items()
        },
        "expected_protocol_sha256": expected_protocol_sha,
        "expected_audit_sha256": expected_audit_sha,
        "training_totals": totals,
        "reused_observer_training_totals": reused_totals,
        "matched_warmup": audit["matched_warmup"],
        "evaluation_matches": audit["evaluation_matches"],
        "evaluation_decisions": audit["evaluation_decisions"],
        "shared_unique_deal_streams": 5000,
        "outcomes": analysis["outcomes"],
        "contrasts": analysis["contrasts"],
        "primary_positive_lower_bounds": primary,
        "replicated_seat_gain": qualified,
        "scope": "Completed development comparison on shared 28M deals. All 16 outcomes and 24 paired contrasts retained. Public-minus-masked isolates the added public information with matched architecture; archived-observer contrasts are secondary. New and reused training work are separate. No new simulation, independent final confirmation, inference replay or training-seed uncertainty estimate.",
    }
    ledger = folder / "seat_observer_evidence.json"
    outputs[ledger] = json.dumps(result, indent=2) + "\n"
    assert all(tables.sha(path) == digest for path, digest in inputs.items())
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Refusing to replace different evidence: {path}")
    folder.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    assert all(tables.sha(path) == digest for path, digest in result["outputs"].items())
    print(json.dumps({"output": str(ledger), "sha256": tables.sha(ledger)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/seat_observer_study"))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/paper/generated"))
    args = parser.parse_args()
    export(args.base, args.protocol_sha256, args.audit_sha256, args.output_dir)
