"""Export the completed multi-round memory comparison into the manuscript."""

from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def main():
    root = Path("runs/public_continuation_study")
    audit_path, receipt_path = (
        root / "independent_audit.json",
        root / "summary_receipt.json",
    )
    assert (
        sha(audit_path)
        == "c1694086e2f91a7b13005324b409957cdd4aee4f84d09e2640e3e01a8601c322"
    )
    audit, receipt = read(audit_path), read(receipt_path)
    assert read(root / "process.json")["status"] == receipt["status"] == "complete"
    assert audit["status"] == "passed" and not audit["replicated_gain"]
    assert not receipt["replicated_gain"]
    assert receipt["independent_audit_sha256"] == sha(audit_path)
    assert receipt["summary_sha256"] == sha(root / "summary.md")
    assert receipt["evaluation_work"] == audit["evaluation_replay_work"]
    assert receipt["training_work"] == {
        key: sum(row[key] for row in audit["training_counters"])
        for key in receipt["training_work"]
    }
    assert receipt["evaluation_work"]["completed_matches"] == 24576
    assert receipt["distinct_deals"] == 2048
    expected = {
        (p, contrast, scope)
        for p in (1, 2)
        for contrast in (
            "public-minus-masked",
            "public-minus-parent",
            "masked-minus-parent",
        )
        for scope in ("mixture", "wp", "point")
    }
    assert len(audit["contrasts"]) == 18
    assert {
        (r["parent"], r["comparison"], r["scope"]) for r in audit["contrasts"]
    } == expected
    assert all(
        r["win"]["interval95"][0] <= 0 <= r["win"]["interval95"][1]
        for r in audit["contrasts"]
    )
    point_effects = {
        (r["parent"], r["comparison"], r["scope"])
        for r in audit["contrasts"]
        if not r["point"]["interval95"][0] <= 0 <= r["point"]["interval95"][1]
    }
    assert point_effects == {
        (1, comparison, scope)
        for comparison in ("public-minus-masked", "masked-minus-parent")
        for scope in ("mixture", "point")
    }
    inputs = {
        **audit["inputs"],
        **{
            str(p): sha(p)
            for p in (audit_path, receipt_path, root / "summary.md", Path(__file__))
        },
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    primary = [
        row
        for row in audit["contrasts"]
        if row["scope"] == "mixture" and row["comparison"] != "masked-minus-parent"
    ]
    assert len(primary) == 4
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Reference & Win change (pp) & Point change \\",
        r"\midrule",
    ]
    for row in primary:
        cells = []
        for name, scale in (("win", 100), ("point", 1)):
            value = row[name]
            lo, hi = value["interval95"]
            cells.append(
                f"{scale * value['change']:+.3f} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
            )
        reference = (
            "Masked memory"
            if row["comparison"] == "public-minus-masked"
            else "Original parent"
        )
        rows.append(f"{row['parent']} & {reference} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    folder = Path("docs/paper/generated")
    table = folder / "public_continuation_table.tex"
    table.write_bytes("\n".join(rows).encode("utf-8"))
    evidence = folder / "public_continuation_evidence.json"
    write(
        evidence,
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": {str(table): sha(table)},
            "contrasts": audit["contrasts"],
            "replicated_gain": False,
            "training_work": receipt["training_work"],
            "evaluation_work": receipt["evaluation_work"],
            "root_replay_work": receipt["root_replay_work"],
            "prediction_calls": receipt["prediction_calls"],
            "strict_input_checks": audit["strict_input_checks"],
            "distinct_deals": receipt["distinct_deals"],
            "scope": "Completed fixed-root multi-round memory comparison; no renewed-chance result or new simulations.",
        },
    )
    print(
        {
            "inputs": len(inputs),
            "table_sha256": sha(table),
            "evidence_sha256": sha(evidence),
        }
    )


if __name__ == "__main__":
    main()
