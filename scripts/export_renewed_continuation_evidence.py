"""Export the completed fixed/renewed chance comparison into the manuscript."""

from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def main():
    root = Path("runs/renewed_continuation_study")
    audit_path, receipt_path = (
        root / "independent_audit.json",
        root / "summary_receipt.json",
    )
    assert (
        sha(audit_path)
        == "0f920e4be44cf833964975f3f7784846eb556c32f215011597a5cc892480da0a"
    )
    audit, receipt = read(audit_path), read(receipt_path)
    assert read(root / "process.json")["status"] == receipt["status"] == "complete"
    assert audit["status"] == "passed" and not any(audit["decisions"].values())
    assert audit["decisions"] == receipt["decisions"]
    assert receipt["independent_audit_sha256"] == sha(audit_path)
    assert receipt["summary_sha256"] == sha(root / "summary.md")
    assert receipt["evaluation_work"] == audit["evaluation_replay_work"]
    assert receipt["training_work"] == {
        key: sum(row[key] for row in audit["training_counters"])
        for key in receipt["training_work"]
    }
    assert receipt["evaluation_work"]["completed_matches"] == 40960
    assert receipt["distinct_evaluation_deals"] == 2048
    assert (
        len(audit["contrasts"])
        == len({(r["parent"], r["scope"], r["comparison"]) for r in audit["contrasts"]})
        == 66
    )
    primary = [
        r
        for r in audit["contrasts"]
        if r["scope"] == "mixture"
        and r["comparison"]
        in ("renewed_public-minus-parent", "renewed_public-minus-fixed_public")
    ]
    assert len(primary) == 4 and all(
        r["win"]["interval95"][0] <= 0 <= r["win"]["interval95"][1] for r in primary
    )
    inputs = {
        **audit["inputs"],
        **{
            str(p): sha(p)
            for p in (audit_path, receipt_path, root / "summary.md", Path(__file__))
        },
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Reference & Win change (pp) & Point change \\",
        r"\midrule",
    ]
    for row in primary:
        cells = []
        for metric, scale in (("win", 100), ("point", 1)):
            value = row[metric]
            lo, hi = value["interval95"]
            cells.append(
                f"{scale * value['change']:+.3f} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
            )
        reference = (
            "Original parent"
            if row["comparison"] == "renewed_public-minus-parent"
            else "Fixed public memory"
        )
        rows.append(f"{row['parent']} & {reference} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", ""]
    folder = Path("docs/paper/generated")
    table = folder / "renewed_continuation_table.tex"
    table.write_bytes("\n".join(rows).encode("utf-8"))
    evidence = folder / "renewed_continuation_evidence.json"
    write(
        evidence,
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": {str(table): sha(table)},
            "contrasts": audit["contrasts"],
            "decisions": audit["decisions"],
            **{
                name: receipt[name]
                for name in (
                    "training_work",
                    "evaluation_work",
                    "root_replay_work",
                    "prediction_calls",
                    "distinct_evaluation_deals",
                    "distinct_training_chance_keys",
                )
            },
            "strict_input_checks": audit["strict_input_checks"],
            "scope": "Completed renewed-chance study only; all three replication criteria fail; no final confirmation or promotion.",
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
