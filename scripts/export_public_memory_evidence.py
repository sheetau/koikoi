"""Export only independently audited final-round memory outcomes to the paper."""

from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def main():
    root = Path("runs/public_memory_study")
    audit_path = root / "independent_audit.json"
    assert (
        sha(audit_path)
        == "a5a9f350f2388294cdd106758d60eb453a2f03a24ff41f192436a04d4925acd7"
    )
    audit = read(audit_path)
    assert audit["status"] == "passed" and not audit["replicated_gain"]
    inputs = {**audit["inputs"], str(audit_path): sha(audit_path)}
    receipt_path = root / "summary_receipt.json"
    assert (
        sha(receipt_path)
        == "a566ffd3a66a755d6ff260750a67f909a7cab426d1ad1550bb86f9db8e26c9a9"
    )
    receipt = read(receipt_path)
    assert receipt["status"] == "complete" and not receipt["replicated_gain"]
    assert receipt["independent_audit_sha256"] == sha(audit_path)
    assert receipt["evaluation_work"] == audit["evaluation_replay_work"]
    inputs[str(receipt_path)] = sha(receipt_path)
    inputs[str(Path(__file__))] = sha(__file__)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    primary = [
        r
        for r in audit["contrasts"]
        if r["scope"] == "mixture" and r["comparison"] != "masked-minus-parent"
    ]
    assert len(primary) == 4
    assert all(
        r[m]["interval95"][0] <= 0 <= r[m]["interval95"][1]
        for r in audit["contrasts"]
        for m in ("win", "point")
    )
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parent & Reference & Win change (pp) & Point change \\",
        r"\midrule",
    ]
    for row in primary:
        cells = []
        for metric, scale in (("win", 100), ("point", 1)):
            data = row[metric]
            lo, hi = data["interval95"]
            cells.append(
                f"{scale * data['change']:+.3f} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
            )
        reference = (
            "Masked memory"
            if row["comparison"] == "public-minus-masked"
            else "Original parent"
        )
        rows.append(f"{row['parent']} & {reference} & " + " & ".join(cells) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    folder = Path("docs/paper/generated")
    table = folder / "public_memory_table.tex"
    table.write_bytes("\n".join(rows).encode("utf-8"))
    write(
        folder / "public_memory_evidence.json",
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": {str(table): sha(table)},
            "contrasts": audit["contrasts"],
            "training_work": receipt["training_work"],
            "evaluation_work": receipt["evaluation_work"],
            "prediction_calls": receipt["prediction_calls"],
            "strict_input_checks": audit["strict_input_checks"],
            "distinct_deals": receipt["distinct_deals"],
            "scope": "Completed final-round memory comparison; no multi-round result.",
        },
    )
    print(
        {
            "inputs": len(inputs),
            "table_sha256": sha(table),
            "evidence_sha256": sha(folder / "public_memory_evidence.json"),
        }
    )


if __name__ == "__main__":
    main()
