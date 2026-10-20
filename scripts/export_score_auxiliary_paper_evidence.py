"""Transcribe the completed auxiliary comparison into a manuscript table."""

import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export():
    source = Path("runs/score_auxiliary_results/evidence.json")
    audit_path = Path("runs/score_auxiliary_evaluation/audit.json")
    mechanism_path = Path("runs/score_auxiliary_mechanism/report.json")
    mechanism_audit = Path("runs/score_auxiliary_mechanism/independent_audit.json")
    if (
        sha(source)
        != "a8f3ac919fba4cfe1e10f0e3caa1fb23b5542fcb5e76beac2d10cfa5ecbaf311"
    ):
        raise ValueError("Unexpected completed result export")
    if (
        sha(audit_path)
        != "cf5f8fc9c5873dca98893ae3abac13722ec4e71f7c091f98464e5ca59d868ceb"
    ):
        raise ValueError("Unexpected evaluation audit")
    if (
        sha(mechanism_audit)
        != "d619258403def4da3aa71cbc3b02b45179abe964a73c6ccb70e802da2ea5ab52"
    ):
        raise ValueError("Unexpected fixed-state audit")
    evidence = read(source)
    assert evidence["status"] == "complete"
    assert read(audit_path)["status"] == read(mechanism_audit)["status"] == "passed"
    inputs = {}
    for collection in (evidence["inputs"], read(mechanism_audit)["inputs"]):
        for name, digest in collection.items():
            key = str(Path(name).resolve())
            if key in inputs and inputs[key] != digest:
                raise ValueError(f"Inconsistent evidence: {name}")
            inputs[key] = digest
    for path in (source, audit_path, mechanism_path, mechanism_audit, Path(__file__)):
        inputs[str(path.resolve())] = sha(path)
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError(f"Evidence changed: {name}")

    summary = evidence["summary"]
    assert len(summary["primary"]["rows"]) == 6
    assert len(summary["rows"]) == 120
    assert summary["full_matches"] == 131072
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Seed & Margin minus & Win change (pp) & Simultaneous 95\% interval \\",
        r"\midrule",
    ]
    for seed in (1, 2):
        for control, label in (
            ("none", "No auxiliary"),
            ("win", "Outcome"),
            ("parent", "Retained WP parent"),
        ):
            row = summary["primary"]["rows"][f"seed{seed}/mixture/score-{control}/win"]
            mean = 100 * row["mean"]
            low, high = (100 * value for value in row["simultaneous95"])
            lines.append(
                f"{seed} & {label} & ${mean:+.4f}$ & $[{low:+.4f}, {high:+.4f}]$"
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    target = Path("docs/paper/generated")
    table = target / "score_auxiliary_primary_table.tex"
    table.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    ledger = target / "score_auxiliary_evidence.json"
    ledger.write_bytes(
        (
            json.dumps(
                {
                    "inputs": inputs,
                    "outputs": {str(table): sha(table)},
                    "summary": summary,
                    "mechanism_report": read(mechanism_path),
                    "scope": "Transcription of the complete audited comparison; no new analysis, inference, matches or promotion.",
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    )
    print(
        json.dumps(
            {
                "inputs": len(inputs),
                "table_sha256": sha(table),
                "ledger_sha256": sha(ledger),
            }
        )
    )


if __name__ == "__main__":
    export()
