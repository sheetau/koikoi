"""Describe fallback causes in the closed uniform cohort; no strength inference."""

import json
from collections import Counter
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def summarize(base, evidence):
    ledger = read(evidence)
    summary = ledger["summary"]
    protocol = read(base / "protocol.json")
    if ledger["status"] != "passed" or read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete checked cohort required")
    for path, digest in ledger["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed export input: {path}")
    counts = {p: Counter() for p in ("0:wp", "0:point", "1:wp", "1:point")}
    seen, bindings = [], {}
    for chunk in range(512):
        folder = base / f"chunk_{chunk:04}"
        complete = folder / "complete.json"
        if sha(complete) != summary["bindings"][str(complete)]:
            raise ValueError("Changed complete receipt")
        records = folder / "records.json"
        if sha(records) != read(complete)["records_sha256"]:
            raise ValueError("Changed source records")
        bindings[str(records)] = sha(records)
        for row in read(records):
            case = row["case"]
            seen.append(case)
            count = counts[case["population"]]
            count["sources"] += 1
            if row["status"] == "missing_root":
                prior = any(
                    event["round"] < 7 and len(event["after"]) == 1
                    for event in row["source"]["identification_events"]
                )
                count["missing_root_with_prior_identity" if prior else "missing_root_without_prior_identity"] += 1
                continue
            if row["status"] != "evaluated":
                raise ValueError(f"Unexpected source status: {row['status']}")
            count["evaluated"] += 1
            selection = row["selection"]
            count[f"selection:{selection['mode']}:{selection['reason']}"] += 1
            if selection["parent_q"] == "1":
                count["parent_value_one"] += 1
            for name, phase in row["phases"].items():
                if phase["refusal"]:
                    count[f"refusal:{name}:{phase['refusal']}"] += 1
    if seen != protocol["cases"] or len(seen) != 8192:
        raise ValueError("Cohort order/count differs")
    return {
        "status": "passed",
        "scope": "Post-completion descriptive diagnosis; no new samples or revised inference.",
        "evidence_sha256": sha(evidence),
        "script_sha256": sha(__file__),
        "records": bindings,
        "counts": {k: dict(v) for k, v in counts.items()},
    }


if __name__ == "__main__":
    result = summarize(
        Path("runs/uniform_controller_evaluation"),
        Path("docs/paper/generated/uniform_results_evidence.json"),
    )
    destination = Path("runs/uniform_controller_opportunities.json")
    if destination.exists():
        raise FileExistsError(destination)
    write(destination, result)
    print(json.dumps(result["counts"], indent=2))
