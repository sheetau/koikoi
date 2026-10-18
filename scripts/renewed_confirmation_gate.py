"""Read-only qualification for a separate, prospective final confirmation."""

import argparse
import json
from pathlib import Path

from scripts.renewed_continuation_analysis import decisions, load_points
from scripts.run_arbitration_study import read, sha

PROTOCOL_SHA = "b380344727a3ac878e1c542b1754d6eac478ae66269cd662c60c0cd85dec46aa"


def qualification(contrasts):
    verdicts = decisions(contrasts)
    return {
        "qualified": verdicts["replicated_strength_gain"],
        "development_claims": verdicts,
        "candidate_policy": "renewed_public",
        "parents": [1, 2],
        "endpoint_update": 520,
        "automatic_promotion": False,
    }


def check(root):
    process = root / "process.json"
    if not process.exists() or read(process)["status"] != "complete":
        raise ValueError("Completed renewed study required")
    audit_path, receipt_path = (
        root / "independent_audit.json",
        root / "summary_receipt.json",
    )
    if not audit_path.exists() or read(audit_path)["status"] != "passed":
        raise ValueError("Passed independent audit required")
    if not receipt_path.exists() or read(receipt_path)["status"] != "complete":
        raise ValueError("Completed study summary required")
    audit, receipt = read(audit_path), read(receipt_path)
    protocol = root / "protocol.json"
    if sha(protocol) != PROTOCOL_SHA or receipt["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Unexpected development protocol")
    for path, expected in audit["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"Changed audited input: {path}")
    if receipt["independent_audit_sha256"] != sha(audit_path):
        raise ValueError("Audit receipt mismatch")
    if receipt["summary_sha256"] != sha(root / "summary.md"):
        raise ValueError("Summary receipt mismatch")
    if receipt["summarizer_sha256"] != sha(
        "scripts/summarize_renewed_continuation_study.py"
    ):
        raise ValueError("Unexpected summarizer")
    produced = read(root / "analysis.json")
    result = qualification(audit["contrasts"])
    if produced["status"] != "complete" or produced["contrasts"] != audit["contrasts"]:
        raise ValueError("Producer/audit contrast mismatch")
    if not (
        produced["decisions"]
        == audit["decisions"]
        == receipt["decisions"]
        == result["development_claims"]
    ):
        raise ValueError("Inconsistent qualification claims")
    if (
        audit["evaluation_replay_work"]["completed_matches"] != 40960
        or audit["distinct_evaluation_deals"] != 2048
        or receipt["evaluation_work"] != audit["evaluation_replay_work"]
    ):
        raise ValueError("Incomplete development evaluation")
    load_points(root)
    result.update(
        protocol_sha256=PROTOCOL_SHA,
        independent_audit_sha256=sha(audit_path),
        summary_sha256=sha(root / "summary.md"),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("runs/renewed_continuation_study")
    )
    print(json.dumps(check(parser.parse_args().study)))
