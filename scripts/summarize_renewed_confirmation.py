"""Render every audited confirmation result, including a negative outcome."""

import argparse
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def summarize(root):
    summary, receipt = root / "summary.md", root / "summary_receipt.json"
    if summary.exists() or receipt.exists():
        raise ValueError("Refusing to overwrite confirmation summary")
    if read(root / "process.json")["status"] != "complete":
        raise ValueError("Completed confirmation required")
    audit_path = root / "independent_audit.json"
    audit, produced = read(audit_path), read(root / "analysis.json")
    if audit["status"] != "passed" or produced["status"] != "complete":
        raise ValueError("Passed independent confirmation audit required")
    for path, digest in audit["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed audited input: {path}")
    for key in ("contrasts", "absolute", "confirmed_strength_gain"):
        if audit[key] != produced[key]:
            raise ValueError("Producer/audit results disagree")
    assert len(audit["contrasts"]) == len(audit["absolute"]) == 18
    assert audit["evaluation_replay_work"]["completed_matches"] == 98304
    assert audit["distinct_evaluation_deals"] == 8192
    lines = [
        "# Final renewed-chance confirmation",
        "",
        f"Replicated primary strength criterion: **{audit['confirmed_strength_gain']}**.",
        "",
        "98,304 complete matches on 8,192 shared fresh deals; 4,096 paired deals per opponent, both seats and two parent seeds. Win scores count draws as one half. Intervals are paired stratified 95% bootstrap intervals with 10,000 resamples.",
        "",
        "The primary claim concerns the balanced frozen WP/Point benchmark. Opponent-specific and point results are secondary. Public-memory and interaction evidence retains its development status; this confirmation does not retest either mechanism. No automatic promotion follows.",
        "",
        "## All paired contrasts",
        "",
        "| Parent | Scope | Comparison | Win change, pp [95% CI] | Point change [95% CI] |",
        "|---|---|---|---:|---:|",
    ]

    def cell(metric, key, scale):
        lo, hi = metric["interval95"]
        return f"{scale * metric[key]:+.6f} [{scale * lo:+.6f}, {scale * hi:+.6f}]"

    for row in audit["contrasts"]:
        lines.append(
            f"| {row['parent']} | {row['scope']} | {row['comparison']} | {cell(row['win'], 'change', 100)} | {cell(row['point'], 'change', 1)} |"
        )
    lines += [
        "",
        "## All absolute performance cells",
        "",
        "| Parent | Scope | Policy | Win score, % [95% CI] | Point difference [95% CI] |",
        "|---|---|---|---:|---:|",
    ]
    for row in audit["absolute"]:
        lines.append(
            f"| {row['parent']} | {row['scope']} | {row['policy']} | {cell(row['win'], 'mean', 100)} | {cell(row['point'], 'mean', 1)} |"
        )
    lines += [
        "",
        f"Evaluation and replay work per pass: {audit['evaluation_replay_work']}. Historical public-input checks: {audit['strict_input_checks']}. Independent replay adds zero neural calls and zero independent deals.",
        "",
        f"Development mechanism claims (not upgraded): {audit['development_mechanism_claims']}.",
        "",
        f"Independent audit SHA256: {sha(audit_path)}.",
    ]
    with summary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    write(
        receipt,
        {
            "status": "complete",
            "summary_sha256": sha(summary),
            "protocol_sha256": sha(root / "protocol.json"),
            "independent_audit_sha256": sha(audit_path),
            "summarizer_sha256": sha(__file__),
            "confirmed_strength_gain": audit["confirmed_strength_gain"],
            "development_mechanism_claims": audit["development_mechanism_claims"],
            "evaluation_work": audit["evaluation_replay_work"],
            "automatic_promotion": False,
        },
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(summarize(parser.parse_args().output))
