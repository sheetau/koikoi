"""Summarize only a completed, independently audited final-round pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_arbitration_study import sha, write

PHASES = ("discard", "hand capture", "draw capture")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(folder):
    output, markdown = folder / "summary.json", folder / "summary.md"
    if output.exists() or markdown.exists():
        raise ValueError("Preserve previous summaries")
    report, independent = [
        read(folder / name) for name in ("audit.json", "independent_audit.json")
    ]
    assert report["status"] == independent["status"] == "passed"
    assert all(sha(path) == digest for path, digest in independent["inputs"].items())
    sources = read(folder / "source.json")
    rows = independent["summaries"]
    metrics, accepted, positive_references = [], [], []
    for parent in (1, 2):
        for phase in range(3):
            selected = [
                r for r in rows if r["parent"] == parent and r["phase"] == phase
            ]
            errors = [
                error for r in selected for error in r["proposal_errors"].values()
            ]
            times = [r["act_seconds"] for r in selected]
            metrics.append(
                {
                    "parent": parent,
                    "phase": phase,
                    "roots": len(selected),
                    "source_matches": 64,
                    "missing_opportunities": 64 - len(selected),
                    "proposed": sum(r["proposed"] for r in selected),
                    "changed": sum(r["changed"] for r in selected),
                    "alternative_count": len(errors),
                    "same_proposal_reference_rmse": float(
                        np.sqrt(np.mean(np.square(errors)))
                    )
                    if errors
                    else None,
                    "same_proposal_reference_mae": float(np.mean(np.abs(errors)))
                    if errors
                    else None,
                    "act_seconds": sum(times),
                    "act_mean_seconds": float(np.mean(times)) if times else None,
                    "act_median_seconds": float(np.median(times)) if times else None,
                    "reference_seconds": sum(r["reference_seconds"] for r in selected),
                }
            )
    for row in rows:
        metadata = {
            k: row[k]
            for k in ("root_index", "case_index", "parent", "phase", "base", "action")
        }
        if row["changed"]:
            accepted.append(
                {
                    **metadata,
                    "reference": row["references"][str(row["action"])],
                    "act_seconds": row["act_seconds"],
                }
            )
        for action, value in row["references"].items():
            if value["lower"] > 0:
                positive_references.append(
                    {**metadata, "alternative": int(action), "reference": value}
                )
    actual = independent["contrasts"]
    assert len(actual) == 12
    result = {
        "status": "complete",
        "inputs": {
            str(p): sha(p)
            for p in (
                Path(__file__),
                folder / "audit.json",
                folder / "independent_audit.json",
            )
        },
        "source_unique_deals": 32,
        "source_work": report["source_work"],
        "actual_pilot_work": report["actual_pilot_work"],
        "root_count": report["root_count"],
        "source_matches_ending_before_round8": sum(
            r["terminal_round"] < 7 for r in sources
        ),
        "metrics": metrics,
        "accepted_actions": accepted,
        "positive_unadjusted_reference_margins": positive_references,
        "actual_contrasts": actual,
        "search_totals": report["totals"],
        "scope": "All first-per-phase opportunities and missing cases retained. Same-proposal references measure Monte Carlo error against an approximate uniform target, not posterior calibration. Actual pilot uses32 shared deals; phase interventions are separate policies, and descriptive intervals do not establish replicated population strength or justify automatic promotion.",
    }
    lines = [
        "# Natural final-round search: complete audited pilot",
        "",
        "128 source matches and384 candidate matches share32 distinct deals. Each phase defines a separate policy that can change one action. All missing opportunities and unchanged choices remain in the actual comparisons.",
        "",
        "| Parent | Opponent | Phase | Win change (pp) | 95% interval (pp) | Point change | 95% interval |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in actual:
        win, points = row["win"], row["points"]
        lo, hi = (100 * x for x in win["interval95"])
        plo, phi = points["interval95"]
        lines.append(
            f"| {row['parent']} | {row['opponent']} | {PHASES[row['phase']]} | {100 * win['mean']:.6f} | [{lo:.6f}, {hi:.6f}] | {points['mean']:.6f} | [{plo:.6f}, {phi:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Intervals resample16 whole duplicate-deal pairs per cell. These are small-pilot descriptive intervals; a zero empirical interval is not a population equivalence guarantee.",
            "",
            "| Parent | Phase | Roots | Missing | Proposed | Changed | Reference RMSE | Act seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics:
        rmse = row["same_proposal_reference_rmse"]
        error = "n/a" if rmse is None else f"{rmse:.9f}"
        lines.append(
            f"| {row['parent']} | {PHASES[row['phase']]} | {row['roots']} | {row['missing_opportunities']} | {row['proposed']} | {row['changed']} | {error} | {row['act_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "References use independent particles from the same approximate uniform hidden-world/model mixture. Their error does not measure the effect of omitting opponent-action history. Positive2.5-SE reference margins are unadjusted diagnostics, not actual wins.",
            "",
            "All accepted choices, every positive reference margin and separate source/candidate/branch work totals are retained in summary.json. Timing covers these runs and is not a separately isolated deployment benchmark.",
            "",
        ]
    )
    write(output, result)
    markdown.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "summary_sha256": sha(output),
                "roots": len(rows),
                "changed": len(accepted),
                "metrics": metrics,
                "contrasts": actual,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
