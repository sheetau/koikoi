"""Summarize only the independently completed fixed terminal pilot."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_arbitration_study import sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(folder):
    output = folder / "summary.json"
    if output.exists():
        raise ValueError("Preserve previous summary")
    report, independent = (
        read(folder / "audit.json"),
        read(folder / "independent_audit.json"),
    )
    assert report["status"] == independent["status"] == "passed"
    assert all(sha(p) == h for p, h in independent["inputs"].items())
    metrics, accepted = [], []
    for parent in (1, 2):
        roots = [r for r in independent["summaries"] if r["parent"] == parent]
        for mode in ("uniform", "history"):
            errors = [v for r in roots for v in r["proposal_errors"][mode].values()]
            entry = report["totals"][f"parent{parent}_{mode}"]
            metrics.append(
                {
                    "parent": parent,
                    "mode": mode,
                    "roots": len(roots),
                    "targets": len(errors),
                    "all_action_conditional_reference_rmse": float(
                        np.sqrt(np.mean(np.square(errors)))
                    )
                    if errors
                    else None,
                    "all_action_conditional_reference_mae": float(
                        np.mean(np.abs(errors))
                    )
                    if errors
                    else None,
                    "act_seconds": sum(r["seconds"][mode] for r in roots),
                    "proposed": entry["proposed"],
                    "changed": entry["changed"],
                }
            )
            for r in roots:
                if r["accepted"][mode] != r["base"]:
                    accepted.append(
                        {
                            "case_index": r["case_index"],
                            "parent": parent,
                            "mode": mode,
                            "action": r["accepted"][mode],
                            "conditional_reference": r["references"][
                                str(r["accepted"][mode])
                            ],
                        }
                    )
    full_calls = sum(
        r["history_validation"]["full_replay_equivalent_model_calls"]
        for r in independent["summaries"]
    )
    pruned_calls = sum(
        report["totals"][f"parent{p}_history"]["history_work"].get("model_calls", 0)
        for p in (1, 2)
    )
    positive = [
        {"case_index": r["case_index"], "alternative": int(a), **values}
        for r in independent["summaries"]
        for a, values in r["references"].items()
        if values["lower"] > 0
    ]
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
        "source_work": report["source_work"],
        "actual_pilot_work": report["actual_pilot_work"],
        "root_count": report["root_count"],
        "metrics": metrics,
        "accepted_action_references": accepted,
        "positive_conditional_reference_lower_margins": positive,
        "actual_pilot_summary": independent["actual_pilot_summary"],
        "actual_pruned_model_calls": pruned_calls,
        "full_replay_equivalent_model_calls": full_calls,
        "retained_call_fraction": pruned_calls / full_calls if full_calls else None,
        "audit_native_history_decisions": sum(
            r["history_validation"]["native_validation_decisions"]
            for r in independent["summaries"]
        ),
        "search_totals": report["totals"],
        "scope": "Complete small prospective pilot on16 unique deals. References are noisy and model-conditional; uniform reference errors include a different target distribution. No population-strength or promotion claim.",
    }
    write(output, result)
    lines = [
        "# Fixed64 terminal pilot: complete audited results",
        "",
        ("Sixteen unique deals; eight duplicate-deal pairs per parent/opponent cell. "
        "Intervals below describe this small pilot and do not establish replicated population strength."),
        "",
        "| Parent | Opponent | Contrast | Win change (pp) | 95% interval (pp) | Point change | 95% interval |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for cell in result["actual_pilot_summary"]:
        for row in cell["contrasts"]:
            win, points = row["win"], row["points"]
            lo, hi = [100 * x for x in win["interval95"]]
            plo, phi = points["interval95"]
            lines.append(
                f"| {cell['parent']} | {cell['opponent']} | {row['contrast']} | "
                f"{100 * win['mean']:.6f} | [{lo:.6f}, {hi:.6f}] | "
                f"{points['mean']:.6f} | [{plo:.6f}, {phi:.6f}] |"
            )
    lines += [
        "",
        ("All proposal errors below use the conditional terminal reference. "
        "For uniform search this includes omission of historical evidence, not only Monte Carlo error."),
        "",
        "| Parent | Mode | Roots | Proposed | Changed | Conditional-reference RMSE | Act seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics:
        error = m["all_action_conditional_reference_rmse"]
        formatted = "n/a" if error is None else f"{error:.9f}"
        lines.append(
            f"| {m['parent']} | {m['mode']} | {m['roots']} | {m['proposed']} | "
            f"{m['changed']} | {formatted} | {m['act_seconds']:.6f} |"
        )
    lines += [
        "",
        ("Accepted actions and all raw references remain in summary.json and rows.jsonl. "
        "Source play, actual candidate play, future branches and historical replay are counted separately."),
        "",
    ]
    (folder / "summary.md").write_bytes("\n".join(lines).encode("utf-8"))
    print(
        json.dumps(
            {"summary_sha256": sha(output), "metrics": metrics, "accepted": accepted}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
