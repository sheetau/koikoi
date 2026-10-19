"""Present the complete, independently audited duration study without selection."""

import argparse
import json
import math
import re
from pathlib import Path

from scripts.export_koi_temperature_results import interval, read, require, sha, table

PROTOCOL_SHA = "76e1b6af987e21f4e05e270aaabcd85ee9a89e68621589d1802a3a7543b24f18"
TRAINING = Path("runs/published_duration_study/training_complete.json")
TRAINING_SHA = "ce033a7a2b2c870efa0269509981c5a7913bfef2caca83c388fd8314ad6197ca"
RECOVERY_SHA = "bf1163339fdb52303344f118c92bb5c75204eea7e0211cdc295ab3ad9fca86c8"
RECOVERY_CAPABILITY = Path("runs/published_duration_recovery_capability/receipt.json")
POLICIES = ("parent", "short", "long")
CONTRASTS = ("long-short", "long-parent", "short-parent")
SCOPES = ("mixture", "wp", "point")
PRIMARY = {f"parent{p}/mixture/{c}/win" for p in (1, 2) for c in CONTRASTS[:2]}
LABELS = {
    f"parent{p}/{s}/{c}/{m}"
    for p in (1, 2)
    for s in SCOPES
    for c in (*POLICIES, *CONTRASTS)
    for m in ("win", "points")
}
CELLS = {f"parent{p}_{c}_{o}" for p in (1, 2) for c in POLICIES for o in SCOPES[1:]}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_analysis(summary):
    require(summary["mode"] == "main", "Capability is not main evidence")
    for key, expected in {
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 98304,
        "bootstrap_seed": 99410001,
        "bootstrap_replicates": 10000,
    }.items():
        require(summary[key] == expected, f"Unexpected budget/design: {key}")
    require(set(summary["rows"]) == LABELS, "Require all 72 secondary coordinates")
    for label, row in summary["rows"].items():
        bounds = row["marginal95"]
        require(
            len(bounds) == 2 and all(finite(x) for x in [row["mean"], *bounds]),
            "Nonfinite result",
        )
        require(bounds[0] <= bounds[1], "Reversed interval")
        if label.endswith("/win"):
            lower = -1 if "-" in label else 0
            require(all(lower <= x <= 1 for x in [row["mean"], *bounds]), "Win units")
    primary = summary["primary"]
    require(set(primary["rows"]) == PRIMARY, "Require all four primary coordinates")
    for key, expected in {
        "deals": 4096,
        "replicates": 10000,
        "seed": 99410001,
        "family_size": 4,
    }.items():
        require(primary[key] == expected, f"Unexpected primary design: {key}")
    require(
        finite(primary["critical"]) and primary["critical"] >= 0, "Invalid critical"
    )
    active = 0
    for label, row in primary["rows"].items():
        require(row["mean"] == summary["rows"][label]["mean"], "Primary mean differs")
        error = row["standard_error"]
        require(finite(error) and error >= 0, "Invalid standard error")
        bounds = row["simultaneous95"]
        if error == 0:
            require(
                bounds == [None, None] and row["sign"] == 0, "Degenerate sign claim"
            )
        else:
            active += 1
            expected = [
                row["mean"] - primary["critical"] * error,
                row["mean"] + primary["critical"] * error,
            ]
            require(
                len(bounds) == 2
                and all(finite(x) for x in bounds)
                and all(
                    math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                    for a, b in zip(bounds, expected)
                ),
                "Primary band differs",
            )
            sign = 1 if bounds[0] > 0 else -1 if bounds[1] < 0 else 0
            require(row["sign"] == sign, "Primary sign differs")
    require(primary["nondegenerate"] == active, "Active family count differs")
    qualified = all(row["sign"] == 1 for row in primary["rows"].values())
    require(
        type(summary["qualified"]) is bool and summary["qualified"] == qualified,
        "Qualification differs from four unrounded simultaneous bands",
    )


def recovery_lineage(root):
    """Verify preserved data directly; never reinterpret the failed old queue."""
    final_path = root / "recovery_complete.json"
    require(final_path.exists(), "Wait for recovery completion and independent audit")
    require(not (root / "recovery_failed.json").exists(), "Recovery failed")
    finish = read(final_path)
    plan_path = root / "recovery_protocol.json"
    require(
        finish["status"] == "complete"
        and sha(plan_path) == finish["recovery_protocol_sha256"] == RECOVERY_SHA,
        "Recovery identity differs",
    )
    plan = read(plan_path)
    source, old_queue = Path(plan["source"]), Path(plan["failed_queue"])
    require(plan["protocol_sha256"] == PROTOCOL_SHA, "Recovery protocol differs")
    require(
        read(source / "evaluation_process.json")["status"] == "failed"
        and read(old_queue / "process.json")["status"] == "failed",
        "Preserve original failed receipts",
    )
    copied = {c for c in CELLS if c.startswith(("parent1_", "parent2_parent_"))}
    require(
        len(plan["copied"]) == 8
        and set(plan["copied"]) == copied
        and len(plan["rerun_whole_cells"]) == 4
        and set(plan["rerun_whole_cells"]) == CELLS - copied,
        "Recovery cell partition differs",
    )
    partial = plan["preserved_partial"]
    require(
        set(partial) == {"parent2_short_wp", "parent2_short_point"}
        and len(finish["prefixes_exact"]) == 2
        and set(finish["prefixes_exact"]) == set(partial),
        "Require both preserved prefixes",
    )
    inputs = dict(plan["inputs"])
    required = [
        source / "evaluation_protocol.json",
        source / "evaluation_process.json",
        old_queue / "process.json",
        RECOVERY_CAPABILITY,
    ]
    for key in copied:
        for suffix in (".json", ".jsonl"):
            old = source / "evaluation" / (key + suffix)
            require(
                sha(old) == sha(root / "evaluation" / (key + suffix)),
                "Copied cell differs",
            )
            required.append(old)
    duplicate = {name: 0 for name in ("decisions", "round_endings", "full_matches")}
    for key, info in partial.items():
        old = source / "evaluation" / f"{key}.jsonl"
        raw = old.read_bytes()
        end = raw.rfind(b"\n") + 1
        prefix = raw[:end]
        rows = [json.loads(line) for line in prefix.splitlines()]
        require(
            end == info["prefix_bytes"]
            and len(raw) - end == info["trailing_bytes"]
            and len(rows) == info["complete_rows"],
            "Saved prefix extent differs",
        )
        with (root / "evaluation" / f"{key}.jsonl").open("rb") as stream:
            require(stream.read(end) == prefix, "Repeated inference prefix differs")
        for name in duplicate:
            total = sum(row["work"][name] for row in rows)
            require(total == info["work"][name], "Saved prefix work differs")
            duplicate[name] += total
        required.append(old)
    require(
        duplicate == finish["additional_recorded_duplicate_work"]
        and finish["unrecorded_interrupted_work"] == "unknown",
        "Recovery work differs",
    )
    require(
        sha(source / "evaluation_protocol.json") == PROTOCOL_SHA,
        "Original protocol changed",
    )
    for path in required:
        require(inputs.get(str(path)) == sha(path), f"Missing recovery binding: {path}")
    capability = read(RECOVERY_CAPABILITY)
    require(capability["status"] == "passed", "Recovery capability failed")
    for name, digest in capability["inputs"].items():
        require(
            name not in inputs or inputs[name] == digest, "Conflicting recovery input"
        )
        inputs[name] = digest
    for name, digest in inputs.items():
        require(sha(name) == digest, f"Changed recovery dependency: {name}")
    for path in (final_path, plan_path, root / "recovery_started.json"):
        inputs[str(path)] = sha(path)
    require(
        read(root / "recovery_started.json")["protocol_sha256"] == RECOVERY_SHA,
        "Recovery launch differs",
    )
    info = {
        "copied_cells": sorted(copied),
        "rerun_whole_cells": plan["rerun_whole_cells"],
        "additional_recorded_duplicate_work": duplicate,
        "unrecorded_interrupted_work": "unknown",
        "separate_recovery_capability_work": capability["new_execution_work"],
    }
    return finish, inputs, info


def load_completed(root, queue, audit_sha, *, recovered=False):
    require(bool(re.fullmatch(r"[0-9a-f]{64}", audit_sha)), "Pin the audit SHA256")
    recovery_inputs = {}
    recovery_info = None
    if recovered:
        finish, recovery_inputs, recovery_info = recovery_lineage(root)
        finish_path = root / "recovery_complete.json"
    else:
        finish_path = queue / "process.json"
        finish = read(finish_path)
        require(
            finish["status"] == "complete"
            and finish["completed"] == ["prepare", "evaluate", "audit"],
            "Wait for the completion queue",
        )
    process = read(root / "evaluation_process.json")
    require(
        process["status"] == "complete_awaiting_audit"
        and len(process["completed"]) == 12
        and set(process["completed"]) == CELLS,
        "Require all twelve completed cells",
    )
    protocol_path = root / "evaluation_protocol.json"
    require(sha(protocol_path) == PROTOCOL_SHA, "Study protocol changed")
    audit_path, summary_path = root / "audit.json", root / "summary.json"
    require(
        sha(audit_path) == audit_sha == finish["audit_sha256"], "Audit identity differs"
    )
    audit, summary = read(audit_path), read(summary_path)
    require(audit["status"] == "passed", "Audit must pass")
    require(
        audit["protocol_sha256"] == summary["protocol_sha256"] == PROTOCOL_SHA,
        "Protocol identity differs",
    )
    validate_analysis(summary)
    require(
        finish["deployment_changed"] is False
        and finish["qualified"] == audit["qualified"] == summary["qualified"],
        "Queue conclusion differs",
    )
    require(audit["work"]["full_matches"] == 98304, "Incomplete replay work")
    inputs = dict(audit["inputs"])
    required = [protocol_path, summary_path, TRAINING]
    for cell in sorted(CELLS):
        required.extend(
            root / "evaluation" / (cell + suffix) for suffix in (".json", ".jsonl")
        )
    for path in required:
        require(inputs.get(str(path)) == sha(path), f"Missing audit binding: {path}")
    require(sha(TRAINING) == TRAINING_SHA, "Training identity differs")
    for name, digest in inputs.items():
        require(sha(name) == digest, f"Changed dependency: {name}")
    for name, digest in recovery_inputs.items():
        require(
            name not in inputs or inputs[name] == digest,
            "Conflicting audit/recovery binding",
        )
        inputs[name] = digest
    for path in (
        audit_path,
        finish_path,
        root / "evaluation_process.json",
        Path(__file__),
        Path("scripts/export_koi_temperature_results.py"),
    ):
        inputs[str(path)] = sha(path)
    training = read(TRAINING)
    require(training["status"] == "complete", "Training incomplete")
    work = {
        key: sum(case["counters"][key] for case in training["cases"].values())
        for key in ("updates", "decisions", "rounds", "matches")
    }
    return summary, audit["work"], work, inputs, recovery_info


def render(summary, replay_work, training_work):
    validate_analysis(summary)
    decision = "passes" if summary["qualified"] else "fails"
    primary = []
    for label in sorted(PRIMARY):
        row = summary["primary"]["rows"][label]
        result = (
            interval(row["mean"], row["simultaneous95"], 100, True)
            if row["standard_error"]
            else f"{100 * row['mean']:+.4f} [undefined]"
        )
        primary.append([label, result, "yes" if row["sign"] == 1 else "no"])
    lines = [
        "Duration comparison: complete development results",
        "=================================================",
        "",
        f"The prespecified four-contrast improvement criterion {decision}.",
        "All four unrounded simultaneous lower bounds must exceed zero.",
        "A passing development gate needs separate prospective confirmation.",
        "",
        "Group parent1: fresh seed 99110001, retained comparator A.",
        "Group parent2: fresh seed 99210001, retained comparator B.",
        "Short and long start from original WP and share one training trajectory.",
        "Short: 500 policy updates; long: 4,000. Each includes 20 warmup updates.",
        "This is unequal-budget scaling, not a new algorithm or equal-compute gain.",
        "",
        "98,304 matches use 8,192 shared initial deals and both seats.",
        "Draws count as half wins; mixture weights assisted WP and Point equally.",
        "4,096 compound blocks pair WP/Point deals and retain all policies/seats.",
        "10,000 bootstrap replicates use seed 99410001. Primary intervals form one",
        "simultaneous 95% family; secondary intervals are descriptive marginal 95%.",
        "They measure deal variation, not variation across new training seeds.",
        "",
        "Primary comparisons",
        "-------------------",
        "",
        table(
            ["Comparison", "Win change (pp), simultaneous 95%", "Lower > 0"], primary
        ),
    ]
    for title, choices, contrast in (
        ("All absolute results", POLICIES, False),
        ("All paired comparisons", CONTRASTS, True),
    ):
        rows = []
        for parent in (1, 2):
            for scope in SCOPES:
                for choice in choices:
                    prefix = f"parent{parent}/{scope}/{choice}"
                    values = [
                        summary["rows"][prefix + "/" + m] for m in ("win", "points")
                    ]
                    rows.append(
                        [
                            parent,
                            scope,
                            choice,
                            interval(
                                values[0]["mean"],
                                values[0]["marginal95"],
                                100,
                                contrast,
                            ),
                            interval(
                                values[1]["mean"], values[1]["marginal95"], signed=True
                            ),
                        ]
                    )
        lines.extend(
            [
                "",
                title,
                "-" * len(title),
                "",
                table(
                    [
                        "Group",
                        "Scope",
                        "Policy/comparison",
                        "Win change (pp)" if contrast else "Win score (%)",
                        "Point change" if contrast else "Point margin",
                    ],
                    rows,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Work and interpretation",
            "-----------------------",
            "",
            "Training totals (short prefixes included once): "
            + json.dumps(training_work, sort_keys=True),
            "Actual evaluation and separately replayed work: "
            + json.dumps(replay_work, sort_keys=True),
            "Point margins are candidate minus opponent; divide by two for the",
            "original paper's final-minus-initial point convention.",
            "The audit replays recorded actions and checks statistics; it does not",
            "rerun training or neural inference. This export adds no matches.",
            "JSON preserves unrounded values in fraction units for win scores.",
            "No endpoint is selected or promoted by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def export(root, queue, folder, audit_sha, *, recovered=False):
    require(not folder.exists(), "Preserve existing exports; use a fresh directory")
    summary, replay_work, training_work, inputs, recovery = load_completed(
        root, queue, audit_sha, recovered=recovered
    )
    content = render(summary, replay_work, training_work)
    if recovered:
        content += "\nAdministrative recovery\n-----------------------\n\n"
        content += (
            "The original evaluation stopped on a Windows progress-file replacement error.\n"
            "Eight completed cells were copied unchanged; four incomplete cells were\n"
            "rerun on the identical full schedule. All preserved complete prefix rows\n"
            "match repeated neural inference exactly. The original failed receipts remain.\n"
            "The 98,304 analysis matches and replay work above exclude the additional\n"
            "recorded duplicate execution below; these repeats add no independent deals.\n"
            "Additional recorded duplicate work: "
            + json.dumps(recovery["additional_recorded_duplicate_work"], sort_keys=True)
            + "\nUnrecorded interrupted work: unknown.\n"
            "Separate recovery capability execution and native replay: "
            + json.dumps(recovery["separate_recovery_capability_work"], sort_keys=True)
            + "\nCapability matches repeat existing fixtures and are not strength evidence.\n"
        )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "results.rst").write_bytes(content.encode("utf-8"))
    evidence = {
        "status": "complete",
        "summary": summary,
        "inputs": inputs,
        "training_work": training_work,
        "evaluation_and_replay_work": replay_work,
        "outputs": {"results.rst": sha(folder / "results.rst")},
        "new_matches": 0,
        "new_neural_calls": 0,
        "scope": "Presentation of audited results, not an independent analysis.",
    }
    if recovered:
        evidence["recovery"] = recovery
    (folder / "evidence.json").write_bytes(
        (json.dumps(evidence, indent=2) + "\n").encode("utf-8")
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("runs/published_duration_evaluation")
    )
    parser.add_argument(
        "--queue", type=Path, default=Path("runs/published_duration_completion")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument(
        "--recovered",
        action="store_true",
        help="Verify the pinned recovery lineage instead of the original completion queue",
    )
    args = parser.parse_args()
    export(
        args.study, args.queue, args.output, args.audit_sha256, recovered=args.recovered
    )
