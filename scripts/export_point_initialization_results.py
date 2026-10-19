"""Export every completed, audited Point-initialization result without selection."""

import argparse
import json
import math
import re
from pathlib import Path

from scripts.export_koi_temperature_results import interval, read, require, sha, table

TRAINING = Path("runs/point_initialization_study")
TRAINING_SHA = "defd0e993e2c85efb844164b0c1edfa4ac6bae917c31ea6a6eb33d7ba4618048"
QUEUE_SHA = "3419c9da7ec674828b8865ea61dddc6e253600586093c9a23c8d2d940864ac8a"
POLICIES = ("parent", "wp_fixed", "point_fixed", "point_matched")
CONTRASTS = (
    "point_matched-wp_fixed",
    "point_matched-point_fixed",
    "point_matched-parent",
    "point_fixed-wp_fixed",
    "point_fixed-parent",
    "wp_fixed-parent",
)
SCOPES = ("mixture", "wp", "point")
PRIMARY = {
    f"parent{p}/mixture/{c}/win"
    for p in (1, 2)
    for c in ("point_matched-wp_fixed", "point_matched-parent")
}
LABELS = {
    f"parent{p}/{s}/{c}/{m}"
    for p in (1, 2)
    for s in SCOPES
    for c in (*POLICIES, *CONTRASTS)
    for m in ("win", "points")
}
CELLS = {f"parent{p}_{c}_{o}" for p in (1, 2) for c in POLICIES for o in SCOPES[1:]}
ARMS = {f"seed{p}_{c}" for p in (1, 2) for c in POLICIES[1:]}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_analysis(summary):
    require(summary["mode"] == "main", "Capability is not main evidence")
    for key, expected in {
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 131072,
        "bootstrap_seed": 104110001,
        "bootstrap_replicates": 10000,
    }.items():
        require(summary[key] == expected, f"Unexpected budget/design: {key}")
    require(set(summary["rows"]) == LABELS, "Require all 120 secondary coordinates")
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
        "seed": 104110001,
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


def load_completed(root, queue, audit_sha):
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", audit_sha)), "Pin the completed audit SHA256"
    )
    require(
        (queue / "complete.json").exists(),
        "Wait for the completion queue and full audit",
    )
    for path in (
        queue / "failed.json",
        root / "evaluation_failed.json",
        TRAINING / "failed.json",
    ):
        require(not path.exists(), f"Failure receipt exists: {path}")
    finish = read(queue / "complete.json")
    require(
        finish["status"] == "complete"
        and finish["completed"] == ["prepare", "evaluate", "audit"],
        "Wait for all completion stages",
    )
    require(
        sha(queue / "protocol.json") == finish["protocol_sha256"] == QUEUE_SHA,
        "Queue protocol differs",
    )
    queue_plan = read(queue / "protocol.json")
    process = read(root / "evaluation_complete.json")
    require(
        process["status"] == "complete_awaiting_audit"
        and len(process["completed"]) == 16
        and set(process["completed"]) == CELLS,
        "Require all sixteen completed cells",
    )
    protocol_path = root / "evaluation_protocol.json"
    protocol_sha = sha(protocol_path)
    protocol = read(protocol_path)
    require(
        protocol["mode"] == "main"
        and protocol["pairs"] == 4096
        and set(protocol["cells"]) == CELLS,
        "Main evaluation design differs",
    )
    require(
        protocol["bootstrap"] == {"seed": 104110001, "replicates": 10000},
        "Bootstrap design differs",
    )
    require(
        protocol["main_training_protocol_sha256"] == TRAINING_SHA
        and process["protocol_sha256"] == protocol_sha,
        "Training/evaluation identity differs",
    )
    audit_path, summary_path = root / "audit.json", root / "summary.json"
    require(
        sha(audit_path) == audit_sha == finish["audit_sha256"], "Audit identity differs"
    )
    audit, summary = read(audit_path), read(summary_path)
    require(audit["status"] == "passed", "Audit must pass")
    require(
        audit["protocol_sha256"] == summary["protocol_sha256"] == protocol_sha,
        "Audited protocol differs",
    )
    validate_analysis(summary)
    require(
        finish["deployment_changed"] is False
        and finish["qualified"] == audit["qualified"] == summary["qualified"],
        "Queue conclusion differs",
    )
    inputs = dict(audit["inputs"])
    required = [
        protocol_path,
        summary_path,
        TRAINING / "protocol.json",
        TRAINING / "complete.json",
    ]
    for cell in sorted(CELLS):
        required.extend(
            root / "evaluation" / (cell + suffix) for suffix in (".json", ".jsonl")
        )
    for path in required:
        require(inputs.get(str(path)) == sha(path), f"Missing audit binding: {path}")
    for bindings in (protocol["inputs"], queue_plan["inputs"]):
        for name, digest in bindings.items():
            require(
                name not in inputs or inputs[name] == digest, "Conflicting dependency"
            )
            inputs[name] = digest
    for name, digest in inputs.items():
        require(sha(name) == digest, f"Changed dependency: {name}")
    require(
        sha(TRAINING / "protocol.json") == TRAINING_SHA, "Training protocol differs"
    )
    training = read(TRAINING / "complete.json")
    require(
        training["status"] == "complete"
        and training["protocol_sha256"] == TRAINING_SHA
        and set(training["arms"]) == ARMS,
        "Require all six training arms",
    )
    training_work = {
        key: sum(arm["counters"][key] for arm in training["arms"].values())
        for key in ("decisions", "rounds", "matches", "updates")
    }
    require(
        training_work == training["total_training_work"]
        and training_work["decisions"] == 51118080
        and training_work["updates"] == 3120,
        "Training work differs",
    )
    require(
        set(protocol["training_audits"])
        == {f"{a}/{u}" for a in ARMS for u in (20, 520)},
        "Require all twelve saved-state audits",
    )
    for arm, result in training["arms"].items():
        counts = result["counters"]
        require(
            result["status"] == "complete"
            and result["protocol_sha256"] == TRAINING_SHA,
            "Incomplete training arm",
        )
        require(
            counts["updates"] == 520
            and counts["decisions"] == 8519680
            and counts["warmup_updates"] == 20
            and counts["policy_updates"] == 500
            and counts["warmup_decisions"] == 327680,
            "Wrong arm budget",
        )
        for stage in (20, 520):
            check = protocol["training_audits"][f"{arm}/{stage}"]
            require(
                check["status"] == "passed"
                and check["protocol_sha256"] == TRAINING_SHA
                and check["arm"] == arm
                and check["update"] == stage,
                "Saved-state audit differs",
            )
        final = protocol["training_audits"][f"{arm}/520"]["checkpoint"]
        require(
            final["counters"] == counts
            and final["sha256"] == result["checkpoint_sha256"],
            "Training endpoint differs",
        )
    total = dict.fromkeys(("decisions", "round_endings", "full_matches"), 0)
    for key, cell in protocol["cells"].items():
        report = read(root / "evaluation" / f"{key}.json")
        require(
            report["status"] == "complete"
            and report["cell"] == key
            and all(report[k] == v for k, v in cell.items()),
            "Completed cell differs",
        )
        require(
            report["protocol_sha256"] == protocol_sha
            and report["rows_sha256"] == sha(root / "evaluation" / f"{key}.jsonl"),
            "Cell evidence differs",
        )
        require(
            report["checkpoint_sha256"] == sha(cell["checkpoint"]), "Cell model differs"
        )
        for name in total:
            require(
                type(report["work"][name]) is int and report["work"][name] >= 0,
                "Invalid work count",
            )
            total[name] += report["work"][name]
        require(report["work"]["full_matches"] == 8192, "Incomplete cell")
    require(
        total == audit["work"] and total["full_matches"] == 131072,
        "Replay work differs",
    )
    for path in (
        queue / "protocol.json",
        queue / "complete.json",
        root / "evaluation_complete.json",
        audit_path,
        Path(__file__),
        Path("scripts/export_koi_temperature_results.py"),
    ):
        inputs[str(path)] = sha(path)
    return summary, total, training_work, inputs


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
        "Point initialization: complete development results",
        "=================================================",
        "",
        f"The prespecified four-contrast improvement criterion {decision}.",
        "All four unrounded simultaneous lower bounds must exceed zero.",
        "A passing development gate needs separate prospective confirmation.",
        "",
        "Group parent1: fresh seed 102110001, retained comparator A.",
        "Group parent2: fresh seed 102210001, retained comparator B.",
        "Three arms per seed: original-WP fixed, original-Point fixed and Point calibrated.",
        "All have 20 critic-only plus 500 policy updates with identical decision budgets.",
        "Calibration matches phase-average entropy on the old calibration corpus only.",
        "It changes both sampling and policy-loss derivatives, not exploration alone.",
        "Initializations and fixed divisors do not constitute a new algorithm.",
        "",
        "131,072 matches use 8,192 shared initial deals and both seats.",
        "Draws count as half wins; mixture weights assisted WP and Point equally.",
        "4,096 compound blocks pair WP/Point deals and retain all policies/seats.",
        "10,000 bootstrap replicates use seed 104110001. Primary intervals form one",
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
            "Main training totals (warmup included once): "
            + json.dumps(training_work, sort_keys=True),
            "Actual evaluation and separately replayed work: "
            + json.dumps(replay_work, sort_keys=True),
            "Point margins are candidate minus opponent; divide by two for the",
            "original paper's final-minus-initial point convention.",
            "The audit replays recorded actions and checks statistics; it does not",
            "rerun training or neural inference. This export adds no matches.",
            "Calibration and capability work is separate from these main totals;",
            "the experiment ledger retains those costs and repeated verification work.",
            "JSON preserves unrounded values in fraction units for win scores.",
            "If qualified, the predeclared candidate is seed102110001 point_matched.",
            "Reserved101M prospective confirmation is still required for promotion.",
            "No endpoint is selected or promoted by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def export(root, queue, folder, audit_sha):
    require(not folder.exists(), "Preserve existing exports; use a fresh directory")
    summary, replay_work, training_work, inputs = load_completed(root, queue, audit_sha)
    content = render(summary, replay_work, training_work)
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
        "scope": "Presentation of completed audited development results, not a new analysis or model promotion.",
    }
    (folder / "evidence.json").write_bytes(
        (json.dumps(evidence, indent=2) + "\n").encode("utf-8")
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("runs/point_initialization_evaluation")
    )
    parser.add_argument(
        "--queue", type=Path, default=Path("runs/point_initialization_completion")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    export(args.study, args.queue, args.output, args.audit_sha256)
