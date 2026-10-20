"""Present the complete audited auxiliary comparison, without selecting results."""

import argparse
import json
import math
import re
from pathlib import Path

from scripts.export_koi_temperature_results import interval, read, require, sha, table

TRAINING = Path("runs/score_auxiliary_study")
TRAINING_SHA = "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
QUEUE_SHA = "14a3eb5fed234ade9ce2bc5bd6d8e3b510f702f3cfbc32e4a880bca4486309cf"
POLICIES = ("parent", "none", "win", "score")
CONTRASTS = (
    "score-none",
    "score-win",
    "score-parent",
    "win-none",
    "win-parent",
    "none-parent",
)
SCOPES = ("mixture", "wp", "point")
PRIMARY = {f"seed{s}/mixture/{c}/win" for s in (1, 2) for c in CONTRASTS[:3]}
LABELS = {
    f"seed{s}/{o}/{c}/{m}"
    for s in (1, 2)
    for o in SCOPES
    for c in (*POLICIES, *CONTRASTS)
    for m in ("win", "points")
}
CELLS = {f"parent{s}_{p}_{o}" for s in (1, 2) for p in POLICIES for o in SCOPES[1:]}
ARMS = {f"seed{s}_{p}" for s in (1, 2) for p in POLICIES[1:]}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_analysis(summary):
    require(summary["mode"] == "main", "Capability is not main evidence")
    for key, expected in {
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 131072,
        "bootstrap_seed": 109310001,
        "bootstrap_replicates": 10000,
    }.items():
        require(summary[key] == expected, f"Unexpected design: {key}")
    require(set(summary["rows"]) == LABELS, "Require all 120 secondary coordinates")
    for label, row in summary["rows"].items():
        bounds = row["marginal95"]
        require(
            len(bounds) == 2 and all(finite(v) for v in [row["mean"], *bounds]),
            "Nonfinite result",
        )
        require(bounds[0] <= bounds[1], "Reversed interval")
        if label.endswith("/win"):
            lower = -1 if "-" in label else 0
            require(all(lower <= v <= 1 for v in [row["mean"], *bounds]), "Win units")
    primary = summary["primary"]
    require(set(primary["rows"]) == PRIMARY, "Require all six primary comparisons")
    for key, expected in {
        "deals": 4096,
        "replicates": 10000,
        "seed": 109310001,
        "family_size": 6,
    }.items():
        require(primary[key] == expected, f"Primary design differs: {key}")
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
                and all(finite(v) for v in bounds)
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
        "Qualification differs from six unrounded simultaneous bands",
    )


def load_completed(root, queue, audit_sha):
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", audit_sha)), "Pin the completed audit SHA256"
    )
    require(
        (queue / "complete.json").exists(), "Wait for the completion queue and audit"
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
        and finish["completed"]
        == ["training_audit", "prepare", "evaluate", "evaluation_audit"],
        "Require all four completion stages",
    )
    require(
        sha(queue / "protocol.json") == finish["protocol_sha256"] == QUEUE_SHA,
        "Queue protocol differs",
    )
    require(
        sha(TRAINING / "protocol.json") == TRAINING_SHA, "Training protocol differs"
    )
    protocol = read(root / "evaluation_protocol.json")
    protocol_sha = sha(root / "evaluation_protocol.json")
    require(
        protocol["mode"] == "main"
        and protocol["pairs"] == 4096
        and set(protocol["cells"]) == CELLS,
        "Main evaluation design differs",
    )
    require(
        protocol["main_training_protocol_sha256"] == TRAINING_SHA,
        "Training identity differs",
    )
    require(
        protocol["bootstrap"] == {"seed": 109310001, "replicates": 10000},
        "Bootstrap differs",
    )
    process = read(root / "evaluation_complete.json")
    require(
        process["status"] == "complete_awaiting_audit"
        and len(process["completed"]) == 16
        and set(process["completed"]) == CELLS
        and process["protocol_sha256"] == protocol_sha,
        "Require all sixteen completed cells",
    )
    require(
        sha(root / "audit.json") == audit_sha == finish["evaluation_audit_sha256"],
        "Audit identity differs",
    )
    audit, summary = read(root / "audit.json"), read(root / "summary.json")
    require(
        audit["status"] == "passed"
        and audit["protocol_sha256"] == summary["protocol_sha256"] == protocol_sha,
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
        root / "evaluation_protocol.json",
        root / "summary.json",
        TRAINING / "protocol.json",
        TRAINING / "complete.json",
        TRAINING / "training_audit.json",
    ]
    for cell in sorted(CELLS):
        required.extend(
            root / "evaluation" / (cell + suffix) for suffix in (".json", ".jsonl")
        )
    for path in required:
        require(inputs.get(str(path)) == sha(path), f"Missing audit binding: {path}")
    training, training_audit = (
        read(TRAINING / "complete.json"),
        read(TRAINING / "training_audit.json"),
    )
    require(
        training["status"] == "complete"
        and training["protocol_sha256"] == TRAINING_SHA
        and set(training["arms"]) == ARMS,
        "Require all six training arms",
    )
    require(
        training_audit["status"] == "passed"
        and training_audit["format"] == "score_auxiliary_training_audit_v1"
        and training_audit["protocol_sha256"] == TRAINING_SHA
        and set(training_audit["arms"]) == ARMS,
        "Training audit differs",
    )
    require(
        sha(TRAINING / "training_audit.json") == finish["training_audit_sha256"]
        and training_audit["complete_sha256"] == sha(TRAINING / "complete.json"),
        "Training audit identity differs",
    )
    work = {
        group: {
            key: sum(arm[group][key] for arm in training["arms"].values())
            for key in ("decisions", "rounds", "matches")
        }
        for group in ("ppo_work", "auxiliary_work")
    }
    require(
        work == training["totals"] == training_audit["totals"], "Training work differs"
    )
    require(
        work["ppo_work"]["decisions"] == 51118080
        and work["auxiliary_work"]["matches"] == 76800,
        "Training budget differs",
    )
    for arm, receipt in training["arms"].items():
        ppo, aux = receipt["ppo_work"], receipt["auxiliary_work"]
        require(
            receipt["status"] == "complete"
            and receipt["protocol_sha256"] == TRAINING_SHA,
            "Incomplete arm",
        )
        require(
            ppo["updates"] == 520
            and ppo["warmup_updates"] == 20
            and ppo["policy_updates"] == 500
            and ppo["decisions"] == 8519680
            and ppo["warmup_decisions"] == 327680,
            "Wrong PPO budget",
        )
        require(
            aux["cycles"] == 100 and aux["matches"] == 12800, "Wrong auxiliary budget"
        )
        endpoint = TRAINING / arm / "update_00000520"
        require(
            Path(receipt["endpoint"]) == endpoint
            and sha(endpoint / "manifest.json")
            == receipt["manifest_sha256"]
            == training_audit["arms"][arm]["endpoint_manifest_sha256"],
            "Training endpoint differs",
        )
    for bindings in (
        protocol["inputs"],
        read(queue / "protocol.json")["inputs"],
        training_audit["inputs"],
    ):
        for name, digest in bindings.items():
            require(
                name not in inputs or inputs[name] == digest, "Conflicting dependency"
            )
            inputs[name] = digest
    for name, digest in inputs.items():
        require(sha(name) == digest, f"Changed dependency: {name}")
    evaluation = dict.fromkeys(("decisions", "round_endings", "full_matches"), 0)
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
        for name in evaluation:
            require(
                type(report["work"][name]) is int and report["work"][name] >= 0,
                "Invalid work",
            )
            evaluation[name] += report["work"][name]
        require(report["work"]["full_matches"] == 8192, "Incomplete cell")
    require(
        evaluation == audit["work"] and evaluation["full_matches"] == 131072,
        "Evaluation work differs",
    )
    for path in (
        queue / "protocol.json",
        queue / "complete.json",
        root / "evaluation_complete.json",
        root / "audit.json",
        Path(__file__),
        Path("scripts/export_koi_temperature_results.py"),
    ):
        inputs[str(path)] = sha(path)
    return summary, evaluation, training, inputs


def render(summary, evaluation, training):
    validate_analysis(summary)
    decision = "passes" if summary["qualified"] else "fails"
    rows = []
    for label in sorted(PRIMARY):
        row = summary["primary"]["rows"][label]
        value = (
            interval(row["mean"], row["simultaneous95"], 100, True)
            if row["standard_error"]
            else f"{100 * row['mean']:+.4f} [undefined]"
        )
        rows.append([label, value, "yes" if row["sign"] == 1 else "no"])
    lines = [
        "Score-auxiliary learning: complete development results",
        "====================================================",
        "",
        f"The prespecified six-contrast improvement criterion {decision}.",
        "All six unrounded simultaneous lower bounds must exceed zero.",
        "",
        "Seed 1: 106110001; seed 2: 106210001. Each compares none, outcome-only (win)",
        "and full-margin (score) auxiliary learning from original Point initialization.",
        "The win/score arms share 65-logit heads; score adds within-outcome likelihood.",
        "All arms collect 128 complete matches every 5 policy updates; none does not optimize.",
        "Auxiliary rows and compute can differ because learned policies differ.",
        "Retained parents A/B are separate WP-initialized reference policies.",
        "",
        "131,072 matches use 8,192 shared initial deals and both seats. Draws count as",
        "half wins; the mixture weights strict assisted WP and Point equally.",
        "4,096 compound blocks retain all policies, learner groups, opponents and seats.",
        "10,000 bootstrap replicates use 109310001. Six primary comparisons form one",
        "simultaneous 95% family; all 120 secondary coordinates have descriptive marginal",
        "95% intervals. These quantify deal variation, not variation across new learner seeds.",
        "",
        "Primary comparisons",
        "-------------------",
        "",
        table(["Comparison", "Win change (pp), simultaneous 95%", "Lower > 0"], rows),
    ]
    for title, choices, contrast in (
        ("All absolute results", POLICIES, False),
        ("All paired comparisons", CONTRASTS, True),
    ):
        rows = []
        for seed in (1, 2):
            for scope in SCOPES:
                for choice in choices:
                    prefix = f"seed{seed}/{scope}/{choice}"
                    win, points = [
                        summary["rows"][prefix + "/" + m] for m in ("win", "points")
                    ]
                    rows.append(
                        [
                            seed,
                            scope,
                            choice,
                            interval(win["mean"], win["marginal95"], 100, contrast),
                            interval(points["mean"], points["marginal95"], signed=True),
                        ]
                    )
        lines += [
            "",
            title,
            "-" * len(title),
            "",
            table(
                [
                    "Seed",
                    "Scope",
                    "Policy/comparison",
                    "Win change (pp)" if contrast else "Win score (%)",
                    "Point change" if contrast else "Point margin",
                ],
                rows,
            ),
        ]
    lines += ["", "Actual work", "-----------", ""]
    rows = []
    for arm in sorted(training["arms"]):
        receipt = training["arms"][arm]
        for name in ("ppo_work", "auxiliary_work"):
            counts = receipt[name]
            rows.append(
                [
                    arm,
                    name,
                    counts["decisions"],
                    counts["rounds"],
                    counts["matches"],
                    counts.get("retained_rows", "-"),
                    counts.get("example_presentations", "-"),
                ]
            )
    lines += [
        table(
            [
                "Arm",
                "Stream",
                "Decisions",
                "Rounds",
                "Matches",
                "Retained rows",
                "Auxiliary presentations",
            ],
            rows,
        ),
        "",
        "Main training totals (warmup included once): "
        + json.dumps(training["totals"], sort_keys=True),
        "Actual evaluation and separately replayed work: "
        + json.dumps(evaluation, sort_keys=True),
        "",
        "Point margins are candidate minus opponent; divide by two for the original",
        "paper's final-minus-initial point convention. Capability and repeated-check",
        "work are recorded separately in the experiment ledger. Export adds no matches.",
        "JSON retains unrounded win fractions. The fixed mechanism report is separate.",
        "Passing identifies the prospectively named seed1_score for a separately",
        "prepared confirmation; it does not promote an endpoint. Reserved 101M is unused.",
        "No better-seed selection or universal-strength claim follows from this report.",
        "",
    ]
    return "\n".join(lines)


def export(root, queue, folder, audit_sha):
    require(not folder.exists(), "Preserve existing exports; use a fresh directory")
    summary, evaluation, training, inputs = load_completed(root, queue, audit_sha)
    content = render(summary, evaluation, training)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "results.rst").write_bytes(content.encode("utf-8"))
    evidence = {
        "status": "complete",
        "summary": summary,
        "training": training,
        "evaluation_and_replay_work": evaluation,
        "inputs": inputs,
        "outputs": {"results.rst": sha(folder / "results.rst")},
        "new_matches": 0,
        "new_neural_calls": 0,
        "scope": "Presentation of all completed audited development results, not new analysis or promotion.",
    }
    (folder / "evidence.json").write_bytes(
        (json.dumps(evidence, indent=2) + "\n").encode("utf-8")
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("runs/score_auxiliary_evaluation")
    )
    parser.add_argument(
        "--queue", type=Path, default=Path("runs/score_auxiliary_completion")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    export(args.study, args.queue, args.output, args.audit_sha256)
