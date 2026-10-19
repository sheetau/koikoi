"""Export every result from the completed, independently audited comparison."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

PROTOCOL_SHA = "b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19"
POLICIES = ("parent", "fixed", "calibrated")
SCOPES = ("mixture", "wp", "point")
COMPARISONS = (("calibrated", "fixed"), ("calibrated", "parent"), ("fixed", "parent"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_analysis(analysis):
    """Reject omissions, duplicates and a conclusion inconsistent with raw CIs."""
    require(
        (analysis["bootstrap_seed"], analysis["bootstrap_replicates"])
        == (97310001, 10000),
        "Unexpected bootstrap design",
    )
    expected_absolute = {
        (seed, policy, scope)
        for seed in (1, 2)
        for policy in POLICIES
        for scope in SCOPES
    }
    expected_contrasts = {
        (seed, left, right, scope)
        for seed in (1, 2)
        for left, right in COMPARISONS
        for scope in SCOPES
    }
    for group, keys, expected, means in (
        (
            "absolute",
            ("parent", "policy", "scope"),
            expected_absolute,
            ("win", "points"),
        ),
        (
            "contrasts",
            ("parent", "left", "right", "scope"),
            expected_contrasts,
            ("win_change", "point_change"),
        ),
    ):
        rows = analysis[group]
        require(len(rows) == 18, "Each result group must contain all 18 rows")
        require(
            {tuple(row[k] for k in keys) for row in rows} == expected,
            "Result identities are missing, duplicated or unexpected",
        )
        lower = 0 if group == "absolute" else -1
        for row in rows:
            numbers = [row[k] for k in means]
            for key in ("win_interval", "point_interval"):
                bounds = row[key]
                require(len(bounds) == 2 and bounds[0] <= bounds[1], "Invalid interval")
                numbers.extend(bounds)
            require(
                all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(x)
                    for x in numbers
                ),
                "Nonfinite result",
            )
            require(
                all(lower <= x <= 1 for x in [row[means[0]], *row["win_interval"]]),
                "Win values are outside their declared units",
            )
    primary = [
        row
        for row in analysis["contrasts"]
        if row["scope"] == "mixture" and row["left"] == "calibrated"
    ]
    qualified = all(row["win_interval"][0] > 0 for row in primary)
    require(
        type(analysis["replicated_strength_gain"]) is bool
        and analysis["replicated_strength_gain"] == qualified,
        "Primary conclusion differs from the four unrounded intervals",
    )
    return primary


def load_completed(root, expected_audit_sha):
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha)), "Pin the audit SHA256"
    )
    require(
        read(root / "finish_process.json")["status"] == "complete",
        "Wait for the finalizer to complete",
    )
    require(
        read(root / "process.json")["status"] == "complete_awaiting_audit",
        "The match producer is not complete",
    )
    require(sha(root / "protocol.json") == PROTOCOL_SHA, "Study protocol changed")
    audit_path = root / "independent_audit.json"
    require(
        sha(audit_path) == expected_audit_sha, "Independent audit identity mismatch"
    )
    audit, summary = read(audit_path), read(root / "summary.json")
    require(
        audit["status"] == "passed" and summary["status"] == "complete",
        "Only completed and passed results can be exported",
    )
    for key in (
        "analysis",
        "training_counters",
        "evaluation_work",
        "evaluation_replay_work",
        "shared_evaluation_deals",
    ):
        require(audit[key] == summary[key], f"Summary disagrees with audit: {key}")
    require(summary["deployment_changed"] is False, "Unexpected deployment change")
    require(
        summary["evaluation_work"]["full_matches"] == 49152
        and summary["shared_evaluation_deals"] == 4096,
        "Incomplete evaluation budget",
    )
    require(
        set(summary["training_counters"])
        == {
            f"seed{seed}_{mode}" for seed in (1, 2) for mode in ("fixed", "calibrated")
        },
        "Missing or unexpected training arms",
    )
    require(
        summary["training_work"]
        == {
            key: sum(row[key] for row in summary["training_counters"].values())
            for key in ("updates", "decisions", "rounds", "matches")
        },
        "Training totals disagree with per-arm counters",
    )
    require(
        summary["training_work"]["updates"] == 2080
        and summary["training_work"]["decisions"] == 34078720,
        "Incomplete training budget",
    )
    validate_analysis(summary["analysis"])
    inputs = dict(summary["inputs"])
    require(
        all(inputs.get(name) == digest for name, digest in audit["inputs"].items()),
        "Summary omits or changes audit dependencies",
    )
    require(
        any(
            Path(name) == audit_path and digest == expected_audit_sha
            for name, digest in inputs.items()
        ),
        "Summary does not bind the audit",
    )
    for name, digest in inputs.items():
        require(sha(name) == digest, f"Changed dependency: {name}")
    for path in (
        root / "summary.json",
        root / "finish_process.json",
        root / "process.json",
        root / "protocol.json",
        Path(__file__),
    ):
        inputs[str(path)] = sha(path)
    return summary, inputs


def csv_text(rows, contrast):
    columns = ["parent", *(["left", "right"] if contrast else ["policy"]), "scope"]
    means = ("win_change", "point_change") if contrast else ("win", "points")
    columns += [*means, "win_lower", "win_upper", "point_lower", "point_upper"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        result = {key: row[key] for key in columns if key in row}
        for metric in ("win", "point"):
            result[metric + "_lower"], result[metric + "_upper"] = row[
                metric + "_interval"
            ]
        writer.writerow(result)
    return stream.getvalue()


def table(headers, rows):
    widths = [
        max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))
    ]
    border = "  ".join("=" * width for width in widths)

    def line(row):
        return "  ".join(
            str(value).ljust(widths[i]) for i, value in enumerate(row)
        ).rstrip()

    return "\n".join([border, line(headers), border, *map(line, rows), border])


def interval(mean, bounds, scale=1, signed=False):
    pattern = "+.4f" if signed else ".4f"
    return f"{mean * scale:{pattern}} [{bounds[0] * scale:{pattern}}, {bounds[1] * scale:{pattern}}]"


def render(summary):
    analysis = summary["analysis"]
    primary = validate_analysis(analysis)
    decision = "passes" if analysis["replicated_strength_gain"] else "fails"
    lines = [
        "Koi-temperature comparison: complete development results",
        "========================================================",
        "",
        f"The prespecified replicated improvement criterion {decision}.",
        "It requires positive lower win-change bounds against both the matched",
        "fixed-temperature control and the retained parent at both training seeds.",
        "The gate uses unrounded values; displayed values are rounded to four decimals.",
        "",
        "All six policies share 4,096 development deal streams: 2,048 per opponent,",
        "with both seats played. The study contains 49,152 full matches. Draws count",
        "as half wins. The mixture gives equal weight to assisted WP and Point.",
        "Parent 1/2 means retained A/B; new learner seeds are 96110001/96210001.",
        "Both new conditions start from original WP, not the retained parents.",
        "",
        "Primary comparisons",
        "-------------------",
        "",
        table(
            ["Seed", "Comparison", "Win change (pp), 95% CI", "Lower > 0"],
            [
                [
                    row["parent"],
                    row["left"] + " - " + row["right"],
                    interval(row["win_change"], row["win_interval"], 100, True),
                    "yes" if row["win_interval"][0] > 0 else "no",
                ]
                for row in primary
            ],
        ),
        "",
        "All paired comparisons",
        "----------------------",
        "",
        table(
            [
                "Seed",
                "Scope",
                "Comparison",
                "Win change (pp), 95% CI",
                "Point change, 95% CI",
            ],
            [
                [
                    row["parent"],
                    row["scope"],
                    row["left"] + " - " + row["right"],
                    interval(row["win_change"], row["win_interval"], 100, True),
                    interval(row["point_change"], row["point_interval"], signed=True),
                ]
                for row in analysis["contrasts"]
            ],
        ),
        "",
        "All absolute results",
        "--------------------",
        "",
        table(
            [
                "Seed",
                "Scope",
                "Policy",
                "Win score (%), 95% CI",
                "Point margin, 95% CI",
            ],
            [
                [
                    row["parent"],
                    row["scope"],
                    row["policy"],
                    interval(row["win"], row["win_interval"], 100),
                    interval(row["points"], row["point_interval"], signed=True),
                ]
                for row in analysis["absolute"]
            ],
        ),
        "",
        "Interpretation and work",
        "-----------------------",
        "",
        "Intervals use 10,000 paired bootstrap replicates (seed 97310001). Each",
        "block retains both seats and both fixed opponent strata. These are marginal",
        "95% intervals, not simultaneous intervals for all displayed comparisons.",
        "They measure deal variation, not the variation of new training seeds.",
        "Point margins are candidate minus opponent; divide by two for the original",
        "paper's final-minus-initial point convention. CSV files retain unrounded",
        "values and use fractions for wins and win changes, not percent units.",
        "",
        "Training work (warmup included): "
        + json.dumps(summary["training_work"], sort_keys=True),
        "Actual evaluation work: "
        + json.dumps(summary["evaluation_work"], sort_keys=True),
        "Recorded-action replay work: "
        + json.dumps(summary["evaluation_replay_work"], sort_keys=True),
        "",
        "The independent audit checks saved training states, recorded trajectories",
        "and paired statistics. It does not rerun training or neural action selection.",
        "This export adds no neural calls, simulator steps, matches or training.",
        "A positive development result needs prospective confirmation before promotion.",
        "No model, checkpoint or temperature is selected by this report.",
        "",
    ]
    return "\n".join(lines)


def export(root, folder, expected_audit_sha):
    require(not folder.exists(), "Preserve existing exports; use a fresh directory")
    summary, inputs = load_completed(root, expected_audit_sha)
    outputs = {"results.rst": render(summary)}
    for group in ("absolute", "contrasts"):
        outputs[group + ".csv"] = csv_text(
            summary["analysis"][group], group == "contrasts"
        )
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        (folder / name).write_bytes(content.encode("utf-8"))
    evidence = {
        "status": "complete",
        "inputs": inputs,
        "outputs": {str(folder / name): sha(folder / name) for name in outputs},
        "absolute_rows": 18,
        "contrast_rows": 18,
        "primary_rows": 4,
        "replicated_strength_gain": summary["analysis"]["replicated_strength_gain"],
        "new_neural_calls": 0,
        "new_simulator_steps": 0,
        "scope": "Presentation of audited results, not an additional independent analysis.",
    }
    (folder / "evidence.json").write_bytes(
        (json.dumps(evidence, indent=2) + "\n").encode("utf-8")
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=Path("runs/koi_temperature_study")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    export(args.study, args.output, args.audit_sha256)
