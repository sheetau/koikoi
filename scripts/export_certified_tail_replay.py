"""Export descriptive certificate comparisons after the complete-cohort audit."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from koikoi.layout import POINTS
from scripts.certified_tail_replay import check, source_records
from scripts.export_belief_tail import score
from scripts.export_full_source_results import table
from scripts.run_arbitration_study import read, sha, write


def aggregate(rows):
    """Keep every source in its parent's denominator, including absent roots."""
    counts = Counter()
    cells = {
        (parent, arm): {"gain": Fraction(0), "actual": Fraction(0), "adoptions": 0}
        for parent in (0, 1)
        for arm in ("controller", "root_only")
    }
    for row in rows:
        source, status = row["source"], row["status"]
        parent, seat = source["parent"], source["seat"]
        counts[parent] += 1
        base_score = score(
            source["final"][POINTS + seat] - source["final"][POINTS + 1 - seat]
        )
        if status != "evaluated":
            assert status in ("missing_root", "support_refused")
            continue
        for arm in ("controller", "root_only"):
            selected = row["arms"][arm]["selection"]
            cell = cells[parent, arm]
            cell["gain"] += Fraction(selected["gain"])
            cell["adoptions"] += selected["action"] is not None
            cell["actual"] += (
                score(row["assessment"]["outcomes"][arm]["margin"]) - base_score
            )
    assert counts == {0: 256, 1: 256}
    for (parent, _), cell in cells.items():
        for key in ("gain", "actual"):
            cell[key] /= counts[parent]
    return cells


def compare_coverage(original, certified):
    result = {arm: Counter() for arm in ("controller", "root_only")}
    for old, new in zip(original, certified, strict=True):
        assert (old["index"], old["source"], old["status"]) == (
            new["index"],
            new["source"],
            new["status"],
        )
        if old["status"] != "evaluated":
            continue
        for arm, counts in result.items():
            a = old["arms"][arm]["selection"]
            b = new["arms"][arm]["selection"]
            counts["eligible"] += 1
            counts["new_parent_values"] += (
                a["parent_q"] is None and b["parent_q"] is not None
            )
            counts["lost_parent_values"] += (
                a["parent_q"] is not None and b["parent_q"] is None
            )
            counts["new_adoptions"] += a["action"] is None and b["action"] is not None
            counts["lost_adoptions"] += a["action"] is not None and b["action"] is None
            delta = Fraction(b["gain"]) - Fraction(a["gain"])
            counts["greater_gain"] += delta > 0
            counts["smaller_gain"] += delta < 0
    return {arm: dict(counts) for arm, counts in result.items()}


def export(base, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh report and receipt paths")
    protocol, _ = check(base)
    assert read(base / "queue.json")["status"] == "complete"
    summary = read(base / "summary.json")
    assert summary["status"] == "passed"
    assert summary["protocol_sha256"] == sha(base / "protocol.json")
    assert summary["new_source_matches"] == 0 and summary["reused_sources"] == 512
    old_rows, new_rows, bindings = [], [], {}
    source = Path(protocol["source"])
    for chunk in protocol["chunks"]:
        folder = base / f"chunk_{chunk:03}"
        complete_path = folder / "complete.json"
        bindings[str(complete_path)] = sha(complete_path)
        assert (
            bindings[str(complete_path)]
            == summary["chunk_bindings"][str(complete_path)]
        )
        complete = read(complete_path)
        for name, expected in complete["artifacts"].items():
            assert sha(folder / name) == expected
        old_rows.extend(source_records(source / f"chunk_{chunk:03}"))
        new_rows.extend(source_records(folder))
    assert [r["index"] for r in new_rows] == list(range(512))
    assert bindings == summary["chunk_bindings"]
    old, new = aggregate(old_rows), aggregate(new_rows)
    coverage = compare_coverage(old_rows, new_rows)
    for name, cells in (("original", old), ("certified", new)):
        for field, key in (("gain", "conditional"), ("actual", "actual")):
            expected = [
                float(
                    cells[parent, "controller"][field]
                    - (
                        cells[parent, "root_only"][field]
                        if control == "root_only"
                        else 0
                    )
                )
                for parent in (0, 1)
                for control in ("parent", "root_only")
            ]
            assert all(
                abs(a - b) < 1e-12
                for a, b in zip(expected, summary[f"{name}_{key}_mean"], strict=True)
            )
    result_rows = [
        [
            version,
            arm,
            parent + 1,
            cells[parent, arm]["adoptions"],
            str(cells[parent, arm]["gain"]),
            f"{100 * float(cells[parent, arm]['gain']):+.4f}",
            str(cells[parent, arm]["actual"]),
            f"{100 * float(cells[parent, arm]['actual']):+.4f}",
        ]
        for version, cells in (("Original", old), ("Certificate", new))
        for arm in ("controller", "root_only")
        for parent in (0, 1)
    ]
    cost_rows = []
    for version, arms in summary["costs"].items():
        for arm, cost in arms.items():
            cost_rows.append(
                [
                    version,
                    arm,
                    cost.get("decisions", 0),
                    cost.get("completions", 0),
                    cost.get("represented_completions", 0),
                    cost.get("certified_opponent_rows", 0),
                    cost.get("certified_opponent_mass", 0),
                    cost.get("parent_valued", 0),
                    cost.get("capped_phases", 0),
                ]
            )
    original_gate = read(source / "summary.json")["development_strength_gate"]
    text = (
        "Final-opponent-turn certificate: complete-cohort comparison\n"
        "==========================================================\n\n"
        "Both implementations use the same saved512 sources/128 deal streams. "
        "This comparison adds zero source matches. All missing roots and support "
        "refusals remain in the256-source denominator per parent. Gains are relative "
        "to the unchanged parent. Actual scores count wins1, draws1/2, losses0.\n\n"
        + table(
            "Policy outcomes",
            [
                "Version",
                "Arm",
                "Parent",
                "Adoptions",
                "Exact conditional mean",
                "Conditional pp",
                "Exact actual mean",
                "Actual pp",
            ],
            result_rows,
        )
        + table(
            "Matched computational allowances",
            [
                "Version",
                "Arm",
                "Native decisions",
                "Physical completions",
                "Represented simulated completions",
                "Certified rows",
                "Certified mass",
                "Parent values",
                "Capped phases",
            ],
            cost_rows,
        )
        + table(
            "Coverage changes", ["Arm", "Counts"], [[a, c] for a, c in coverage.items()]
        )
        + "The controller retains separate2M-transition/500k-live-node phases; "
        "root-only uses4M transitions/1M nodes. Both start with empty caches. "
        "Completing a previously unavailable parent value can enable an additional "
        "phase, so reduced branch cost need not reduce total scheduled work. "
        "Symbolic resolutions are separate from simulated terminal completions.\n\n"
        + table(
            "Operational planning times",
            ["Version", "Arm", "Seconds"],
            [
                [v, a, f"{c.get('planning_seconds', 0):.3f}"]
                for v, arms in summary["costs"].items()
                for a, c in arms.items()
            ],
        )
        + "The original runs provide saved computational controls. These are "
        "separate concurrent runs, not an alternating isolated latency benchmark.\n\n"
        + table(
            "Additional verification and conditional execution",
            ["Category", "Counts"],
            [[name, counts] for name, counts in summary["work"].items()],
        )
        + f"The original prospective strength gate {'passed' if original_gate else 'failed'}. "
        "It is unchanged by this extension. The reused-cohort means above are "
        "descriptive; they carry no fresh confirmation claim or new confidence intervals.\n\n"
        "Completed common root values and complete learner maps matched. Every "
        "adoption's parent and selected values were independently replayed under "
        "Python rules, including omitted opponent moves. Complete-only audit checked "
        "all bindings, selections, common maps, actual traces and work totals. "
        "This export verifies bound artifacts and exact aggregate arithmetic; it "
        "adds no simulations or inference calls. The conditional values require "
        "the specified belief and opponent model and are separate from source-win changes.\n\n"
        f"Protocol SHA256: {sha(base / 'protocol.json')}.\n"
        f"Audit summary SHA256: {sha(base / 'summary.json')}.\n"
    )
    output.write_text(text, encoding="utf-8", newline="\n")
    write(
        receipt,
        {
            "status": "passed",
            "report_sha256": sha(output),
            "summary_sha256": sha(base / "summary.json"),
            "exporter_sha256": sha(__file__),
            "chunk_bindings": bindings,
            "coverage": coverage,
            "new_simulator_work": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/certified_tail_replay"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/CERTIFIED_TAIL_REPLAY_RESULTS.rst")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("runs/certified_tail_replay_export.json")
    )
    args = parser.parse_args()
    export(args.base, args.output, args.receipt)
