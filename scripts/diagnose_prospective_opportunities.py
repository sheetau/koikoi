"""Partition the closed prospective comparison without running another policy."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write


def diagnose(base, receipt):
    summary = read(base / "summary.json")
    exported = read(receipt)
    assert read(base / "queue.json")["status"] == summary["status"] == "complete"
    assert sha(base / "protocol.json") == summary["protocol_sha256"]
    assert read(base / "protocol.json")["phase_decisions"] == 2000000
    assert exported["status"] == "passed"
    assert exported["summary_sha256"] == sha(base / "summary.json")
    assert exported["bindings"] == summary["chunk_bindings"]
    categories, by_parent, unresolved, indices = Counter(), {}, [], []
    for name, digest in summary["chunk_bindings"].items():
        complete_path = Path(name)
        assert sha(complete_path) == digest
        record_path = complete_path.parent / "records.json"
        assert sha(record_path) == read(complete_path)["records_sha256"]
        for row in read(record_path):
            indices.append(row["index"])
            parent = str(row["source"]["parent"] + 1)
            by_parent.setdefault(parent, Counter())
            remaining = None
            if row["status"] != "evaluated":
                label = row["status"]
            else:
                entry = row["arms"]["controller"]
                selection = entry["selection"]
                remaining = 4000000 - sum(
                    phase["work"]["decisions"] for phase in entry["phases"].values()
                )
                assert 0 <= remaining <= 4000000
                if selection["reason"] == "strict_exact_gain":
                    label = "adopted"
                elif selection["reason"] == "parent_unvalued":
                    label = "parent_evaluation_incomplete"
                else:
                    assert selection["reason"] == "no_strict_gain"
                    if Fraction(selection["parent_q"]) == 1:
                        label = "parent_value_one"
                    else:
                        optimized = entry["phases"]["optimized"]
                        label = (
                            "completed_optimization_no_gain"
                            if optimized["refusal"] is None
                            else "incomplete_optimization_no_gain"
                        )
                if label in (
                    "parent_evaluation_incomplete",
                    "incomplete_optimization_no_gain",
                ):
                    unresolved.append(
                        {
                            "index": row["index"],
                            "parent": parent,
                            "category": label,
                            "unused_native_allowance": remaining,
                            "refusals": {
                                key: value["refusal"]
                                for key, value in entry["phases"].items()
                            },
                        }
                    )
            categories[label] += 1
            by_parent[parent][label] += 1
    assert sorted(indices) == list(range(summary["sources"]))
    assert len(set(indices)) == summary["sources"] == 512
    assert all(sum(counts.values()) == 256 for counts in by_parent.values())
    return {
        "status": "passed",
        "scope": "Descriptive partition of the closed cohort; no new strength test.",
        "bindings": {
            str(base / "protocol.json"): sha(base / "protocol.json"),
            str(base / "summary.json"): sha(base / "summary.json"),
            str(receipt): sha(receipt),
            **summary["chunk_bindings"],
        },
        "categories": dict(categories),
        "by_parent": {key: dict(value) for key, value in by_parent.items()},
        "unresolved": unresolved,
        "unresolved_roots_with_10k_unused": sum(
            r["unused_native_allowance"] >= 10000 for r in unresolved
        ),
        "new_source_matches": 0,
        "new_simulator_transitions": 0,
        "new_policy_calls": 0,
    }


def report(result):
    labels = {
        "missing_root": "No eligible root",
        "adopted": "Verified improvement adopted",
        "parent_value_one": "Parent conditional value already one",
        "completed_optimization_no_gain": "Optimization completed without a gain",
        "parent_evaluation_incomplete": "Parent evaluation incomplete",
        "incomplete_optimization_no_gain": "Optimization incomplete without a retained gain",
    }
    lines = [
        "Where the prospective controller leaves opportunities unresolved",
        "===============================================================",
        "",
        "This is a descriptive partition of the completed 512-source comparison.",
        "It adds no simulation, policy inference, source matches or strength test.",
        "",
        ".. list-table:: All-source controller outcomes",
        "   :header-rows: 1",
        "",
        "   * - Outcome",
        "     - Parent 1",
        "     - Parent 2",
        "     - Total",
    ]
    for key, count in result["categories"].items():
        lines.extend(
            [
                f"   * - {labels[key]}",
                f"     - {result['by_parent']['1'].get(key, 0)}",
                f"     - {result['by_parent']['2'].get(key, 0)}",
                f"     - {count}",
            ]
        )
    lines.extend(
        [
            "",
            f"There are {len(result['unresolved'])} eligible roots with an unresolved",
            "comparison and no adopted improvement. Of these,",
            f"{result['unresolved_roots_with_10k_unused']} retain at least 10,000 unused",
            "native transitions under the original 4M maximum allowance. That threshold",
            "describes remaining work only; it is not a new allocation or test rule.",
            "",
            "The fixed/optimized phases originally have separate 2M caps, with no",
            "transfer. Incomplete parent evaluation therefore leaves the optimized",
            "phase unused. Reassigning that allowance would be a different controller.",
            "No such reassignment is made in the original or ongoing certificate study.",
            "",
            "Completed no-gain searches and parent values of one should be separated",
            "from budget refusals when assessing the certificate's coverage benefit.",
            "The completed optimization is conditional on the declared finite belief;",
            "this diagnosis supplies neither a broader optimality nor a strength claim.",
            "The first-root gate also leaves missing roots outside this search.",
            "",
            "All input chunks and their record hashes match the completed export.",
            "Reproduce with python -m scripts.diagnose_prospective_opportunities.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/prospective_incumbent"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("runs/prospective_incumbent_export.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/PROSPECTIVE_OPPORTUNITIES.rst")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/prospective_opportunities.json")
    )
    args = parser.parse_args()
    if args.report.exists() or args.output.exists():
        raise ValueError("Choose new report and output paths")
    result = diagnose(args.base, args.receipt)
    args.report.write_text(report(result), encoding="utf-8", newline="\n")
    result["report_sha256"] = sha(args.report)
    result["script_sha256"] = sha(__file__)
    write(args.output, result)
    print(json.dumps(result["categories"], indent=2))
