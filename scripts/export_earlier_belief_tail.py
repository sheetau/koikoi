"""Export the fixed five-draw capability check, including all refusals and work."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import POINTS
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.check_belief_tail import add_step, empty_work
from scripts.check_earlier_belief_tail import cases
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions, score, sum_work
from scripts.export_full_source_results import table
from scripts.train_branch_policy_gradient import sha, write


def export(base, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh report and receipt paths")
    complete = read(base / "complete.json")
    protocol = read(base / "protocol.json")
    sources = read(base / "sources.json")
    plans = read(base / "plans_complete.json")
    assessments = read(base / "assessments.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    bindings = {}
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("sources_sha256", "sources.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[key] == sha(base / name)
        bindings[str(base / name)] = complete[key]
    assert plans["sources_sha256"] == complete["sources_sha256"]
    assert protocol["cases"] == cases()
    assert len(sources) == protocol["source_matches"] == 64
    assert len({source["seed"] for source in sources}) == 16
    assert provenance()["source_sha256"] == protocol["source"]["source_sha256"]
    for path, digest in protocol["inputs"].items():
        assert sha(path) == digest
        bindings[path] = digest
    for name, digest in plans["artifacts"].items():
        assert sha(base / name) == digest
        bindings[str(base / name)] = digest
    records = plans["records"]
    assert [r["source_index"] for r in records] == list(range(64))
    statuses = Counter(r["status"] for r in records)
    assert all(statuses[key] == count for key, count in complete["statuses"].items())
    selected = [r for r in records if r["status"] == "complete"]
    assert set(assessments) == {str(r["source_index"]) for r in selected}
    assert set(plans["artifacts"]) == {r["evidence_file"] for r in selected}
    source_audit, tail_audit, rows, gains = [], [], [], []
    for source, case, record in zip(sources, cases(), records, strict=True):
        assert {key: source[key] for key in case} == case
        work = replay_actions(
            ReferenceEnv(source["seed"], "paper8"), source["actions"], source["final"]
        )
        assert work == source["work"]
        source_audit.append(work)
        root = source["selected_root"]
        assert (root is None) == (record["status"] == "missing_root")
        if root is None:
            continue
        rebuilt = ReferenceEnv(source["seed"], "paper8")
        prefix_work = empty_work()
        for action in source["actions"][: root["decision"]]:
            add_step(prefix_work, rebuilt.step(action))
        np.testing.assert_array_equal(rebuilt.state, root["state"])
        # These additional prefix transitions are an audit, not new sources.
        source_audit.append(prefix_work)
        if record["status"] != "complete":
            continue
        artifact = torch.load(
            base / record["evidence_file"], map_location="cpu", weights_only=False
        )
        values = {}
        for name in ("fixed", "optimized"):
            plan = artifact["plans"][name]
            assert {str(a): str(v) for a, v in plan.values.items()} == record[name][
                "values"
            ]
            assert plan.action == record[name]["action"]
            assert plan.work == record[name]["work"]
            assert len(plan.policy) == record[name]["chosen_policy_nodes"]
            oracle = record[name]["reference_policy_replay"]
            assert oracle["values"] == record[name]["values"]
            for key, count in oracle["physical_work"].items():
                assert count <= plan.work[key]
                if name == "fixed":
                    assert count == plan.work[key]
            for key, count in oracle["represented_work"].items():
                assert count <= plan.work[f"represented_{key}"]
                if name == "fixed":
                    assert count == plan.work[f"represented_{key}"]
            assert oracle["represented_work"]["completions"] == (
                record["represented_prefixes"] * len(plan.values)
            )
            values[name] = plan.values
        fixed, optimized = artifact["plans"]["fixed"], artifact["plans"]["optimized"]
        assert fixed.baseline_action == optimized.baseline_action
        parent_q = values["fixed"][fixed.baseline_action]
        fixed_q, optimized_q = (
            max(values["fixed"].values()),
            max(values["optimized"].values()),
        )
        assert parent_q <= fixed_q <= optimized_q
        assert all(values["optimized"][a] >= q for a, q in values["fixed"].items())
        actual = assessments[str(record["source_index"])]
        for name, result in actual.items():
            assert (
                replay_actions(
                    reference(root["state"]), result["actions"], result["final"]
                )
                == result["work"]
            )
            tail_audit.append(result["work"])
            margin = (
                result["final"][POINTS + source["seat"]]
                - result["final"][POINTS + 1 - source["seat"]]
            )
            assert margin == result["point_margin"]
            if name == "parent":
                assert result["final"] == source["final"]
                assert result["actions"] == source["actions"][root["decision"] :]
        gain = {
            "source_index": record["source_index"],
            "parent": source["parent"],
            "fixed_minus_parent": str(fixed_q - parent_q),
            "optimized_minus_fixed": str(optimized_q - fixed_q),
            "optimized_minus_parent": str(optimized_q - parent_q),
            "realized_optimized_minus_parent": str(
                score(actual["optimized"]["point_margin"])
                - score(actual["parent"]["point_margin"])
            ),
            "realized_fixed_minus_parent": str(
                score(actual["fixed"]["point_margin"])
                - score(actual["parent"]["point_margin"])
            ),
        }
        gains.append(gain)
        rows.append(
            [
                record["source_index"],
                source["parent"] + 1,
                source["opponent"],
                source["seat"],
                record["hypothesis_rows"],
                record["represented_prefixes"],
                parent_q,
                fixed_q,
                optimized_q,
                "/".join(
                    str(actual[name]["point_margin"])
                    for name in ("parent", "fixed", "optimized")
                ),
            ]
        )
    work = {
        "sources": sum_work(source["work"] for source in sources),
        "native_source_verification": sum_work(source["work"] for source in sources),
        "history_support": sum_work(r.get("support_counts", {}) for r in records),
        "planning": sum_work(
            r[name]["work"]
            for r in records
            for name in ("fixed", "optimized")
            if name in r
        ),
        "independent_policy_replay": sum_work(
            r[name]["reference_policy_replay"]["physical_work"]
            for r in records
            for name in ("fixed", "optimized")
            if name in r
        ),
        "actual_tail_assessment": sum_work(
            result["work"] for a in assessments.values() for result in a.values()
        ),
        "export_source_and_root_audit": sum_work(source_audit),
        "export_tail_audit": sum_work(tail_audit),
    }
    assert work["sources"] == complete["source_work"]
    # Partial work is separate; completed fixed plans can also precede refusal.
    work["refused_partial_planning"] = sum_work(
        r.get("partial_work", {}) for r in records
    )
    work["independent_policy_replay_represented"] = sum_work(
        r[name]["reference_policy_replay"]["represented_work"]
        for r in records
        for name in ("fixed", "optimized")
        if name in r
    )
    for category, counts in (
        (
            "planning",
            [
                r[name]["work"]
                for r in records
                for name in ("fixed", "optimized")
                if name in r
            ],
        ),
        ("refused_partial_planning", [r.get("partial_work", {}) for r in records]),
    ):
        work[category]["maximum_frontier_rows"] = max(
            (row.get("maximum_frontier_rows", 0) for row in counts), default=0
        )
    aggregated = {}
    for parent in (0, 1):
        group = [g for g in gains if g["parent"] == parent]
        aggregated[f"parent{parent + 1}"] = {
            "sources": 32,
            "complete_roots": len(group),
            **{
                key: {
                    "sum": str(sum((Fraction(g[key]) for g in group), Fraction())),
                    "mean_over_all_32_sources": str(
                        sum((Fraction(g[key]) for g in group), Fraction()) / 32
                    ),
                    "positive_roots": sum(Fraction(g[key]) > 0 for g in group),
                    "negative_roots": sum(Fraction(g[key]) < 0 for g in group),
                }
                for key in (
                    "fixed_minus_parent",
                    "optimized_minus_fixed",
                    "optimized_minus_parent",
                    "realized_fixed_minus_parent",
                    "realized_optimized_minus_parent",
                )
            },
        }
    timings = {
        key: {
            "total_seconds": sum(r[key]["seconds"] for r in records if key in r),
            "maximum_seconds": max(
                (r[key]["seconds"] for r in records if key in r), default=0
            ),
        }
        for key in ("fixed", "optimized")
    }
    timings["support"] = {
        "total_seconds": sum(r.get("support_seconds", 0) for r in records),
        "maximum_seconds": max(
            (r.get("support_seconds", 0) for r in records), default=0
        ),
    }
    for name in ("fixed", "optimized"):
        timings[f"{name}_oracle"] = {
            "total_seconds": sum(
                r[name]["reference_replay_seconds"] for r in records if name in r
            ),
            "maximum_seconds": max(
                (r[name]["reference_replay_seconds"] for r in records if name in r),
                default=0,
            ),
        }
    timings["refused_partial_stage"] = {
        "total_seconds": sum(r.get("partial_stage_seconds", 0) for r in records),
        "maximum_seconds": max(
            (r.get("partial_stage_seconds", 0) for r in records), default=0
        ),
    }
    text = (
        "Five-draw belief-tail capability results\n=======================================\n\n"
        f"All 64 source matches completed on 16 distinct deal streams. Exact plans\n"
        f"completed at {statuses['complete']} roots; {statuses['missing_root']} sources had no\n"
        f"eligible root and {statuses['budget_refused']} roots exceeded a computation cap.\n"
        "Sources use two retained parents, assisted original WP/Point and both seats.\n"
        "This fixed capability check measures conditional opportunities and cost.\n"
        "Its actual-tail outcomes do not constitute a prospective strength test.\n\n"
    )
    text += table(
        "All completed roots",
        [
            "Source",
            "Parent",
            "Opponent",
            "Seat",
            "Hand/model rows",
            "Represented prefixes",
            "Parent Q",
            "Fixed Q",
            "Optimized Q",
            "Actual point margins parent/fixed/optimized",
        ],
        rows,
    )
    text += (
        "Q is the exact draw-adjusted match-win expectation under the declared\n"
        "uniform valid-deal and publicly identified deterministic-opponent model.\n"
        "Every root action's entire contingent policy was replayed with deferred chance\n"
        "with independent Python transitions. Opponent predictions in that replay\n"
        "came from recorded observation-keyed caches; neural calls were not repeated.\n"
        "All plans were frozen before actual-world assessment. Export independently\n"
        "replays source matches, selected prefixes and all actual tails.\n\n"
    )
    text += table(
        "All budget refusals",
        [
            "Source",
            "Parent",
            "Opponent",
            "Stage",
            "Reason",
            "Partial physical decisions",
        ],
        [
            [
                r["source_index"],
                sources[r["source_index"]]["parent"] + 1,
                sources[r["source_index"]]["opponent"],
                r["partial_stage"],
                r["reason"],
                r["partial_work"].get("decisions", 0),
            ]
            for r in records
            if r["status"] == "budget_refused"
        ],
    )
    text += (
        "Missing-root source indices: "
        + ", ".join(
            str(r["source_index"]) for r in records if r["status"] == "missing_root"
        )
        + ".\n\n"
    )
    text += (
        "Only roots with both plans complete enter actual-tail intervention. Both\n"
        "arms retain the parent for missing/refused roots in the following means.\n"
        "A completed fixed plan before an optimized refusal still contributes to\n"
        "the computation ledger, separately from the refused partial stage.\n\n"
    )
    text += table(
        "Means over all 32 sources per parent (percentage points; missing/refused roots contribute zero)",
        [
            "Parent",
            "Fixed minus parent Q",
            "Optimized minus fixed Q",
            "Optimized minus parent Q",
            "Realized fixed minus parent",
            "Realized optimized minus parent",
        ],
        [
            [
                parent,
                *[
                    f"{100 * float(Fraction(values[key]['mean_over_all_32_sources'])):+.6f}"
                    for key in (
                        "fixed_minus_parent",
                        "optimized_minus_fixed",
                        "optimized_minus_parent",
                        "realized_fixed_minus_parent",
                        "realized_optimized_minus_parent",
                    )
                ],
            ]
            for parent, values in aggregated.items()
        ],
    )
    text += "These conditional-value means describe the sampled source histories; they\nare not population estimates with a validated confidence interval. Source seats\nand parents share deals. No new independent sample is created by world expansion.\n\n"
    text += table(
        "Work accounting",
        ["Stage", "Counts"],
        [[key, json.dumps(value, sort_keys=True)] for key, value in work.items()],
    )
    text += table(
        "Measured planning cost",
        ["Stage", "Total seconds", "Maximum root seconds"],
        [
            [key, f"{value['total_seconds']:.6f}", f"{value['maximum_seconds']:.6f}"]
            for key, value in timings.items()
        ],
    )
    text += (
        "Fixed planning runs first; optimized planning reuses its policy caches.\n"
        "Fixed/optimized times exclude independent policy replay; the oracle rows\n"
        "report that additional work. These are not isolated throughput comparisons.\n"
        "Support cost is additional. Represented work counts prefix-mass units,\n"
        "not additional executed simulations or independent samples. No training, production-model\n"
        "promotion or reserved101M evaluation occurred. The broader strength\n"
        "objective remains open. See EARLIER_BELIEF_TAIL.rst for the fixed protocol.\n\n"
        f"Protocol SHA256: {complete['protocol_sha256']}.\n"
        f"Complete SHA256: {sha(base / 'complete.json')}.\n"
    )
    output.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "bindings": bindings,
            "complete_sha256": sha(base / "complete.json"),
            "report_sha256": sha(output),
            "statuses": dict(statuses),
            "gains": gains,
            "aggregated": aggregated,
            "work": work,
            "timings": timings,
        },
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "statuses": dict(statuses),
                "aggregated": aggregated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/earlier_belief_tail_capability")
    parser.add_argument("--output", default="docs/EARLIER_BELIEF_TAIL_RESULTS.rst")
    parser.add_argument("--receipt", default="runs/earlier_belief_tail_export.json")
    args = parser.parse_args()
    export(Path(args.base), Path(args.output), Path(args.receipt))
