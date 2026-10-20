"""Audit and summarize the closed all-root partial-action bound comparison."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch

from scripts.export_belief_tail import read
from scripts.export_full_source_results import table
from scripts.train_branch_policy_gradient import sha, write


def export():
    base = Path("runs/bounded_tail_equivalence")
    output = Path("docs/BOUNDED_TAIL_RESULTS.rst")
    receipt = Path("runs/bounded_tail_export.json")
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh outputs")
    complete, protocol = read(base / "complete.json"), read(base / "protocol.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    assert complete["protocol_sha256"] == sha(base / "protocol.json")
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    records = complete["records"]
    assert [r["source"] for r in records] == protocol["source_indices"]
    assert len(records) == 28
    totals = {method: Counter() for method in ("ceiling", "bounded")}
    seconds, maximum = dict.fromkeys(totals, 0.0), dict.fromkeys(totals, 0.0)
    root_bounds = 0
    rows = []
    for record in records:
        path = base / f"root_{record['source']:03}.pt"
        assert sha(path) == complete["artifacts"][path.name]
        plans = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("fixed", "optimized"):
            plan, bounded = plans[name], record["bounded"][name]
            control = record["ceiling"][name]
            assert plan.work == bounded["work"]
            assert {str(a): str(q) for a, q in plan.values.items()} == bounded["values"]
            assert {
                str(a): str(q) for a, q in plan.action_upper_bounds.items()
            } == bounded["action_upper_bounds"]
            assert plan.action == bounded["action"] == control["action"]
            assert plan.values[plan.action] == Fraction(
                control["values"][str(plan.action)]
            )
            assert plan.policy == plan.root_policies[plan.action]
            assert set(plan.values).isdisjoint(plan.action_upper_bounds)
            assert all(
                Fraction(control["values"][str(a)]) == q for a, q in plan.values.items()
            )
            for action, upper in plan.action_upper_bounds.items():
                if str(action) in control["values"]:
                    assert (
                        Fraction(control["values"][str(action)])
                        <= upper
                        <= plan.values[plan.action]
                    )
                else:
                    assert upper == plan.values[plan.action] == 1
            root_bounds += sum(q < 1 for q in plan.action_upper_bounds.values())
            for method, work in totals.items():
                saved = record[method][name]
                seconds[method] += saved["seconds"]
                maximum[method] = max(maximum[method], saved["seconds"])
                work.update(
                    {
                        k: v
                        for k, v in saved["work"].items()
                        if k != "maximum_frontier_rows"
                    }
                )
            rows.append(
                [
                    record["source"],
                    name,
                    plan.action,
                    str(plan.values[plan.action]),
                    control["work"]["decisions"],
                    bounded["work"]["decisions"],
                    f"{control['seconds']:.6f}",
                    f"{bounded['seconds']:.6f}",
                    len(plan.action_upper_bounds),
                ]
            )
    assert Counter(r["order"][0] for r in records) == {"ceiling": 14, "bounded": 14}
    decision_reduction = 100 * (
        1 - totals["bounded"]["decisions"] / totals["ceiling"]["decisions"]
    )
    time_reduction = 100 * (1 - seconds["bounded"] / seconds["ceiling"])
    text = (
        "Bounded continuation comparison\n"
        "===============================\n\n"
        "All 28 completed roots / 56 fixed and optimized plans preserve the\n"
        "chosen action, exact value and entire chosen policy. Every computed\n"
        "root value/policy matches the saved independently verified solution.\n"
        "Partial upper bounds contain the control's exact values and cannot\n"
        "beat the incumbent. Fourteen roots run each method first; caches are\n"
        "fresh and separate, with fixed then optimized within each method.\n\n"
        f"Physical decisions fall from {totals['ceiling']['decisions']:,} to "
        f"{totals['bounded']['decisions']:,} ({decision_reduction:.4f}% reduction).\n"
        f"Observed total planning time is {seconds['ceiling']:.6f}s versus "
        f"{seconds['bounded']:.6f}s ({time_reduction:.4f}% reduction).\n"
        f"The largest individual plan takes {maximum['ceiling']:.6f}s versus "
        f"{maximum['bounded']:.6f}s. These are paired observations, not a\n"
        "latency distribution or a prospective strength evaluation.\n\n"
        f"There are {root_bounds} root-action upper bounds below1; "
        f"{totals['bounded']['bounded_actions']:,} candidate actions are\n"
        "discarded after partial evaluation across the full trees.\n\n"
    )
    text += table(
        "Every completed plan",
        [
            "Source",
            "Mode",
            "Action",
            "Exact chosen Q",
            "Ceiling decisions",
            "Bounded decisions",
            "Ceiling seconds",
            "Bounded seconds",
            "Root bounds",
        ],
        rows,
    )
    text += table(
        "Additional planning work, kept separate from source matches",
        ["Method", "Counter", "Count"],
        [[method, k, v] for method in totals for k, v in totals[method].items()],
    )
    text += (
        "Both methods use500k live groups,2M physical transitions,4096 frontier\n"
        "rows and depth32. No source match, training update or actual-tail\n"
        "assessment was added. Existing Python oracle evidence is reused by\n"
        "exact equality of full policies and values; no new full Python or\n"
        "neural replay is claimed. This is an exact computational improvement,\n"
        "not a new match-win result or a claim of general algorithm priority.\n\n"
        f"Protocol SHA256: {complete['protocol_sha256']}.\n"
        f"Complete SHA256: {sha(base / 'complete.json')}.\n"
    )
    output.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "complete_sha256": sha(base / "complete.json"),
            "report_sha256": sha(output),
            "totals": {k: dict(v) for k, v in totals.items()},
            "seconds": seconds,
            "maximum_seconds": maximum,
            "root_partial_upper_bounds": root_bounds,
            "decision_reduction_percent": decision_reduction,
            "time_reduction_percent": time_reduction,
        },
    )
    print(
        {
            "seconds": seconds,
            "decisions": {k: v["decisions"] for k, v in totals.items()},
        }
    )


if __name__ == "__main__":
    export()
