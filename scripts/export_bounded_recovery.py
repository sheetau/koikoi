"""Export every bounded recovery attempt, including useful incomplete comparisons."""

from collections import Counter
from pathlib import Path

import torch

from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions
from scripts.export_full_source_results import table
from scripts.train_branch_policy_gradient import sha, write


def export():
    base = Path("runs/bounded_tail_recovery")
    output = Path("docs/BOUNDED_RECOVERY_RESULTS.rst")
    receipt = Path("runs/bounded_tail_recovery_export.json")
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh outputs")
    complete, protocol = read(base / "complete.json"), read(base / "protocol.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    for field, name in (
        ("protocol_sha256", "protocol.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[field] == sha(base / name)
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    plans, actual = read(base / "plans_complete.json"), read(base / "assessments.json")
    assert [r["source_index"] for r in plans["records"]] == protocol["source_indices"]
    sources = read("runs/earlier_belief_tail_capability/sources.json")
    work = {
        name: Counter()
        for name in (
            "completed_plans",
            "refused_partial",
            "oracle",
            "actual",
            "export_actual_replay",
        )
    }
    times = Counter()
    rows, fixed_gains, full_gains = [], [], []
    assert set(actual) == {
        str(r["source_index"]) for r in plans["records"] if r["status"] == "complete"
    }
    for record in plans["records"]:
        index = record["source_index"]
        path = base / record["evidence_file"]
        assert sha(path) == plans["artifacts"][path.name]
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        for name, plan in artifact["plans"].items():
            saved = record[name]
            assert plan.work == saved["work"]
            assert (
                {str(a): str(q) for a, q in plan.values.items()}
                == saved["values"]
                == saved["oracle"]["values"]
            )
            assert {
                str(a): str(q) for a, q in plan.action_upper_bounds.items()
            } == saved["action_upper_bounds"]
            assert (
                plan.action == saved["action"]
                and plan.baseline_action == saved["baseline_action"]
            )
            assert plan.policy == plan.root_policies[plan.action]
            work["completed_plans"].update(
                {k: v for k, v in plan.work.items() if k != "maximum_frontier_rows"}
            )
            work["oracle"].update(saved["oracle"]["physical_work"])
            times[name] += saved["seconds"]
            times["oracle"] += saved["oracle_seconds"]
        if record["status"] == "budget_refused":
            work["refused_partial"].update(
                {
                    k: v
                    for k, v in record["partial_work"].items()
                    if k != "maximum_frontier_rows"
                }
            )
            times["refused_partial"] += record["partial_seconds"]
        parent_q = fixed_q = optimized_q = "-"
        if "fixed" in artifact["plans"]:
            fixed = artifact["plans"]["fixed"]
            parent_q, fixed_q = (
                fixed.values[fixed.baseline_action],
                fixed.values[fixed.action],
            )
            assert fixed_q >= parent_q
            fixed_gains.append({"source": index, "gain": str(fixed_q - parent_q)})
        if record["status"] == "complete":
            optimized = artifact["plans"]["optimized"]
            optimized_q = optimized.values[optimized.action]
            assert optimized_q >= fixed_q
            full_gains.append({"source": index, "gain": str(optimized_q - parent_q)})
            for name, assessment in actual[str(index)].items():
                work["actual"].update(assessment["work"])
                checked = replay_actions(
                    reference(sources[index]["selected_root"]["state"]),
                    assessment["actions"],
                    assessment["final"],
                )
                assert checked == assessment["work"]
                work["export_actual_replay"].update(checked)
                if name == "parent":
                    source = sources[index]
                    assert (
                        assessment["actions"]
                        == source["actions"][source["selected_root"]["decision"] :]
                    )
                    assert assessment["final"] == source["final"]
        rows.append(
            [
                index,
                record["status"],
                record.get("partial_stage", "-"),
                str(parent_q),
                str(fixed_q),
                str(optimized_q),
                record.get("reason", "-"),
            ]
        )
    statuses = Counter(r["status"] for r in plans["records"])
    assert all(statuses[name] == count for name, count in complete["statuses"].items())
    text = (
        "Partial-action bound recovery results\n"
        "====================================\n\n"
        f"All five refusals were attempted: {statuses['complete']} complete "
        f"fixed+optimized comparisons and {statuses['budget_refused']} remaining "
        "refusals. No source matches or history-support decisions were added.\n"
        "Limits remained500k live groups/2M physical transitions per plan,\n"
        "4096 frontier rows/depth32. Support worlds were reconstructed exactly\n"
        "from the bound profile artifacts; policy caches started fresh.\n\n"
    )
    text += table(
        "Every attempt",
        [
            "Source",
            "Status",
            "Interrupted stage",
            "Parent Q",
            "Completed fixed Q",
            "Optimized Q",
            "Reason",
        ],
        rows,
    )
    text += (
        "Completed fixed plans remain valid even when subsequent optimization\n"
        "is refused. Their gains are feasible model-conditional opportunities,\n"
        "not completed fixed-versus-optimized comparisons or actual win gains.\n"
        "The protocol assesses actual tails only after both plans complete.\n\n"
    )
    text += table(
        "Completed fixed-policy opportunities",
        ["Source", "Exact fixed-minus-parent Q"],
        [[r["source"], r["gain"]] for r in fixed_gains],
    )
    text += table(
        "Additional work",
        ["Category", "Counter", "Count"],
        [
            [category, k, v]
            for category, counters in work.items()
            for k, v in counters.items()
        ],
    )
    text += table(
        "Elapsed components (not a paired speed comparison)",
        ["Stage", "Seconds"],
        [[k, f"{v:.6f}"] for k, v in times.items()],
    )
    text += (
        "Actual continuations and independent Python policy replay are separate\n"
        "from source matches. Represented prefix counts are not physical simulations\n"
        "or independent data. The prior28 completed and31 missing roots remain in\n"
        "the original64-source cohort; remaining refusals retain the parent under\n"
        "this comparison protocol. No new empirical strength claim, promotion,\n"
        "reserved101M use or manuscript change follows from these diagnostics.\n\n"
        f"Complete SHA256: {sha(base / 'complete.json')}.\n"
        f"Protocol SHA256: {complete['protocol_sha256']}.\n"
    )
    output.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "complete_sha256": sha(base / "complete.json"),
            "report_sha256": sha(output),
            "statuses": dict(statuses),
            "fixed_gains": fixed_gains,
            "full_gains": full_gains,
            "work": {k: dict(v) for k, v in work.items()},
            "seconds": dict(times),
        },
    )
    print({"statuses": dict(statuses), "fixed_gains": fixed_gains})


if __name__ == "__main__":
    export()
