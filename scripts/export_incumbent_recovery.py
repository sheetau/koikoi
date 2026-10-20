"""Audit retained incumbents and update the original 64-source controller cohort."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch

from koikoi.layout import POINTS
from scripts.check_incumbent_recovery import describe
from scripts.export_belief_tail import read, score
from scripts.export_full_source_results import table
from scripts.incumbent_tail_controller import select_improvement
from scripts.train_branch_policy_gradient import sha, write


def export():
    base = Path("runs/incumbent_tail_recovery")
    previous = Path("runs/incumbent_saved_execution")
    output = Path("docs/INCUMBENT_RECOVERY_RESULTS.rst")
    receipt = Path("runs/incumbent_recovery_export.json")
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh outputs")
    complete, protocol = read(base / "complete.json"), read(base / "protocol.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("selections_sha256", "selections_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[key] == sha(base / name)
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    old = read(previous / "complete.json")
    assert old["status"] == "complete"
    assert old["selections_sha256"] == sha(previous / "selections_complete.json")
    assert old["assessments_sha256"] == sha(previous / "assessments.json")
    selected = read(base / "selections_complete.json")
    assert [r["source_index"] for r in selected["records"]] == protocol[
        "source_indices"
    ]
    combined = read(previous / "selections_complete.json")["records"]
    outcomes = read(previous / "assessments.json")
    fresh = read(base / "assessments.json")
    assert set(fresh) == {str(i) for i in protocol["source_indices"]}
    sources = read("runs/earlier_belief_tail_capability/sources.json")
    work = {
        k: Counter()
        for k in (
            "completed_search_phases",
            "capped_search_phases",
            "new_oracle_physical",
            "new_oracle_represented",
        )
    }
    times, calls = Counter(), Counter()
    rows = []
    for record in selected["records"]:
        index, name = record["source_index"], record["evidence_file"]
        assert sha(base / name) == selected["artifacts"][name]
        evidence = torch.load(base / name, map_location="cpu", weights_only=False)
        selection = select_improvement(evidence["fixed"], evidence["optimized"])
        assert selection == evidence["selection"]
        saved_selection = record["selection"]
        assert str(selection.parent_q) == saved_selection["parent_q"]
        assert str(selection.selected_q) == saved_selection["selected_q"]
        assert str(selection.gain) == saved_selection["gain"]
        assert selection.mode == saved_selection["mode"]
        for mode in ("fixed", "optimized"):
            phase = evidence[mode]
            if phase is None:
                assert mode not in record["phases"]
                continue
            saved = record["phases"][mode]
            assert all(saved[k] == v for k, v in describe(phase.result).items())
            category = (
                "completed_search_phases"
                if phase.result.refusal is None
                else "capped_search_phases"
            )
            work[category].update(
                {
                    k: v
                    for k, v in phase.result.work.items()
                    if k != "maximum_frontier_rows"
                }
            )
            times[mode] += saved["seconds"]
            calls["new_parent_adapter_calls"] += saved["new_parent_calls"]
            calls["new_opponent_adapter_calls"] += saved["new_opponent_calls"]
            oracle = saved.get("oracle")
            assert (phase.result.plan is None) == (oracle is None)
            if oracle is not None:
                if oracle.get("reused_complete_policy_evidence"):
                    prior = torch.load(
                        f"runs/bounded_tail_recovery/root_{index:03}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["plans"][mode]
                    assert phase.result.plan.values == prior.values
                    assert phase.result.plan.root_policies == prior.root_policies
                else:
                    assert {
                        str(a): str(q) for a, q in phase.result.plan.values.items()
                    } == oracle["values"]
                    work["new_oracle_physical"].update(oracle["physical_work"])
                    work["new_oracle_represented"].update(oracle["represented_work"])
                    times["new_oracle"] += saved["oracle_seconds"]
        result = fresh[str(index)]
        source = sources[index]
        assert result["parent"]["final"] == source["final"]
        assert (
            result["parent"]["actions"]
            == source["actions"][source["selected_root"]["decision"] :]
        )
        for assessment in result.values():
            final = assessment["final"]
            assert (
                assessment["point_margin"]
                == final[POINTS + source["seat"]] - final[POINTS + 1 - source["seat"]]
            )
            assert len(assessment["actions"]) == assessment["work"]["decisions"]
        if selection.plan is None:
            assert result["controller"] == result["parent"]
        combined[index].update(saved_selection)
        combined[index]["plan_artifact"] = str(base / name)
        combined[index]["world_artifact"] = str(base / name)
        combined[index]["phase_optimality_gap"] = (
            None if selection.plan is None else str(selection.plan.optimality_gap)
        )
        outcomes[str(index)] = result
        rows.append(
            [
                index,
                selection.mode,
                str(selection.parent_q),
                str(selection.selected_q),
                str(selection.gain),
                combined[index]["phase_optimality_gap"],
                result["parent"]["point_margin"],
                result["controller"]["point_margin"],
            ]
        )
    actual_totals = Counter()
    for result in fresh.values():
        for assessment in result.values():
            actual_totals.update(assessment["work"])
    assert dict(actual_totals) == complete["actual_work"] == complete["replay_work"]
    summary = {}
    for parent in (0, 1):
        indices = [i for i, s in enumerate(sources) if s["parent"] == parent]
        conditional = [Fraction(combined[i]["gain"]) for i in indices]
        realized = [
            score(outcomes[str(i)]["controller"]["point_margin"])
            - score(outcomes[str(i)]["parent"]["point_margin"])
            if str(i) in outcomes
            else Fraction(0)
            for i in indices
        ]
        summary[str(parent + 1)] = {
            "sources": len(indices),
            "adoptions": sum(q > 0 for q in conditional),
            "conditional_mean_gain": str(sum(conditional) / len(indices)),
            "realized_mean_gain": str(sum(realized) / len(indices)),
            "realized_changed_outcomes": sum(q != 0 for q in realized),
        }
    text = (
        "Retained-incumbent recovery results\n"
        "==================================\n\n"
        "All five former full-comparison refusals run the actual capped\n"
        "fixed-then-optimized controller. Every fixed parent reference matches\n"
        "its previously completed root evaluation. Previously attempted phases\n"
        "reproduce exact work with zero new adapter calls; three optimized\n"
        "phases are explicitly new work. No source/history replay was added.\n\n"
    )
    text += table(
        "All five frozen selections and actual tails",
        [
            "Source",
            "Mode",
            "Parent Q",
            "Selected Q",
            "Exact gain",
            "Within-phase optimum gap",
            "Parent margin",
            "Controller margin",
        ],
        rows,
    )
    text += table(
        "Combined original cohort, means over32 sources per parent",
        [
            "Parent",
            "Adoptions",
            "Conditional gain",
            "Realized gain",
            "Changed outcomes",
        ],
        [
            [
                p,
                v["adoptions"],
                v["conditional_mean_gain"],
                v["realized_mean_gain"],
                v["realized_changed_outcomes"],
            ]
            for p, v in summary.items()
        ],
    )
    text += (
        "The selected policy is complete and exactly evaluated even when its\n"
        "phase hits a cap. A positive optimum gap leaves the best possible policy\n"
        "unresolved. A fixed-phase gap says nothing about optimizing later choices.\n"
        "Parent fallback remains explicit if no strict exact gain is available.\n\n"
    )
    text += table(
        "Additional search and independent policy-replay work",
        ["Category", "Counter", "Count"],
        [
            [category, k, v]
            for category, counters in work.items()
            for k, v in counters.items()
        ],
    )
    text += table(
        "Other additional work",
        ["Category", "Counter", "Count"],
        [
            [category, k, v]
            for category, counters in (
                ("new_adapter_calls", calls),
                ("actual_tails", complete["actual_work"]),
                ("separate_trace_replay", complete["replay_work"]),
                ("actual_actor_calls", complete["actor_calls"]),
            )
            for k, v in counters.items()
        ],
    )
    text += table(
        "Elapsed components; caches reused, not a speed comparison",
        ["Stage", "Seconds"],
        [[k, f"{v:.6f}"] for k, v in times.items()],
    )
    text += (
        "The64 original sources /16 streams remain the denominator. These five\n"
        "selections replace their earlier saved-plan selections; the other28\n"
        "available roots and31 missing roots are unchanged. Conditional tail\n"
        "completions and reused action caches are not independent samples.\n"
        "No prospective win-rate replication, training, deployment, reserved101M\n"
        "evaluation or manuscript rebuild is claimed.\n\n"
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
            "previous_complete_sha256": sha(previous / "complete.json"),
            "summary": summary,
            "combined_records": combined,
            "additional_work": {k: dict(v) for k, v in work.items()},
            "new_adapter_calls": dict(calls),
            "seconds": dict(times),
        },
    )
    print(summary)


if __name__ == "__main__":
    export()
