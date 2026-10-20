"""Audit all recovery attempts and combine gains with the original source cohort."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import POINTS
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions, score, sum_work
from scripts.export_full_source_results import table
from scripts.recover_ceiling_tails import prerequisites
from scripts.train_branch_policy_gradient import sha, write


def gains(source, record, actual):
    fixed = {int(a): Fraction(q) for a, q in record["fixed"]["values"].items()}
    optimized = {int(a): Fraction(q) for a, q in record["optimized"]["values"].items()}
    parent = fixed[record["fixed"]["baseline_action"]]
    f, o = fixed[record["fixed"]["action"]], optimized[record["optimized"]["action"]]
    assert parent <= f <= o
    return {
        "source_index": record["source_index"],
        "parent": source["parent"],
        "parent_q": str(parent),
        "fixed_q": str(f),
        "optimized_q": str(o),
        "fixed_minus_parent": str(f - parent),
        "optimized_minus_fixed": str(o - f),
        "optimized_minus_parent": str(o - parent),
        "realized_fixed_minus_parent": str(
            score(actual["fixed"]["point_margin"])
            - score(actual["parent"]["point_margin"])
        ),
        "realized_optimized_minus_parent": str(
            score(actual["optimized"]["point_margin"])
            - score(actual["parent"]["point_margin"])
        ),
    }


def aggregate_work(rows):
    rows = list(rows)
    result = sum_work(rows)
    if "maximum_frontier_rows" in result:
        result["maximum_frontier_rows"] = max(
            row.get("maximum_frontier_rows", 0) for row in rows
        )
    return result


def export(base, equivalence, recovery, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError("Use fresh output and receipt paths")
    indices, old_inputs = prerequisites(base, equivalence)
    complete = read(recovery / "complete.json")
    protocol = read(recovery / "protocol.json")
    assert complete["status"] == read(recovery / "process.json")["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[key] == sha(recovery / name)
    assert protocol["source_indices"] == indices
    assert protocol["original_complete_sha256"] == sha(base / "complete.json")
    assert protocol["equivalence_complete_sha256"] == sha(equivalence / "complete.json")
    assert protocol["sources_sha256"] == sha(base / "sources.json")
    assert complete["new_source_matches"] == protocol["new_source_matches"] == 0
    assert all(sha(name) == digest for name, digest in protocol["inputs"].items())
    old_receipt = read("runs/earlier_belief_tail_export.json")
    assert old_receipt["status"] == "complete"
    assert old_receipt["exporter_sha256"] == sha(
        "scripts/export_earlier_belief_tail.py"
    )
    assert old_receipt["complete_sha256"] == sha(base / "complete.json")
    assert old_receipt["report_sha256"] == sha("docs/EARLIER_BELIEF_TAIL_RESULTS.rst")
    old_plans, old_actual = (
        read(base / "plans_complete.json"),
        read(base / "assessments.json"),
    )
    sources = read(base / "sources.json")
    assert len(sources) == 64 and len({s["seed"] for s in sources}) == 16
    plans, actual = (
        read(recovery / "plans_complete.json"),
        read(recovery / "assessments.json"),
    )
    records = plans["records"]
    assert [r["source_index"] for r in records] == indices
    assert plans["sources_sha256"] == protocol["sources_sha256"]
    selected = [r for r in records if r["status"] == "complete"]
    assert set(actual) == {str(r["source_index"]) for r in selected}
    assert set(plans["artifacts"]) == {r["evidence_file"] for r in selected}
    for name, digest in plans["artifacts"].items():
        assert sha(recovery / name) == digest
    assert all(
        sum(r["status"] == name for r in records) == count
        for name, count in complete["statuses"].items()
    )
    combined, extra, audit_work = [], [], []
    for record in old_plans["records"]:
        if record["status"] == "complete":
            index = record["source_index"]
            result = gains(sources[index], record, old_actual[str(index)])
            prior = next(g for g in old_receipt["gains"] if g["source_index"] == index)
            assert all(result[key] == value for key, value in prior.items())
            combined.append(result)
    assert len(combined) == 18
    for record in selected:
        index, source = record["source_index"], sources[record["source_index"]]
        artifact = torch.load(
            recovery / record["evidence_file"], map_location="cpu", weights_only=False
        )
        legal = set(
            map(
                int, np.flatnonzero(reference(source["selected_root"]["state"]).legal())
            )
        )
        for name in ("fixed", "optimized"):
            plan, saved = artifact["plans"][name], record[name]
            assert (
                plan.action == saved["action"]
                and plan.baseline_action == saved["baseline_action"]
            )
            assert (
                {str(a): str(q) for a, q in plan.values.items()}
                == saved["values"]
                == saved["oracle"]["values"]
            )
            assert list(plan.unevaluated_actions) == saved["unevaluated_actions"]
            assert plan.root_ceiling_attained == saved["root_ceiling_attained"]
            assert set(plan.values).isdisjoint(plan.unevaluated_actions)
            assert set(plan.values) | set(plan.unevaluated_actions) == legal
            assert not plan.unevaluated_actions or (
                plan.root_ceiling_attained and plan.values[plan.action] == 1
            )
            assert plan.policy == plan.root_policies[plan.action]
            assert plan.work == saved["work"]
            assert len(plan.policy) == saved["chosen_policy_nodes"]
            for key, count in saved["oracle"]["physical_work"].items():
                assert count <= plan.work[key]
                if name == "fixed":
                    assert count == plan.work[key]
            assert saved["oracle"]["represented_work"]["completions"] == int(
                artifact["worlds"].weights.sum()
            ) * len(plan.values)
        for name, result in actual[str(index)].items():
            work = replay_actions(
                reference(source["selected_root"]["state"]),
                result["actions"],
                result["final"],
            )
            assert work == result["work"]
            audit_work.append(work)
            margin = (
                result["final"][POINTS + source["seat"]]
                - result["final"][POINTS + 1 - source["seat"]]
            )
            assert result["point_margin"] == margin
            if name == "parent":
                assert (
                    result["actions"]
                    == source["actions"][source["selected_root"]["decision"] :]
                )
                assert result["final"] == source["final"]
        result = gains(source, record, actual[str(index)])
        extra.append(result)
        combined.append(result)
    assert (
        len(combined)
        == len({g["source_index"] for g in combined})
        == 18 + len(selected)
    )
    contrast_keys = (
        "fixed_minus_parent",
        "optimized_minus_fixed",
        "optimized_minus_parent",
        "realized_fixed_minus_parent",
        "realized_optimized_minus_parent",
    )
    summary = {}
    for parent in (0, 1):
        group = [g for g in combined if g["parent"] == parent]
        summary[f"parent{parent + 1}"] = {
            "sources": 32,
            "completed_roots": len(group),
            **{
                key: {
                    "mean_over_all_sources": str(
                        sum((Fraction(g[key]) for g in group), Fraction()) / 32
                    ),
                    "positive_roots": sum(Fraction(g[key]) > 0 for g in group),
                    "negative_roots": sum(Fraction(g[key]) < 0 for g in group),
                }
                for key in contrast_keys
            },
        }
    work = {
        "new_source_matches": 0,
        "history_support": aggregate_work(r["support_counts"] for r in records),
        "completed_planning": aggregate_work(
            r[n]["work"] for r in records for n in ("fixed", "optimized") if n in r
        ),
        "refused_partial_planning": aggregate_work(
            r.get("partial_work", {}) for r in records
        ),
        "independent_policy_oracle": aggregate_work(
            r[n]["oracle"]["physical_work"]
            for r in records
            for n in ("fixed", "optimized")
            if n in r
        ),
        "represented_oracle_units": aggregate_work(
            r[n]["oracle"]["represented_work"]
            for r in records
            for n in ("fixed", "optimized")
            if n in r
        ),
        "actual_tails": aggregate_work(
            v["work"] for row in actual.values() for v in row.values()
        ),
        "export_tail_audit": aggregate_work(audit_work),
    }
    timings = {
        name: sum(r[name]["seconds"] for r in records if name in r)
        for name in ("fixed", "optimized")
    }
    timings.update(
        support=sum(r["support_seconds"] for r in records),
        independent_oracle=sum(
            r[n]["oracle_seconds"]
            for r in records
            for n in ("fixed", "optimized")
            if n in r
        ),
        refused_partial=sum(r.get("partial_stage_seconds", 0) for r in records),
    )
    text = (
        "Ceiling-pruned tail recovery\n===========================\n\n"
        f"All fifteen earlier refusals were attempted. {len(selected)} roots now complete;\n"
        f"{complete['statuses']['budget_refused']} remain refused. The original cohort therefore has\n"
        f"{18 + len(selected)} completed roots, {complete['statuses']['budget_refused']} remaining refusals and 31 missing roots\n"
        "among 64 source matches on 16 distinct deal streams. No source match was added.\n"
        "The limits changed from 500,000 all nodes to 500,000 live nodes plus 2,000,000\n"
        "physical transitions per plan. This is an exploratory computational\n"
        "extension, not a new prospective strength evaluation.\n\n"
    )
    by_index = {g["source_index"]: g for g in extra}
    rows = []
    for r in records:
        index, g = r["source_index"], by_index.get(r["source_index"])
        margins = (
            "/".join(
                str(actual[str(index)][n]["point_margin"])
                for n in ("parent", "fixed", "optimized")
            )
            if g
            else "-"
        )
        rows.append(
            [
                index,
                sources[index]["parent"] + 1,
                sources[index]["opponent"],
                r["status"],
                g["parent_q"] if g else "-",
                g["fixed_q"] if g else "-",
                g["optimized_q"] if g else "-",
                margins,
                r.get("reason", "-"),
            ]
        )
    text += table(
        "All fifteen attempts",
        [
            "Source",
            "Parent",
            "Opponent",
            "Status",
            "Parent Q",
            "Fixed Q",
            "Optimized Q",
            "Actual point margins P/F/O",
            "Refusal reason",
        ],
        rows,
    )
    text += table(
        "Combined means over all 32 sources per parent (percentage points)",
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
                    f"{100 * float(Fraction(v[key]['mean_over_all_sources'])):+.6f}"
                    for key in contrast_keys
                ],
            ]
            for parent, v in summary.items()
        ],
    )
    text += (
        "Missing/refused roots retain the parent in both arms. Earlier 18 completed\n"
        "policies/outcomes are reused through exact Stage 1 equivalence. Q values\n"
        "use the declared uniform valid-deal and publicly identified deterministic\n"
        "opponent model. Every computed recovery value passes independent Python\n"
        "chance/policy replay. Omitted root actions remain unevaluated, certified\n"
        "unnecessary only when a chosen value reaches 1. These source-history means\n"
        "are descriptive; no population confidence interval or new win criterion\n"
        "is asserted. All plans were frozen before actual-tail assessment.\n\n"
    )
    text += table(
        "Additional work only",
        ["Stage", "Counts"],
        [[k, json.dumps(v, sort_keys=True)] for k, v in work.items()],
    )
    text += table(
        "Additional elapsed components",
        ["Stage", "Seconds"],
        [[k, f"{v:.6f}"] for k, v in timings.items()],
    )
    text += (
        "Completed fixed work preceding an optimized refusal remains in the work\n"
        "ledger; the incomplete comparison contributes no intervention gain.\n"
        "Represented prefix units are not actual simulations or independent data.\n"
        "Policy replay uses cached observation-keyed opponent actions; actual tails\n"
        "use fresh strict calls. Export replays all new actual tails and reuses the\n"
        "bound original source/prefix audit. No training, reserved101M evaluation,\n"
        "canonical promotion or manuscript rebuild occurred.\n\n"
        f"Recovery protocol SHA256: {complete['protocol_sha256']}.\n"
        f"Recovery complete SHA256: {sha(recovery / 'complete.json')}.\n"
    )
    output.write_bytes(text.encode("utf-8"))
    write(
        receipt,
        {
            "status": "complete",
            "exporter_sha256": sha(__file__),
            "recovery_complete_sha256": sha(recovery / "complete.json"),
            "original_complete_sha256": sha(base / "complete.json"),
            "equivalence_complete_sha256": sha(equivalence / "complete.json"),
            "original_export_sha256": sha("runs/earlier_belief_tail_export.json"),
            "report_sha256": sha(output),
            "bindings": {**old_inputs, **protocol["inputs"]},
            "recovery_gains": extra,
            "combined_gains": combined,
            "summary": summary,
            "work": work,
            "timings": timings,
            "statuses": complete["statuses"],
        },
    )
    print(
        json.dumps(
            {
                "statuses": complete["statuses"],
                "summary": summary,
                "report": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/earlier_belief_tail_capability")
    parser.add_argument("--equivalence", default="runs/ceiling_tail_equivalence")
    parser.add_argument("--recovery", default="runs/ceiling_tail_recovery")
    parser.add_argument("--output", default="docs/CEILING_RECOVERY_RESULTS.rst")
    parser.add_argument("--receipt", default="runs/ceiling_tail_recovery_export.json")
    args = parser.parse_args()
    export(
        Path(args.base),
        Path(args.equivalence),
        Path(args.recovery),
        Path(args.output),
        Path(args.receipt),
    )
