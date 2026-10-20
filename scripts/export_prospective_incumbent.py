"""Audit and export the complete, fixed prospective endgame comparison."""

import argparse
import gzip
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions, score
from scripts.export_full_source_results import table
from scripts.incumbent_tail_controller import bind_worlds, select_improvement
from scripts.prospective_incumbent import check, describe_selection
from scripts.run_arbitration_study import PARENTS, sha, write


def audit_arm(evidence, row, source, arm, protocol):
    worlds = evidence["worlds"]
    model = source["selected_root"]["public_identification"]["model"]
    view = SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal)
    binding = bind_worlds(
        worlds,
        view,
        PARENTS[source["parent"]][1],
        {model: protocol["opponent_identities"][model]},
    )
    assert (
        binding.root_key
        == visible_keys(
            np.asarray(source["selected_root"]["state"], dtype=np.int32)[None],
            source["seat"],
        )[0]
    )
    phases = evidence["phases"]
    assert set(phases) in ({"fixed"}, {"fixed", "optimized"})
    if arm == "root_only":
        assert set(phases) == {"fixed"}
    for name, phase in phases.items():
        assert phase.binding == binding and phase.mode == name
        assert phase.result.work == row["phases"][name]["work"]
        assert phase.result.refusal == row["phases"][name]["refusal"]
        allowance = protocol["phase_decisions"] * (2 if arm == "root_only" else 1)
        assert phase.result.work["decisions"] <= allowance
    selection = select_improvement(phases["fixed"], phases.get("optimized"))
    assert selection == evidence["selection"]
    assert describe_selection(selection) == row["selection"]
    if selection.plan is not None:
        parent = phases["fixed"].result.plan
        for name, plan, action in (
            ("parent", parent, parent.baseline_action),
            ("selected", selection.plan, selection.plan.action),
        ):
            oracle = row["oracle"][name]
            assert oracle["values"] == {str(action): str(plan.values[action])}
            assert oracle["represented_work"]["completions"] == sum(
                map(int, worlds.weights)
            )
    else:
        assert row["oracle"] == {}
    return selection


def audit_source(source):
    env = ReferenceEnv(source["seed"], "paper8")
    work = empty_work()
    root = source["selected_root"]
    found = False
    for decision, action in enumerate(source["actions"]):
        assert env.state[PHASE] != TERMINAL
        if root is not None and root["decision"] == decision:
            np.testing.assert_array_equal(env.state, root["state"])
            found = True
        assert env.legal()[action]
        add_step(work, env.step(action))
    assert (root is not None) == found
    assert env.state[PHASE] == TERMINAL
    np.testing.assert_array_equal(env.state, source["final"])
    assert work == source["work"]
    return work


def export(base, output, receipt):
    if output.exists() or receipt.exists():
        raise ValueError("Use new report and receipt paths")
    protocol = check(base)
    summary = read(base / "summary.json")
    assert summary["status"] == read(base / "queue.json")["status"] == "complete"
    assert summary["protocol_sha256"] == sha(base / "protocol.json")
    source_work, tail_work = Counter(), Counter()
    actual = [Fraction(0)] * 4
    conditional = [Fraction(0)] * 4
    rows, bindings = [], {}
    for chunk in range(len(protocol["cases"]) // protocol["chunk_size"]):
        folder = base / f"chunk_{chunk:03}"
        complete_path = folder / "complete.json"
        digest = sha(complete_path)
        assert summary["chunk_bindings"][str(complete_path)] == digest
        bindings[str(complete_path)] = digest
        complete = read(complete_path)
        assert complete["protocol_sha256"] == summary["protocol_sha256"]
        assert (
            complete["status"] == read(folder / "process.json")["status"] == "complete"
        )
        assert complete["records_sha256"] == sha(folder / "records.json")
        for name, expected in complete["artifacts"].items():
            assert sha(folder / name) == expected
        records = read(folder / "records.json")
        for row in records:
            source = row["source"]
            source_work.update(audit_source(source))
            parent = source["parent"]
            margin = (
                source["final"][POINTS + source["seat"]]
                - source["final"][POINTS + 1 - source["seat"]]
            )
            scores = dict.fromkeys(("parent", "root_only", "controller"), score(margin))
            gains = dict.fromkeys(("parent", "root_only", "controller"), Fraction(0))
            if row["status"] == "evaluated":
                selections = {}
                for arm, record in row["arms"].items():
                    assert (
                        sha(folder / record["artifact"])
                        == complete["artifacts"][record["artifact"]]
                    )
                    with gzip.open(folder / record["artifact"], "rb") as handle:
                        evidence = pickle.load(handle)
                    selection = audit_arm(evidence, record, source, arm, protocol)
                    selections[arm] = selection
                    gains[arm] = selection.gain or Fraction(0)
                    del evidence
                assert (
                    selections["controller"].binding == selections["root_only"].binding
                )
                assert read(folder / f"selection_{row['index']:04}.json") == row["arms"]
                for arm, outcome in row["assessment"]["outcomes"].items():
                    work = replay_actions(
                        reference(source["selected_root"]["state"]),
                        outcome["actions"],
                        outcome["final"],
                    )
                    assert work == outcome["work"]
                    tail_work.update(work)
                    final = outcome["final"]
                    assert (
                        outcome["margin"]
                        == final[POINTS + source["seat"]]
                        - final[POINTS + 1 - source["seat"]]
                    )
                    scores[arm] = score(outcome["margin"])
                outcomes = row["assessment"]["outcomes"]
                assert outcomes["parent"]["final"] == source["final"]
                assert (
                    outcomes["parent"]["actions"]
                    == source["actions"][source["selected_root"]["decision"] :]
                )
                for arm, selection in selections.items():
                    if selection.plan is None:
                        assert outcomes[arm] == outcomes["parent"]
            else:
                assert row["status"] in ("missing_root", "support_refused")
                assert (source["selected_root"] is None) == (
                    row["status"] == "missing_root"
                )
            for offset, control in enumerate(("parent", "root_only")):
                actual[2 * parent + offset] += scores["controller"] - scores[control]
                conditional[2 * parent + offset] += gains["controller"] - gains[control]
            rows.append(row)
    assert [row["index"] for row in rows] == list(range(512))
    for row, case in zip(rows, protocol["cases"], strict=True):
        assert {k: row["source"][k] for k in case} == case
    for exact, key in ((actual, "actual_mean"), (conditional, "conditional_mean")):
        np.testing.assert_allclose(
            [float(v / 256) for v in exact], summary[key], rtol=0, atol=1e-12
        )
    assert dict(source_work) == summary["work"]["source"]
    assert dict(tail_work) == summary["work"]["actual_tails"]
    assert summary["development_strength_gate"] == all(
        value > 0 for value in summary["actual_simultaneous95_lower"]
    )
    result_rows = [
        [
            name,
            f"{100 * mean:+.4f}",
            f"[{100 * lo:+.4f}, {100 * hi:+.4f}]",
            f"{100 * expected:+.4f}",
        ]
        for name, mean, lo, hi, expected in zip(
            summary["contrast_order"],
            summary["actual_mean"],
            summary["actual_simultaneous95_lower"],
            summary["actual_simultaneous95_upper"],
            summary["conditional_mean"],
            strict=True,
        )
    ]
    verdict = "passes" if summary["development_strength_gate"] else "does not pass"
    text = (
        "Prospective contingent-policy comparison\n"
        "========================================\n\n"
        f"The fixed development strength gate {verdict}. "
        "All512 sources and128 deal streams are included.\n\n"
        + table(
            "Paired improvements",
            [
                "Contrast",
                "Actual win-score difference (pp)",
                "Simultaneous95% interval (pp)",
                "Model-conditional difference (pp)",
            ],
            result_rows,
        )
        + "The actual score is win=1, draw=1/2, loss=0. Two seats are averaged "
        "within a deal; intervals resample deal clusters within opponent. The "
        "conditional column uses the declared finite belief and is a secondary "
        "measure, not an observed win-rate gain.\n\n"
        + f"Opportunity statuses: {summary['status_counts']}.\n\n"
        + f"Adoptions: {summary['adoptions']}. Changed actual win scores: "
        f"{summary['changed_actual_win_scores']}.\n\n"
        + table(
            "Search costs under equal maximum allowances",
            [
                "Arm",
                "Physical decisions",
                "Planning seconds",
                "Parent calls",
                "Opponent calls",
                "Capped phases",
            ],
            [
                [
                    arm,
                    costs["physical_decisions"],
                    f"{costs['planning_seconds']:.3f}",
                    costs["parent_adapter_calls"],
                    costs["opponent_adapter_calls"],
                    costs["capped_phases"],
                ]
                for arm, costs in summary["arm_costs"].items()
            ],
        )
        + "Both arms had at most4M native transitions and1M live nodes per "
        "eligible root. Realized work differs. Caches started empty for each arm; "
        "four concurrent workers make elapsed timings operational costs, not "
        "isolated latency benchmarks.\n\n"
        + table(
            "Work accounting",
            ["Category", "Counts"],
            [[name, counts] for name, counts in summary["work"].items()],
        )
        + f"Additional export source replay: {dict(source_work)}. Additional "
        f"export tail replay: {dict(tail_work)}. These are verification work, "
        "not new matches.\n\n"
        + "Every adopted value pair was independently replayed with Python rules "
        "during the study. This export checks policy bindings, selection, stored "
        "oracle evidence, all actual traces and exact aggregate arithmetic. It "
        "does not rerun search or hypothetical neural inference. No additional "
        "model training, canonical-model promotion or reserved101M use occurred.\n\n"
        + f"Protocol SHA256: {summary['protocol_sha256']}.\n"
        + f"Summary SHA256: {sha(base / 'summary.json')}.\n"
    )
    output.write_text(text, encoding="utf-8", newline="\n")
    write(
        receipt,
        {
            "status": "passed",
            "report_sha256": sha(output),
            "summary_sha256": sha(base / "summary.json"),
            "bindings": bindings,
            "export_source_replay": dict(source_work),
            "export_tail_replay": dict(tail_work),
            "exporter_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/prospective_incumbent"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/PROSPECTIVE_INCUMBENT_RESULTS.rst")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("runs/prospective_incumbent_export.json")
    )
    args = parser.parse_args()
    export(args.base, args.output, args.receipt)
