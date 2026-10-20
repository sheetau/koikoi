"""Run the capped incumbent controller on all five former full-comparison refusals."""

import argparse
import os
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.assess_incumbent_tail import assess
from scripts.belief_tail import KoiTailDynamics
from scripts.check_incumbent_execution import inputs_and_artifacts
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read
from scripts.history_conditioned_worlds import HistorySupport
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def preflight():
    sources, locations, inputs, identities = inputs_and_artifacts()
    saved = Path("runs/incumbent_saved_execution")
    complete, protocol = read(saved / "complete.json"), read(saved / "protocol.json")
    assert complete["status"] == read(saved / "process.json")["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("selections_sha256", "selections_complete.json"),
        ("assessments_sha256", "assessments.json"),
        ("report_sha256", "report.rst"),
    ):
        assert complete[key] == sha(saved / name)
        inputs[str(saved / name)] = complete[key]
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    inputs.update(protocol["inputs"])
    inputs[str(saved / "complete.json")] = sha(saved / "complete.json")
    selection_sha = read(saved / "selections_complete.json")["artifact_sha256"]
    assert sha(saved / "selections.pt") == selection_sha
    inputs[str(saved / "selections.pt")] = selection_sha
    assert identities == protocol["opponent_identities"]
    previous = read("runs/bounded_tail_recovery/plans_complete.json")["records"]
    assert [r["source_index"] for r in previous] == [0, 4, 20, 36, 52]
    return sources, locations, inputs, identities, previous


def describe(result):
    plan = result.plan
    return {
        "work": result.work,
        "refusal": result.refusal,
        "plan": None
        if plan is None
        else {
            "action": plan.action,
            "baseline_action": plan.baseline_action,
            "values": {str(a): str(q) for a, q in plan.values.items()},
            "action_upper_bounds": {
                str(a): str(q) for a, q in plan.action_upper_bounds.items()
            },
            "search_complete": plan.search_complete,
            "phase_optimality_gap": str(plan.optimality_gap),
        },
    }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sources, locations, inputs, identities, previous = preflight()
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "scripts/assess_incumbent_tail.py",
                "tests/test_assess_incumbent_tail.py",
                "docs/INCUMBENT_RECOVERY.rst",
                "scripts/reference_tail_policy.py",
            )
        }
    )
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "source_indices": [r["source_index"] for r in previous],
            "max_live_nodes": 500000,
            "max_decisions_per_phase": 2000000,
            "max_physical_decisions_per_root": 4000000,
            "max_depth": 32,
            "max_frontier_rows": 4096,
            "max_draws": 5,
            "max_hands": 1000,
            "cache": "Reuse bound observation-keyed action caches; count new calls separately.",
            "scope": "All five old refusals, old source cohort, explicit new optimization phases; exploratory.",
        },
    )

    def stage(name, index=None):
        write(
            output / "process.json",
            {"status": "running", "pid": os.getpid(), "stage": name, "source": index},
        )

    try:
        stage("setup")
        profiled = {
            r["source"]: r
            for r in read("runs/refused_tail_profile/complete.json")["records"]
        }
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        records, artifacts = [], {}
        for old in previous:
            index, source = old["source_index"], sources[old["source_index"]]
            path = locations[index][0]
            evidence = torch.load(path, map_location="cpu", weights_only=False)
            root = reference(source["selected_root"]["state"])
            worlds = deferred_worlds(
                root, HistorySupport(**evidence["support"]), max_draws=5, max_hands=1000
            )
            for key in ("states", "models", "weights"):
                np.testing.assert_array_equal(
                    getattr(worlds, key), getattr(evidence["worlds"], key)
                )
            dynamics = KoiTailDynamics(
                4096,
                worlds.model_names,
                {name: opponents[name] for name in worlds.model_names},
                parents[source["parent"]],
            )
            dynamics.baseline_cache = dict(evidence["baseline_action_cache"])
            dynamics.policy_cache = dict(evidence["opponent_action_cache"])
            row = {
                "source_index": index,
                "initial_cache_entries": {
                    "parent": len(dynamics.baseline_cache),
                    "opponent": len(dynamics.policy_cache),
                },
                "phases": {},
            }

            class TrackedPlanner(IncumbentBeliefPlanner):
                def solve(
                    self,
                    worlds,
                    optimize_continuation=True,
                    *,
                    index=index,
                    dynamics=dynamics,
                    row=row,
                    old=old,
                ):
                    mode = "optimized" if optimize_continuation else "fixed"
                    stage(mode, index)
                    before = (dynamics.baseline_calls, dynamics.policy_calls)
                    start = time.perf_counter()
                    result = super().solve(worlds, optimize_continuation)
                    row["phases"][mode] = {
                        **describe(result),
                        "seconds": time.perf_counter() - start,
                        "new_parent_calls": dynamics.baseline_calls - before[0],
                        "new_opponent_calls": dynamics.policy_calls - before[1],
                    }
                    if mode == "fixed" or old["partial_stage"] == "optimized":
                        expected_work = (
                            old[mode]["work"] if mode in old else old["partial_work"]
                        )
                        assert result.work == expected_work
                        assert before == (
                            dynamics.baseline_calls,
                            dynamics.policy_calls,
                        )
                    return result

            fixed, optimized, selection = evaluate_controller(
                worlds,
                dynamics,
                PARENTS[source["parent"]][1],
                {name: identities[name] for name in worlds.model_names},
                planner_type=TrackedPlanner,
            )
            assert fixed.result.plan is not None and optimized is not None
            parent_plan = fixed.result.plan
            first = profiled[index]["plans"]["fixed"]["root_actions"][0]
            assert first["status"] == "complete"
            assert parent_plan.baseline_action == first["action"]
            assert parent_plan.values[parent_plan.baseline_action] == Fraction(
                first["q"]
            )
            for phase in (fixed, optimized):
                if phase is None or phase.result.plan is None:
                    continue
                mode, plan = phase.mode, phase.result.plan
                stage(f"{mode}_verify", index)
                if mode in evidence["plans"]:
                    prior = evidence["plans"][mode]
                    for key in (
                        "action",
                        "baseline_action",
                        "values",
                        "policy",
                        "root_policies",
                        "action_upper_bounds",
                        "work",
                    ):
                        assert getattr(plan, key) == getattr(prior, key)
                    row["phases"][mode]["oracle"] = {
                        "reused_complete_policy_evidence": True
                    }
                else:
                    start = time.perf_counter()
                    row["phases"][mode]["oracle"] = verify_deferred_policy_values(
                        worlds, plan, dynamics.policy_cache
                    )
                    row["phases"][mode]["oracle_seconds"] = time.perf_counter() - start
            row["selection"] = {
                "mode": selection.mode,
                "reason": selection.reason,
                "parent_q": None
                if selection.parent_q is None
                else str(selection.parent_q),
                "selected_q": None
                if selection.selected_q is None
                else str(selection.selected_q),
                "gain": str(selection.gain or 0),
                "binding": selection.binding.digest,
            }
            row["final_cache_entries"] = {
                "parent": len(dynamics.baseline_cache),
                "opponent": len(dynamics.policy_cache),
            }
            target = output / f"root_{index:03}.pt"
            torch.save(
                {
                    "worlds": worlds,
                    "fixed": fixed,
                    "optimized": optimized,
                    "selection": selection,
                    "opponent_action_cache": dynamics.policy_cache,
                    "baseline_action_cache": dynamics.baseline_cache,
                },
                target,
            )
            artifacts[target.name] = sha(target)
            row["evidence_file"] = target.name
            records.append(row)
            write(output / "progress.json", records)
            print(
                f"source {index}: {selection.mode}, exact gain {selection.gain}",
                flush=True,
            )
        write(
            output / "selections_complete.json",
            {"records": records, "artifacts": artifacts},
        )
        stage("actual_tails")
        assessments, actual, replay, calls = {}, Counter(), Counter(), Counter()
        for row in records:
            index, source = row["source_index"], sources[row["source_index"]]
            artifact = torch.load(
                output / row["evidence_file"], map_location="cpu", weights_only=False
            )
            result, physical, audited, used_calls = assess(
                source,
                artifact["selection"],
                parents[source["parent"]],
                opponents[source["opponent"]],
                PARENTS[source["parent"]][1],
            )
            assessments[str(index)] = result
            actual.update(physical)
            replay.update(audited)
            calls.update(used_calls)
        assert all(sha(path) == digest for path, digest in inputs.items())
        assert all(sha(output / name) == digest for name, digest in artifacts.items())
        write(output / "assessments.json", assessments)
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "selections_sha256": sha(output / "selections_complete.json"),
                "assessments_sha256": sha(output / "assessments.json"),
                "actual_work": dict(actual),
                "replay_work": dict(replay),
                "actor_calls": dict(calls),
                "new_source_matches": 0,
                "new_history_decisions": 0,
            },
        )
        write(output / "process.json", {"status": "complete", "pid": os.getpid()})
    except BaseException as error:
        write(
            output / "process.json",
            {"status": "failed", "pid": os.getpid(), "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/incumbent_tail_recovery")
    run(Path(parser.parse_args().output))
