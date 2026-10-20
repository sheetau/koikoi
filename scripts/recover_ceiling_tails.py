"""Attempt all fifteen old five-draw refusals after ceiling equivalence passes."""

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.check_belief_tail import assess_actual
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def prerequisites(base, equivalence):
    complete, original = read(base / "complete.json"), read(base / "protocol.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("sources_sha256", "sources.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[key] == sha(base / name)
    closed = read(base / "plans_complete.json")
    for name, digest in closed["artifacts"].items():
        assert sha(base / name) == digest
    verified = read(equivalence / "complete.json")
    gate = read(equivalence / "protocol.json")
    assert (
        verified["status"] == read(equivalence / "process.json")["status"] == "complete"
    )
    assert verified["protocol_sha256"] == sha(equivalence / "protocol.json")
    assert gate["previous_complete_sha256"] == sha(base / "complete.json")
    expected = [
        r["source_index"] for r in closed["records"] if r["status"] == "complete"
    ]
    assert len(expected) == 18
    assert (
        gate["source_indices"]
        == [r["source_index"] for r in verified["records"]]
        == expected
    )
    inputs = {**original["inputs"], **gate["inputs"]}
    assert all(sha(name) == digest for name, digest in inputs.items())
    assert provenance()["source_sha256"] == original["source"]["source_sha256"]
    refused = [
        r["source_index"] for r in closed["records"] if r["status"] == "budget_refused"
    ]
    assert len(refused) == 15
    return refused, inputs


def run(base, equivalence, output):
    if output.exists():
        raise ValueError("Use a fresh recovery directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    indices, inputs = prerequisites(base, equivalence)
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "docs/CEILING_RECOVERY.rst",
                "scripts/check_belief_tail.py",
                "scripts/history_conditioned_worlds.py",
                "scripts/deferred_belief_tail.py",
            )
        }
    )
    protocol = {
        "inputs": inputs,
        "original_complete_sha256": sha(base / "complete.json"),
        "equivalence_complete_sha256": sha(equivalence / "complete.json"),
        "sources_sha256": sha(base / "sources.json"),
        "source_sha256": provenance()["source_sha256"],
        "source_indices": indices,
        "max_draws": 5,
        "max_hands": 1000,
        "max_frontier_rows": 4096,
        "max_live_nodes": 500000,
        "max_decisions_per_plan": 2000000,
        "max_depth": 32,
        "new_source_matches": 0,
        "scope": "All prior refusals, after observing computational limits; exploratory capability, not a new prospective strength sample.",
    }
    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)

    def stage(name, index=None):
        write(
            output / "process.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "stage": name,
                "source_index": index,
            },
        )

    try:
        stage("setup")
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        sources = read(base / "sources.json")
        records, artifacts = [], {}
        for index in indices:
            source = sources[index]
            root = reference(source["selected_root"]["state"])
            model = source["selected_root"]["public_identification"]["model"]
            assert source["selected_root"]["public_identification"]["round"] < 7
            stage("history_support", index)
            start = time.perf_counter()
            support = enumerate_support(
                root, {model: opponents[model]}, max_hands=protocol["max_hands"]
            )
            record = {
                "source_index": index,
                "public_model": model,
                "support_seconds": time.perf_counter() - start,
                "candidate_hands": support.candidate_hands,
                "support_counts": support.counts,
            }
            plans, planner, dynamics = {}, None, None
            current_stage, start = "world_setup", time.perf_counter()
            try:
                if support.status == "too_many_hands":
                    raise TailBudgetExceeded("Public candidate-hand cap exceeded")
                worlds = deferred_worlds(
                    root,
                    support,
                    max_draws=protocol["max_draws"],
                    max_hands=protocol["max_hands"],
                )
                dynamics = KoiTailDynamics(
                    protocol["max_frontier_rows"],
                    worlds.model_names,
                    {model: opponents[model]},
                    parents[source["parent"]],
                )
                record.update(
                    hypothesis_rows=len(worlds.states),
                    represented_prefixes=int(worlds.weights.sum()),
                    draws=worlds.draws,
                )
                legal = set(map(int, np.flatnonzero(root.legal())))
                for name, optimize in (("fixed", False), ("optimized", True)):
                    current_stage = name
                    stage(name, index)
                    planner = CeilingBeliefPlanner(
                        dynamics,
                        protocol["max_live_nodes"],
                        protocol["max_depth"],
                        protocol["max_decisions_per_plan"],
                    )
                    start = time.perf_counter()
                    plan = plans[name] = planner.solve(worlds, optimize)
                    assert set(plan.values).isdisjoint(plan.unevaluated_actions)
                    assert set(plan.values) | set(plan.unevaluated_actions) == legal
                    assert not plan.unevaluated_actions or (
                        plan.root_ceiling_attained and plan.values[plan.action] == 1
                    )
                    record[name] = {
                        "seconds": time.perf_counter() - start,
                        "action": plan.action,
                        "baseline_action": plan.baseline_action,
                        "values": {str(a): str(v) for a, v in plan.values.items()},
                        "unevaluated_actions": list(plan.unevaluated_actions),
                        "root_ceiling_attained": plan.root_ceiling_attained,
                        "work": plan.work,
                        "chosen_policy_nodes": len(plan.policy),
                    }
                    stage(f"{name}_reference_replay", index)
                    oracle_start = time.perf_counter()
                    record[name]["oracle"] = verify_deferred_policy_values(
                        worlds, plan, dynamics.policy_cache
                    )
                    record[name]["oracle_seconds"] = time.perf_counter() - oracle_start
                fixed, optimized = plans["fixed"], plans["optimized"]
                assert (
                    optimized.values[optimized.action]
                    >= fixed.values[fixed.action]
                    >= fixed.values[fixed.baseline_action]
                )
                assert all(
                    optimized.values[a] >= fixed.values[a]
                    for a in fixed.values.keys() & optimized.values.keys()
                )
                path = output / f"root_{index:03}.pt"
                torch.save(
                    {
                        "worlds": worlds,
                        "support": asdict(support),
                        "plans": plans,
                        "opponent_action_cache": dynamics.policy_cache,
                    },
                    path,
                )
                artifacts[path.name] = sha(path)
                record.update(status="complete", evidence_file=path.name)
            except TailBudgetExceeded as error:
                record.update(
                    status="budget_refused",
                    reason=str(error),
                    partial_stage=current_stage,
                    partial_work=dict(planner.work) if planner is not None else {},
                    partial_stage_seconds=time.perf_counter() - start,
                )
            record.update(
                opponent_policy_calls=dynamics.policy_calls if dynamics else 0,
                baseline_policy_calls=dynamics.baseline_calls if dynamics else 0,
            )
            records.append(record)
            write(output / "progress.json", records)
            print(f"source {index}: {record['status']}", flush=True)
        write(
            output / "plans_complete.json",
            {
                "records": records,
                "artifacts": artifacts,
                "sources_sha256": protocol["sources_sha256"],
            },
        )
        stage("actual_tail_assessment")
        assessments = {}
        for record in records:
            if record["status"] != "complete":
                continue
            index = record["source_index"]
            source = sources[index]
            assert source["opponent"] == record["public_model"]
            evidence = torch.load(
                output / record["evidence_file"], map_location="cpu", weights_only=False
            )
            assessments[str(index)] = assess_actual(
                source,
                evidence["plans"],
                parents[source["parent"]],
                opponents[record["public_model"]],
            )
        assert all(sha(name) == digest for name, digest in inputs.items())
        assert all(sha(output / name) == digest for name, digest in artifacts.items())
        assert sha(base / "sources.json") == protocol["sources_sha256"]
        assert sha(base / "complete.json") == protocol["original_complete_sha256"]
        assert (
            sha(equivalence / "complete.json")
            == protocol["equivalence_complete_sha256"]
        )
        assert provenance()["source_sha256"] == protocol["source_sha256"]
        write(output / "assessments.json", assessments)
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "plans_sha256": sha(output / "plans_complete.json"),
                "assessments_sha256": sha(output / "assessments.json"),
                "statuses": {
                    name: sum(r["status"] == name for r in records)
                    for name in ("complete", "budget_refused")
                },
                "new_source_matches": 0,
                "scope": protocol["scope"],
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
    parser.add_argument("--base", default="runs/earlier_belief_tail_capability")
    parser.add_argument("--equivalence", default="runs/ceiling_tail_equivalence")
    parser.add_argument("--output", default="runs/ceiling_tail_recovery")
    args = parser.parse_args()
    run(Path(args.base), Path(args.equivalence), Path(args.output))
