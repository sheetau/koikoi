"""Try the five profiled refusals with partial-action bounds and unchanged caps."""

import argparse
import os
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded, visible_keys
from scripts.bounded_belief_tail import BoundedBeliefPlanner
from scripts.check_belief_tail import assess_actual
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read
from scripts.history_conditioned_worlds import HistorySupport
from scripts.recover_ceiling_tails import prerequisites
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def preflight():
    base = Path("runs/earlier_belief_tail_capability")
    profile = Path("runs/refused_tail_profile")
    comparison = Path("runs/bounded_tail_equivalence")
    _, inputs = prerequisites(base, Path("runs/ceiling_tail_equivalence"))
    for directory in (profile, comparison):
        complete, protocol = (
            read(directory / "complete.json"),
            read(directory / "protocol.json"),
        )
        assert (
            complete["status"]
            == read(directory / "process.json")["status"]
            == "complete"
        )
        assert complete["protocol_sha256"] == sha(directory / "protocol.json")
        assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
        inputs.update(protocol["inputs"])
        inputs[str(directory / "complete.json")] = sha(directory / "complete.json")
        inputs[str(directory / "protocol.json")] = sha(directory / "protocol.json")
        for path, digest in complete["artifacts"].items():
            assert sha(directory / path) == digest
            inputs[str(directory / path)] = digest
    receipt = read("runs/bounded_tail_export.json")
    assert receipt["status"] == "complete"
    assert receipt["complete_sha256"] == sha(comparison / "complete.json")
    assert receipt["exporter_sha256"] == sha("scripts/export_bounded_tail.py")
    assert receipt["report_sha256"] == sha("docs/BOUNDED_TAIL_RESULTS.rst")
    inputs["runs/bounded_tail_export.json"] = sha("runs/bounded_tail_export.json")
    previous = read(profile / "complete.json")["records"]
    assert [r["source"] for r in previous] == [0, 4, 20, 36, 52]
    verified = read(comparison / "complete.json")["records"]
    assert len(verified) == len({r["source"] for r in verified}) == 28
    assert not {r["source"] for r in previous} & {r["source"] for r in verified}
    sources = read(base / "sources.json")
    assert len(sources) == 64
    inputs[str(base / "sources.json")] = sha(base / "sources.json")
    return sources, previous, inputs


def run(output):
    if output.exists():
        raise ValueError("Use a fresh recovery directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sources, previous, inputs = preflight()
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "docs/BOUNDED_RECOVERY.rst",
                "scripts/reference_tail_policy.py",
                "scripts/check_belief_tail.py",
                "scripts/deferred_belief_tail.py",
            )
        }
    )
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "source_indices": [r["source"] for r in previous],
            "max_live_nodes": 500000,
            "max_decisions_per_plan": 2000000,
            "max_frontier_rows": 4096,
            "max_depth": 32,
            "max_draws": 5,
            "max_hands": 1000,
            "history_support": "Reuse bound complete support; reconstruct worlds and require exact equality.",
            "scope": "All five refusals; no new source matches; exploratory additional computation.",
        },
    )

    def stage(name, index=None):
        write(
            output / "process.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "stage": name,
                "source": index,
            },
        )

    try:
        stage("setup")
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        records, artifacts = [], {}
        for old in previous:
            index, source = old["source"], sources[old["source"]]
            stage("world_reconstruction", index)
            saved = torch.load(
                f"runs/refused_tail_profile/worlds_{index:03}.pt",
                map_location="cpu",
                weights_only=False,
            )
            root = reference(source["selected_root"]["state"])
            model = source["selected_root"]["public_identification"]["model"]
            assert source["selected_root"]["public_identification"]["round"] < 7
            worlds = deferred_worlds(
                root,
                HistorySupport(**saved["support"]),
                max_draws=5,
                max_hands=1000,
            )
            for key in ("states", "models", "weights"):
                np.testing.assert_array_equal(
                    getattr(worlds, key), getattr(saved["worlds"], key)
                )
            assert worlds.model_names == (model,) == saved["worlds"].model_names
            assert worlds.viewer == source["seat"] == saved["worlds"].viewer
            assert set(visible_keys(worlds.states, worlds.viewer)) == {
                visible_keys(root.state[None], worlds.viewer)[0]
            }
            dynamics = KoiTailDynamics(
                4096,
                worlds.model_names,
                {model: opponents[model]},
                parents[source["parent"]],
            )
            record, plans = {"source_index": index, "reused_history_support": True}, {}
            for name, optimize in (("fixed", False), ("optimized", True)):
                stage(name, index)
                planner = BoundedBeliefPlanner(dynamics, 500000, 32, 2000000)
                start = time.perf_counter()
                try:
                    plan = planner.solve(worlds, optimize)
                except TailBudgetExceeded as error:
                    record.update(
                        status="budget_refused",
                        partial_stage=name,
                        reason=str(error),
                        partial_work=dict(planner.work),
                        partial_seconds=time.perf_counter() - start,
                    )
                    break
                plans[name] = plan
                if name in old["plans"]:
                    for prior in old["plans"][name]["root_actions"]:
                        if prior["status"] != "complete":
                            continue
                        action, value = prior["action"], Fraction(prior["q"])
                        if action in plan.values:
                            assert plan.values[action] == value
                        else:
                            assert value <= plan.action_upper_bounds[action]
                record[name] = {
                    "seconds": time.perf_counter() - start,
                    "action": plan.action,
                    "baseline_action": plan.baseline_action,
                    "values": {str(a): str(q) for a, q in plan.values.items()},
                    "action_upper_bounds": {
                        str(a): str(q) for a, q in plan.action_upper_bounds.items()
                    },
                    "work": plan.work,
                    "chosen_policy_nodes": len(plan.policy),
                }
                stage(f"{name}_reference_replay", index)
                start = time.perf_counter()
                record[name]["oracle"] = verify_deferred_policy_values(
                    worlds, plan, dynamics.policy_cache
                )
                record[name]["oracle_seconds"] = time.perf_counter() - start
            else:
                record["status"] = "complete"
                fixed, optimized = plans["fixed"], plans["optimized"]
                assert (
                    optimized.values[optimized.action]
                    >= fixed.values[fixed.action]
                    >= fixed.values[fixed.baseline_action]
                )
            path = output / f"root_{index:03}.pt"
            torch.save(
                {
                    "worlds": worlds,
                    "support": saved["support"],
                    "plans": plans,
                    "opponent_action_cache": dynamics.policy_cache,
                    "baseline_action_cache": dynamics.baseline_cache,
                },
                path,
            )
            artifacts[path.name] = sha(path)
            record.update(
                evidence_file=path.name,
                opponent_policy_calls=dynamics.policy_calls,
                baseline_policy_calls=dynamics.baseline_calls,
            )
            records.append(record)
            write(output / "progress.json", records)
            print(f"source {index}: {record['status']}", flush=True)
        write(
            output / "plans_complete.json", {"records": records, "artifacts": artifacts}
        )
        stage("actual_tail_assessment")
        assessments = {}
        for record in records:
            if record["status"] != "complete":
                continue
            index, source = record["source_index"], sources[record["source_index"]]
            evidence = torch.load(
                output / record["evidence_file"], map_location="cpu", weights_only=False
            )
            assessments[str(index)] = assess_actual(
                source,
                evidence["plans"],
                parents[source["parent"]],
                opponents[source["opponent"]],
            )
        assert all(sha(path) == digest for path, digest in inputs.items())
        assert all(sha(output / path) == digest for path, digest in artifacts.items())
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
                "new_history_replay_decisions": 0,
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
    parser.add_argument("--output", default="runs/bounded_tail_recovery")
    run(Path(parser.parse_args().output))
