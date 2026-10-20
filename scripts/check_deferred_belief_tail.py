"""Check delayed chance expansion against every closed eager capability plan."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics
from scripts.deferred_belief_tail import DeferredBeliefPlanner, deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, sum_work
from scripts.history_conditioned_worlds import HistorySupport
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh verification directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    old = read(base / "complete.json")
    assert old["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("sources_sha256", "sources.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert old[key] == sha(base / name)
    previous = read(base / "protocol.json")
    for name, digest in previous["inputs"].items():
        assert sha(name) == digest
    closed = read(base / "plans_complete.json")
    for name, digest in closed["artifacts"].items():
        assert sha(base / name) == digest
    inputs = {
        name: sha(name)
        for name in (
            __file__,
            "scripts/deferred_belief_tail.py",
            "tests/test_deferred_belief_tail.py",
            "docs/DEFERRED_BELIEF_TAIL.rst",
        )
    }
    output.mkdir(parents=True)
    protocol = {
        "inputs": inputs,
        "eager_complete_sha256": sha(base / "complete.json"),
        "eager_artifacts": closed["artifacts"],
        "max_nodes": previous["max_nodes"],
        "max_depth": previous["max_depth"],
        "max_draws": previous["max_draws"],
        "max_frontier_rows": 4096,
        "scope": "All closed roots; numerical and executable-policy equivalence. No new source matches or strength inference.",
    }
    write(output / "protocol.json", protocol)
    write(output / "process.json", {"status": "running", "pid": os.getpid()})
    try:
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        sources = read(base / "sources.json")
        records = []
        for record in closed["records"]:
            if record["status"] != "complete":
                continue
            index = record["source_index"]
            source = sources[index]
            evidence = torch.load(
                base / record["evidence_file"], map_location="cpu", weights_only=False
            )
            root = reference(source["selected_root"]["state"])
            started = time.perf_counter()
            worlds = deferred_worlds(
                root,
                HistorySupport(**evidence["support"]),
                max_draws=protocol["max_draws"],
            )
            setup_seconds = time.perf_counter() - started
            assert worlds.weights.sum() == evidence["worlds"].weights.sum()
            dynamics = KoiTailDynamics(
                protocol["max_frontier_rows"],
                worlds.model_names,
                {name: opponents[name] for name in worlds.model_names},
                parents[source["parent"]],
            )
            native = dynamics.advance
            oracle = {"decisions": 0, "round_endings": 0, "completions": 0}
            oracle_seconds = 0.0

            def checked(states, actions, native=native, oracle=oracle):
                nonlocal oracle_seconds
                after, work = native(states, actions)
                start_check = time.perf_counter()
                expected = {
                    "decisions": len(states),
                    "round_endings": 0,
                    "completions": 0,
                }
                for before, action, actual in zip(states, actions, after, strict=True):
                    env = reference(before)
                    result = env.step(int(action))
                    np.testing.assert_array_equal(actual, env.state)
                    expected["round_endings"] += int(result[1])
                    expected["completions"] += int(result[2])
                assert work == expected
                for key, value in expected.items():
                    oracle[key] += value
                oracle_seconds += time.perf_counter() - start_check
                return after, work

            dynamics.advance = checked
            result = {
                "source_index": index,
                "initial_rows": len(worlds.states),
                "represented_prefixes": int(worlds.weights.sum()),
                "setup_seconds": setup_seconds,
            }
            for name, optimize in (("fixed", False), ("optimized", True)):
                started = time.perf_counter()
                plan = DeferredBeliefPlanner(
                    dynamics, protocol["max_nodes"], protocol["max_depth"]
                ).solve(worlds, optimize)
                seconds = time.perf_counter() - started
                expected = evidence["plans"][name]
                assert (
                    plan.action,
                    plan.baseline_action,
                    plan.values,
                    plan.root_policies,
                ) == (
                    expected.action,
                    expected.baseline_action,
                    expected.values,
                    expected.root_policies,
                )
                for key in ("decisions", "round_endings", "completions"):
                    assert plan.work[f"represented_{key}"] == expected.work[key]
                for key in ("observation_nodes", "own_decision_nodes"):
                    assert plan.work[key] == expected.work[key]
                result[name] = {"work": plan.work, "seconds_including_oracle": seconds}
            result.update(
                oracle_work=oracle,
                oracle_seconds=oracle_seconds,
                opponent_policy_calls=dynamics.policy_calls,
                baseline_policy_calls=dynamics.baseline_calls,
            )
            assert (
                oracle["decisions"]
                == result["fixed"]["work"]["decisions"]
                + result["optimized"]["work"]["decisions"]
            )
            records.append(result)
            write(output / "progress.json", records)
            print(
                json.dumps(
                    {
                        "verified_source": index,
                        "rows": len(worlds.states),
                        "decisions": oracle["decisions"],
                    }
                ),
                flush=True,
            )
        assert len(records) == 16
        assert all(sha(path) == digest for path, digest in inputs.items())
        work = sum_work(
            r[name]["work"] for r in records for name in ("fixed", "optimized")
        )
        # A maximum is not additive across roots/plans.
        work["maximum_frontier_rows"] = max(
            r[name]["work"]["maximum_frontier_rows"]
            for r in records
            for name in ("fixed", "optimized")
        )
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "records": records,
                "work": work,
                "oracle_work": sum_work(r["oracle_work"] for r in records),
                "new_source_matches": 0,
                "scope": protocol["scope"],
            },
        )
        write(output / "process.json", {"status": "complete", "pid": os.getpid()})
        print(json.dumps({"status": "complete", "work": work}), flush=True)
    except BaseException as error:
        write(
            output / "process.json",
            {"status": "failed", "pid": os.getpid(), "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/belief_tail_capability")
    parser.add_argument("--output", default="runs/deferred_belief_tail_equivalence")
    args = parser.parse_args()
    run(Path(args.base), Path(args.output))
