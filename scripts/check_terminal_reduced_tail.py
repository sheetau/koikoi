"""Compare terminal aggregation with deferred planning on all completed five-draw roots."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.deferred_belief_tail import DeferredBeliefPlanner
from scripts.export_belief_tail import read
from scripts.run_arbitration_study import PARENTS
from scripts.terminal_reduced_belief_tail import TerminalReducedPlanner
from scripts.train_branch_policy_gradient import sha, write


def bounded_dynamics(worlds, opponents, parent, cap):
    dynamics = KoiTailDynamics(4096, worlds.model_names, opponents, parent)
    native = dynamics.advance
    used = {"decisions": 0}

    def advance(states, actions):
        if used["decisions"] + len(states) > cap:
            raise TailBudgetExceeded("Matched physical transition cap exceeded")
        after, work = native(states, actions)
        used["decisions"] += work["decisions"]
        return after, work

    dynamics.advance = advance
    return dynamics, used


def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh equivalence directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    complete = read(base / "complete.json")
    assert complete["status"] == read(base / "process.json")["status"] == "complete"
    for key, name in (
        ("protocol_sha256", "protocol.json"),
        ("sources_sha256", "sources.json"),
        ("plans_sha256", "plans_complete.json"),
        ("assessments_sha256", "assessments.json"),
    ):
        assert complete[key] == sha(base / name)
    previous = read(base / "protocol.json")
    for name, digest in previous["inputs"].items():
        assert sha(name) == digest
    closed = read(base / "plans_complete.json")
    for name, digest in closed["artifacts"].items():
        assert sha(base / name) == digest
    selected = [r for r in closed["records"] if r["status"] == "complete"]
    assert len(selected) == 18
    inputs = {
        name: sha(name)
        for name in (
            __file__,
            "scripts/terminal_reduced_belief_tail.py",
            "tests/test_terminal_reduced_belief_tail.py",
            "docs/TERMINAL_REDUCED_TAIL.rst",
        )
    }
    protocol = {
        "inputs": inputs,
        "previous_complete_sha256": sha(base / "complete.json"),
        "source_indices": [r["source_index"] for r in selected],
        "max_frontier_rows": 4096,
        "max_decisions_per_plan": 2000000,
        "max_live_nodes_reduced": 500000,
        "max_all_nodes_control": 2000000,
        "max_depth": 32,
        "order": "alternate by completed-root ordinal; fresh separate policy caches for each method; fixed then optimized within method",
        "scope": "Equivalence and paired cost on all completed closed five-draw roots; no new source or strength sample.",
    }
    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)
    write(output / "process.json", {"status": "running", "pid": os.getpid()})
    try:
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        sources = read(base / "sources.json")
        records = []
        for ordinal, record in enumerate(selected):
            index = record["source_index"]
            evidence = torch.load(
                base / record["evidence_file"], map_location="cpu", weights_only=False
            )
            worlds = evidence["worlds"]
            order = (
                ("deferred", "reduced") if ordinal % 2 == 0 else ("reduced", "deferred")
            )
            result = {"source_index": index, "order": order}
            for method in order:
                dynamics, used = bounded_dynamics(
                    worlds,
                    {name: opponents[name] for name in worlds.model_names},
                    parents[sources[index]["parent"]],
                    protocol["max_decisions_per_plan"],
                )
                result[method] = {}
                for name, optimize in (("fixed", False), ("optimized", True)):
                    used["decisions"] = 0
                    planner = (
                        DeferredBeliefPlanner(
                            dynamics,
                            protocol["max_all_nodes_control"],
                            protocol["max_depth"],
                        )
                        if method == "deferred"
                        else TerminalReducedPlanner(
                            dynamics,
                            protocol["max_live_nodes_reduced"],
                            protocol["max_depth"],
                            protocol["max_decisions_per_plan"],
                        )
                    )
                    write(
                        output / "process.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "source_index": index,
                            "method": method,
                            "stage": name,
                        },
                    )
                    start = time.perf_counter()
                    plan = planner.solve(worlds, optimize)
                    elapsed = time.perf_counter() - start
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
                    assert record[name]["reference_policy_replay"]["values"] == {
                        str(a): str(v) for a, v in plan.values.items()
                    }
                    for key in (
                        "decisions",
                        "round_endings",
                        "completions",
                        "represented_decisions",
                        "represented_round_endings",
                        "represented_completions",
                        "own_decision_nodes",
                        "chance_rows_added",
                        "maximum_frontier_rows",
                    ):
                        assert plan.work[key] == expected.work[key]
                    assert used["decisions"] == plan.work["decisions"]
                    result[method][name] = {"seconds": elapsed, "work": plan.work}
                result[method]["opponent_policy_calls"] = dynamics.policy_calls
                result[method]["baseline_policy_calls"] = dynamics.baseline_calls
            for key in ("opponent_policy_calls", "baseline_policy_calls"):
                assert result["deferred"][key] == result["reduced"][key]
            records.append(result)
            write(output / "progress.json", records)
            print(json.dumps({"verified_source": index}), flush=True)
        assert all(sha(name) == digest for name, digest in inputs.items())
        assert sha(base / "complete.json") == protocol["previous_complete_sha256"]
        totals = {}
        for method in ("deferred", "reduced"):
            rows = [r[method][name] for r in records for name in ("fixed", "optimized")]
            totals[method] = {
                "seconds": sum(r["seconds"] for r in rows),
                "maximum_plan_seconds": max(r["seconds"] for r in rows),
                "decisions": sum(r["work"]["decisions"] for r in rows),
                "completions": sum(r["work"]["completions"] for r in rows),
                "own_decision_nodes": sum(
                    r["work"]["own_decision_nodes"] for r in rows
                ),
            }
            key = (
                "observation_nodes"
                if method == "deferred"
                else "live_observation_nodes"
            )
            totals[method][key] = sum(r["work"][key] for r in rows)
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "records": records,
                "totals": totals,
                "new_source_matches": 0,
                "scope": protocol["scope"],
            },
        )
        write(output / "process.json", {"status": "complete", "pid": os.getpid()})
        print(json.dumps({"status": "complete", "totals": totals}), flush=True)
    except BaseException as error:
        write(
            output / "process.json",
            {"status": "failed", "pid": os.getpid(), "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="runs/earlier_belief_tail_capability")
    parser.add_argument("--output", default="runs/terminal_reduced_tail_equivalence")
    args = parser.parse_args()
    run(Path(args.base), Path(args.output))
