"""Paired exactness and cost check on all 28 saved completed roots."""

import argparse
import os
import time
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics
from scripts.bounded_belief_tail import BoundedBeliefPlanner
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.export_belief_tail import read
from scripts.recover_ceiling_tails import prerequisites
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def run(output):
    if output.exists():
        raise ValueError("Use a fresh comparison directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = Path("runs/earlier_belief_tail_capability")
    recovery = Path("runs/ceiling_tail_recovery")
    _, inputs = prerequisites(base, Path("runs/ceiling_tail_equivalence"))
    receipt = read("runs/ceiling_tail_recovery_export.json")
    assert receipt["status"] == "complete"
    assert receipt["recovery_complete_sha256"] == sha(recovery / "complete.json")
    assert receipt["exporter_sha256"] == sha("scripts/export_ceiling_recovery.py")
    assert receipt["report_sha256"] == sha("docs/CEILING_RECOVERY_RESULTS.rst")
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "scripts/bounded_belief_tail.py",
                "tests/test_bounded_belief_tail.py",
                "docs/BOUNDED_BELIEF_TAIL.rst",
                "runs/ceiling_tail_recovery_export.json",
            )
        }
    )
    selected = []
    for directory in (base, recovery):
        complete = read(directory / "complete.json")
        protocol = read(directory / "protocol.json")
        assert complete["status"] == "complete"
        assert complete["protocol_sha256"] == sha(directory / "protocol.json")
        assert complete["plans_sha256"] == sha(directory / "plans_complete.json")
        inputs.update(protocol["inputs"])
        for path in (directory / "complete.json", directory / "plans_complete.json"):
            inputs[str(path)] = sha(path)
        closed = read(directory / "plans_complete.json")
        for record in closed["records"]:
            if record["status"] != "complete":
                continue
            path = directory / record["evidence_file"]
            assert sha(path) == closed["artifacts"][record["evidence_file"]]
            inputs[str(path)] = sha(path)
            selected.append((record["source_index"], path))
    selected.sort()
    assert len(selected) == len({i for i, _ in selected}) == 28
    assert all(sha(path) == digest for path, digest in inputs.items())
    sources = read(base / "sources.json")
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "source_indices": [i for i, _ in selected],
            "order": "alternate by sorted source index; separate fresh caches; fixed then optimized",
            "max_live_nodes": 500000,
            "max_decisions_per_plan": 2000000,
            "max_frontier_rows": 4096,
            "max_depth": 32,
            "scope": "Exactness and computational comparison only; zero new source matches.",
        },
    )
    try:
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        records, artifacts = [], {}
        for ordinal, (index, path) in enumerate(selected):
            evidence = torch.load(path, map_location="cpu", weights_only=False)
            worlds = evidence["worlds"]
            order = (
                ("ceiling", "bounded") if ordinal % 2 == 0 else ("bounded", "ceiling")
            )
            record, plans = {"source": index, "order": order}, {}
            for method in order:
                dynamics = KoiTailDynamics(
                    4096,
                    worlds.model_names,
                    {name: opponents[name] for name in worlds.model_names},
                    parents[sources[index]["parent"]],
                )
                planner_type = (
                    CeilingBeliefPlanner
                    if method == "ceiling"
                    else BoundedBeliefPlanner
                )
                record[method], plans[method] = {}, {}
                for name, optimize in (("fixed", False), ("optimized", True)):
                    write(
                        output / "process.json",
                        {
                            "status": "running",
                            "pid": os.getpid(),
                            "source": index,
                            "method": method,
                            "stage": name,
                        },
                    )
                    start = time.perf_counter()
                    plan = planner_type(dynamics, 500000, 32, 2000000).solve(
                        worlds, optimize
                    )
                    elapsed = time.perf_counter() - start
                    expected = evidence["plans"][name]
                    assert (
                        plan.action,
                        plan.baseline_action,
                        plan.values[plan.action],
                        plan.policy,
                    ) == (
                        expected.action,
                        expected.baseline_action,
                        expected.values[expected.action],
                        expected.policy,
                    )
                    assert all(
                        q == expected.values[a]
                        and plan.root_policies[a] == expected.root_policies[a]
                        for a, q in plan.values.items()
                    )
                    plans[method][name] = plan
                    record[method][name] = {
                        "seconds": elapsed,
                        "work": plan.work,
                        "action": plan.action,
                        "values": {str(a): str(q) for a, q in plan.values.items()},
                        "baseline_calls": dynamics.baseline_calls,
                        "opponent_calls": dynamics.policy_calls,
                    }
                    if method == "bounded":
                        record[method][name]["action_upper_bounds"] = {
                            str(a): str(q) for a, q in plan.action_upper_bounds.items()
                        }
            for name in ("fixed", "optimized"):
                control, bounded = plans["ceiling"][name], plans["bounded"][name]
                assert bounded.work["decisions"] <= control.work["decisions"]
                for action, upper in bounded.action_upper_bounds.items():
                    if action in control.values:
                        assert (
                            control.values[action]
                            <= upper
                            <= bounded.values[bounded.action]
                        )
                    else:
                        assert action in control.unevaluated_actions
                        assert upper == bounded.values[bounded.action] == 1
                for key in ("baseline_calls", "opponent_calls"):
                    assert record["bounded"][name][key] <= record["ceiling"][name][key]
            target = output / f"root_{index:03}.pt"
            torch.save(plans["bounded"], target)
            artifacts[target.name] = sha(target)
            records.append(record)
            write(output / "progress.json", records)
            print(
                f"source {index}: bounded values/policies/certificates agree",
                flush=True,
            )
        assert all(sha(path) == digest for path, digest in inputs.items())
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "records": records,
                "artifacts": artifacts,
                "new_source_matches": 0,
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
    parser.add_argument("--output", default="runs/bounded_tail_equivalence")
    run(Path(parser.parse_args().output))
