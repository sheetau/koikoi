"""Locate exhausted work in the five closed refusals without increasing limits."""

import argparse
import os
import time
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, TURN
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read
from scripts.history_conditioned_worlds import enumerate_support
from scripts.recover_ceiling_tails import prerequisites
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


class ProfiledPlanner(CeilingBeliefPlanner):
    """Passive root-action and successful physical-transition accounting."""

    def solve(self, worlds, optimize_continuation=True):
        self.root_states = worlds.states
        self.root_mass = sum(map(int, worlds.weights))
        self.root_actions = []
        self.phase_turn = Counter()
        self.active_root = None
        return super().solve(worlds, optimize_continuation)

    def transition(self, states, models, weights, actions):
        if states is self.root_states:
            self.active_root = {
                "action": int(actions[0]),
                "start_work": dict(self.work),
                "start_seconds": time.perf_counter(),
                "status": "running",
                "phase_turn_start": dict(self.phase_turn),
            }
            self.root_actions.append(self.active_root)
        return super().transition(states, models, weights, actions)

    def advance(self, states, actions):
        after = super().advance(states, actions)
        codes, counts = np.unique(
            states[:, TURN] * 10 + states[:, PHASE], return_counts=True
        )
        self.phase_turn.update(
            {int(k): int(v) for k, v in zip(codes, counts, strict=True)}
        )
        return after

    def continuation(self, states, models, weights, depth):
        result = super().continuation(states, models, weights, depth)
        if depth == 1:
            value, _ = result
            self.finish_root("complete")
            self.active_root["q"] = str(
                Fraction(value + self.root_mass, 2 * self.root_mass)
            )
        return result

    def finish_root(self, status):
        if self.active_root is None or self.active_root["status"] != "running":
            return
        row = self.active_root
        row["status"] = status
        row["seconds"] = time.perf_counter() - row.pop("start_seconds")
        before = row.pop("start_work")
        row["work"] = {k: v - before[k] for k, v in self.work.items()}
        # Frontier capacity is a maximum, not an additive work counter.
        row["work"].pop("maximum_frontier_rows")
        before = row.pop("phase_turn_start")
        row["phase_turn_decisions"] = {
            str(k): v - before.get(k, 0) for k, v in self.phase_turn.items()
        }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = Path("runs/earlier_belief_tail_capability")
    equivalence = Path("runs/ceiling_tail_equivalence")
    recovery = Path("runs/ceiling_tail_recovery")
    _, inputs = prerequisites(base, equivalence)
    complete, protocol = (
        read(recovery / "complete.json"),
        read(recovery / "protocol.json"),
    )
    assert complete["status"] == read(recovery / "process.json")["status"] == "complete"
    assert complete["protocol_sha256"] == sha(recovery / "protocol.json")
    assert complete["plans_sha256"] == sha(recovery / "plans_complete.json")
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    closed = read(recovery / "plans_complete.json")
    selected = [r for r in closed["records"] if r["status"] == "budget_refused"]
    assert [r["source_index"] for r in selected] == [0, 4, 20, 36, 52]
    inputs.update(protocol["inputs"])
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "docs/REFUSED_TAIL_PROFILE.rst",
                str(recovery / "complete.json"),
                str(recovery / "plans_complete.json"),
            )
        }
    )
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "sources_sha256": protocol["sources_sha256"],
            "source_indices": [r["source_index"] for r in selected],
            "caps": {k: v for k, v in protocol.items() if k.startswith("max_")},
            "scope": "Passive diagnostic replay of all five refusals; unchanged budgets; no new source or strength measurement.",
        },
    )

    def stage(name, index=None):
        write(
            output / "process.json",
            {"status": "running", "pid": os.getpid(), "stage": name, "source": index},
        )

    try:
        stage("setup")
        sources = read(base / "sources.json")
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        rows, artifacts = [], {}
        for previous in selected:
            index = previous["source_index"]
            source = sources[index]
            model = source["selected_root"]["public_identification"]["model"]
            root = reference(source["selected_root"]["state"])
            stage("history_support", index)
            support = enumerate_support(root, {model: opponents[model]}, max_hands=1000)
            worlds = deferred_worlds(root, support, max_draws=5, max_hands=1000)
            assert len(worlds.states) == previous["hypothesis_rows"]
            assert int(worlds.weights.sum()) == previous["represented_prefixes"]
            path = output / f"worlds_{index:03}.pt"
            torch.save({"worlds": worlds, "support": asdict(support)}, path)
            artifacts[path.name] = sha(path)
            dynamics = KoiTailDynamics(
                4096,
                worlds.model_names,
                {model: opponents[model]},
                parents[source["parent"]],
            )
            row = {"source": index, "support": asdict(support), "plans": {}}
            saved_plans = {}
            for name, optimize in (("fixed", False), ("optimized", True)):
                stage(name, index)
                planner = ProfiledPlanner(dynamics, 500000, 32, 2000000)
                plan = None
                try:
                    plan = planner.solve(worlds, optimize)
                    assert name in previous
                    assert plan.work == previous[name]["work"]
                    assert {str(a): str(q) for a, q in plan.values.items()} == previous[
                        name
                    ]["values"]
                    status = "complete"
                    saved_plans[name] = plan
                except TailBudgetExceeded as error:
                    planner.finish_root("budget_refused")
                    assert name == previous["partial_stage"]
                    assert str(error) == previous["reason"]
                    assert planner.work == previous["partial_work"]
                    status = "budget_refused"
                assert sum(planner.phase_turn.values()) == planner.work["decisions"]
                assert (
                    sum(r["work"]["decisions"] for r in planner.root_actions)
                    == planner.work["decisions"]
                )
                row["plans"][name] = {
                    "status": status,
                    "work": planner.work,
                    "phase_turn_decisions": dict(planner.phase_turn),
                    "root_actions": planner.root_actions,
                }
                if plan is None:
                    break
            assert dynamics.baseline_calls == previous["baseline_policy_calls"]
            assert dynamics.policy_calls == previous["opponent_policy_calls"]
            path = output / f"cache_{index:03}.pt"
            torch.save(
                {
                    "opponent_action_cache": dynamics.policy_cache,
                    "baseline_action_cache": dynamics.baseline_cache,
                    "complete_plans": saved_plans,
                },
                path,
            )
            artifacts[path.name] = sha(path)
            rows.append(row)
            write(output / "progress.json", rows)
            print(f"source {index}: exact refusal reproduced", flush=True)
        assert all(sha(path) == digest for path, digest in inputs.items())
        assert all(sha(output / name) == digest for name, digest in artifacts.items())
        assert sha(base / "sources.json") == protocol["sources_sha256"]
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "records": rows,
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
    parser.add_argument("--output", default="runs/refused_tail_profile")
    run(Path(parser.parse_args().output))
