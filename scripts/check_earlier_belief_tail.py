"""Fixed five-draw natural-root capability check with deferred chance expansion."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, PICK_DRAW, PLAYER, ROUND, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.belief_tail import (
    KoiTailDynamics,
    TailBudgetExceeded,
)
from scripts.check_belief_tail import add_step, assess_actual, empty_work
from scripts.deferred_belief_tail import DeferredBeliefPlanner, deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet, StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def cases():
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "seat": seat,
            "replicate": replicate,
            "seed": 122110001 + 1000 * index + replicate,
        }
        for parent in (0, 1)
        for index, opponent in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in range(8)
    ]


def collect(case, parent, opponent, predictors, native, max_draws=5):
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state)[None])
    identity = RevealedModelSet(
        json.dumps(case, sort_keys=True), case["seat"], predictors
    )
    actions, work, selected = [], empty_work(), None
    while env.state[PHASE] != TERMINAL:
        decision = len(actions)
        actor = int(env.state[PLAYER])
        known = identity.first_identification
        draws = 16 - int(env.state[TURN]) - int(env.state[PHASE] in (PICK_DRAW, KOI))
        if (
            selected is None
            and actor == case["seat"]
            and env.state[ROUND] == 7
            and draws <= max_draws
            and identity.status == "identified"
            and known is not None
            and known["round"] < 7
            and known["available_after_decision"] <= decision
        ):
            selected = {
                "decision": decision,
                "state": env.state.tolist(),
                "public_identification": dict(known),
            }
        snapshot = (
            public_snapshot(
                env.observe(), actor, decision, match_key=identity.match_key
            )
            if actor != case["seat"]
            else None
        )
        action = int((parent if actor == case["seat"] else opponent).act(env))
        result = env.step(action)
        other = native.step(torch.tensor([action]), freeze_done=True)
        np.testing.assert_array_equal(native.state[0].numpy(), env.state)
        np.testing.assert_array_equal(other[0].numpy(), result)
        if snapshot is not None:
            identity.observe(snapshot, action)
        actions.append(action)
        add_step(work, result)
    assert work["completions"] == 1
    return {
        **case,
        "actions": actions,
        "work": work,
        "final": env.state.tolist(),
        "selected_root": selected,
        "identification_events": identity.events,
        "identification_calls": identity.prediction_calls,
    }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    files = [
        str(Path(__file__)),
        "scripts/belief_tail.py",
        "scripts/check_belief_tail.py",
        "scripts/deferred_belief_tail.py",
        "scripts/reference_tail_policy.py",
        "tests/test_reference_tail_policy.py",
        "docs/EARLIER_BELIEF_TAIL.rst",
        "scripts/history_conditioned_worlds.py",
        "scripts/enumerate_history_support.py",
        "scripts/diagnose_history_support.py",
        "scripts/revealed_opponent_inputs.py",
        "scripts/revealed_opponent_models.py",
        "scripts/check_seat_feature_reconstruction.py",
    ]
    for path, digest in PARENTS:
        assert sha(path) == digest
        files.append(path)
    files += [
        str(UPSTREAM_PATH / f"model_agent/{phase}_rl_{kind}.pt")
        for phase in ("discard", "pick", "koikoi")
        for kind in ("wp", "point")
    ]
    inputs = {path: sha(path) for path in files}
    output.mkdir(parents=True)
    protocol = {
        "inputs": inputs,
        "source": provenance(),
        "cases": cases(),
        "source_matches": 64,
        "distinct_deal_streams": 16,
        "max_draws": 5,
        "max_hands": 1000,
        "max_frontier_rows": 4096,
        "max_nodes": 500000,
        "max_depth": 32,
        "scope": "Five-draw capability/cost and model-conditional tail comparison; not a prospective strength study.",
    }
    write(output / "protocol.json", protocol)
    try:
        write(
            output / "process.json",
            {"pid": os.getpid(), "status": "running", "stage": "source_collection"},
        )
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        assert all(parent.requires_endgame for parent in parents)
        published = {kind: PublishedAgent(kind) for kind in ("wp", "point")}
        opponents = {kind: EndgameAgent(agent) for kind, agent in published.items()}
        predictors = {
            kind: StrictPublishedPrediction(agent) for kind, agent in published.items()
        }
        native = BatchEnv(1, "cpu", debug=True)
        sources = []
        for index, case in enumerate(protocol["cases"]):
            sources.append(
                collect(
                    case,
                    parents[case["parent"]],
                    opponents[case["opponent"]],
                    predictors,
                    native,
                    max_draws=protocol["max_draws"],
                )
            )
            if (index + 1) % 8 == 0:
                print(json.dumps({"source_matches": index + 1}), flush=True)
        write(output / "sources.json", sources)
        write(
            output / "process.json",
            {"pid": os.getpid(), "status": "running", "stage": "planning"},
        )
        records, artifacts = [], {}
        for index, source in enumerate(sources):
            record = {"source_index": index, "status": "missing_root"}
            if source["selected_root"] is None:
                records.append(record)
                continue
            root = reference(source["selected_root"]["state"])
            identity = source["selected_root"]["public_identification"]["model"]
            write(
                output / "process.json",
                {
                    "pid": os.getpid(),
                    "status": "running",
                    "stage": "history_support",
                    "source_index": index,
                },
            )
            started = time.perf_counter()
            support = enumerate_support(
                root, {identity: opponents[identity]}, max_hands=protocol["max_hands"]
            )
            record.update(
                support_seconds=time.perf_counter() - started,
                support_counts=support.counts,
                candidate_hands=support.candidate_hands,
                public_model=identity,
            )
            plans, dynamics, planner, stage = {}, None, None, "world_setup"
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
                    {identity: opponents[identity]},
                    parents[source["parent"]],
                )
                record.update(
                    hypothesis_rows=len(worlds.states),
                    represented_prefixes=int(worlds.weights.sum()),
                    draws=worlds.draws,
                )
                for name, optimize in (("fixed", False), ("optimized", True)):
                    stage = name
                    write(
                        output / "process.json",
                        {
                            "pid": os.getpid(),
                            "status": "running",
                            "stage": name,
                            "source_index": index,
                        },
                    )
                    planner = DeferredBeliefPlanner(
                        dynamics, protocol["max_nodes"], protocol["max_depth"]
                    )
                    started = time.perf_counter()
                    plans[name] = planner.solve(worlds, optimize)
                    record[name] = {
                        "seconds": time.perf_counter() - started,
                        "action": plans[name].action,
                        "baseline_action": plans[name].baseline_action,
                        "values": {
                            str(a): str(q) for a, q in plans[name].values.items()
                        },
                        "work": plans[name].work,
                        "chosen_policy_nodes": len(plans[name].policy),
                    }
                    write(
                        output / "process.json",
                        {
                            "pid": os.getpid(),
                            "status": "running",
                            "stage": f"{name}_reference_replay",
                            "source_index": index,
                        },
                    )
                    oracle_started = time.perf_counter()
                    record[name]["reference_policy_replay"] = (
                        verify_deferred_policy_values(
                            worlds, plans[name], dynamics.policy_cache
                        )
                    )
                    record[name]["reference_replay_seconds"] = (
                        time.perf_counter() - oracle_started
                    )
                assert plans["fixed"].values.keys() == plans["optimized"].values.keys()
                assert all(
                    plans["optimized"].values[a] >= value
                    for a, value in plans["fixed"].values.items()
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
                record.update(
                    status="complete",
                    evidence_file=path.name,
                    opponent_policy_calls=dynamics.policy_calls,
                    baseline_policy_calls=dynamics.baseline_calls,
                )
            except TailBudgetExceeded as error:
                record.update(
                    status="budget_refused",
                    reason=str(error),
                    partial_stage=stage,
                    partial_work=dict(planner.work) if planner is not None else {},
                    partial_stage_seconds=time.perf_counter() - started,
                    opponent_policy_calls=dynamics.policy_calls if dynamics else 0,
                    baseline_policy_calls=dynamics.baseline_calls if dynamics else 0,
                )
            records.append(record)
            write(output / "planning_progress.json", records)
            print(
                json.dumps({"planned_source": index, "status": record["status"]}),
                flush=True,
            )
        write(
            output / "plans_complete.json",
            {
                "records": records,
                "artifacts": artifacts,
                "sources_sha256": sha(output / "sources.json"),
            },
        )
        write(
            output / "process.json",
            {
                "pid": os.getpid(),
                "status": "running",
                "stage": "actual_tail_assessment",
            },
        )
        assessments = {}
        for record in records:
            if record["status"] != "complete":
                continue
            index = record["source_index"]
            source = sources[index]
            evidence = torch.load(
                output / record["evidence_file"], map_location="cpu", weights_only=False
            )
            assert source["opponent"] == record["public_model"]
            assessments[str(index)] = assess_actual(
                source,
                evidence["plans"],
                parents[source["parent"]],
                opponents[source["opponent"]],
            )
        assert all(sha(path) == digest for path, digest in inputs.items())
        assert provenance()["source_sha256"] == protocol["source"]["source_sha256"]
        assert all(sha(output / name) == digest for name, digest in artifacts.items())
        write(output / "assessments.json", assessments)
        source_work = {
            key: sum(source["work"][key] for source in sources) for key in empty_work()
        }
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "sources_sha256": sha(output / "sources.json"),
                "plans_sha256": sha(output / "plans_complete.json"),
                "assessments_sha256": sha(output / "assessments.json"),
                "source_work": source_work,
                "statuses": {
                    status: sum(r["status"] == status for r in records)
                    for status in ("missing_root", "complete", "budget_refused")
                },
                "scope": protocol["scope"],
            },
        )
        write(output / "process.json", {"pid": os.getpid(), "status": "complete"})
    except BaseException as error:
        write(
            output / "process.json",
            {"pid": os.getpid(), "status": "failed", "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/earlier_belief_tail_capability")
    run(Path(parser.parse_args().output))
