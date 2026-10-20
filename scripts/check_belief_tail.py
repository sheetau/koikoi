"""Fixed natural-root capability/cost check for finite belief-tail planning."""

import argparse
import json
import os
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, PICK_DRAW, PLAYER, POINTS, ROUND, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.belief_tail import (
    FiniteBeliefPlanner,
    KoiTailDynamics,
    TailBudgetExceeded,
    enumerate_tail_worlds,
    visible_keys,
)
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_worlds import enumerate_support
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet, StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


def empty_work():
    return dict.fromkeys(("decisions", "round_endings", "completions"), 0)


def add_step(work, result):
    work["decisions"] += 1
    work["round_endings"] += int(result[1])
    work["completions"] += int(result[2])


def cases():
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "seat": seat,
            "replicate": replicate,
            "seed": 121110001 + 1000 * index + replicate,
        }
        for parent in (0, 1)
        for index, opponent in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in range(8)
    ]


def collect(case, parent, opponent, predictors, native):
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
            and draws <= 3
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


def verify_policy_values(worlds, plan, dynamics):
    """Replay every declared first-action policy with independent Python rules.

    Opponent actions are the frozen planner's recorded observation-keyed actions;
    this does not repeat neural inference. Actual-world assessment below uses
    fresh strict opponent calls after every plan has been frozen.
    """
    work = empty_work()
    mass = sum(map(int, worlds.weights))
    for first, policy in plan.root_policies.items():
        signed = 0
        for state, model, weight in zip(
            worlds.states, worlds.models, worlds.weights, strict=True
        ):
            env = reference(state)
            for _ in range(32):
                if env.state[PHASE] == TERMINAL:
                    break
                actor = int(env.state[PLAYER])
                key = visible_keys(env.state[None], actor)[0]
                action = (
                    policy[key]
                    if actor == worlds.viewer
                    else dynamics.policy_cache[(int(model), key)]
                )
                add_step(work, env.step(action))
            assert env.state[PHASE] == TERMINAL
            signed += int(weight) * int(
                np.sign(
                    env.state[POINTS + worlds.viewer]
                    - env.state[POINTS + 1 - worlds.viewer]
                )
            )
        assert Fraction(signed + mass, 2 * mass) == plan.values[first]
    return work


def assess_actual(source, plans, parent, opponent):
    result = {}
    for name in ("parent", "fixed", "optimized"):
        env = reference(source["selected_root"]["state"])
        actions, work, changes = [], empty_work(), 0
        while env.state[PHASE] != TERMINAL:
            if env.state[PLAYER] == source["seat"]:
                baseline = int(parent.act(env))
                action = (
                    baseline
                    if name == "parent"
                    else plans[name].policy[
                        visible_keys(env.state[None], source["seat"])[0]
                    ]
                )
                changes += int(action != baseline)
            else:
                action = int(opponent.act(env))
            add_step(work, env.step(action))
            actions.append(action)
        if name == "parent":
            assert actions == source["actions"][source["selected_root"]["decision"] :]
            np.testing.assert_array_equal(env.state, source["final"])
        result[name] = {
            "actions": actions,
            "work": work,
            "changed_own_decisions": changes,
            "final": env.state.tolist(),
            "point_margin": int(
                env.state[POINTS + source["seat"]]
                - env.state[POINTS + 1 - source["seat"]]
            ),
        }
    return result


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    files = [
        str(Path(__file__)),
        "scripts/belief_tail.py",
        "tests/test_belief_tail.py",
        "docs/BELIEF_TAIL.rst",
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
        "max_draws": 3,
        "max_hands": 100,
        "max_worlds": 100000,
        "max_nodes": 500000,
        "max_depth": 32,
        "scope": "Natural-root capability/cost and conditional-tail comparison; not a prospective search-policy strength study.",
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
            started = time.perf_counter()
            support = enumerate_support(
                root, {identity: opponents[identity]}, max_hands=protocol["max_hands"]
            )
            record.update(
                support_seconds=time.perf_counter() - started,
                support_counts=support.counts,
                public_model=identity,
            )
            plans, dynamics, planner = {}, None, None
            try:
                worlds = enumerate_tail_worlds(
                    root,
                    support,
                    max_draws=protocol["max_draws"],
                    max_worlds=protocol["max_worlds"],
                )
                dynamics = KoiTailDynamics(
                    len(worlds.states),
                    worlds.model_names,
                    {identity: opponents[identity]},
                    parents[source["parent"]],
                )
                record.update(worlds=len(worlds.states), draws=worlds.draws)
                for name, optimize in (("fixed", False), ("optimized", True)):
                    planner = FiniteBeliefPlanner(
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
                    record[name]["reference_policy_replay"] = verify_policy_values(
                        worlds, plans[name], dynamics
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
                    partial_work=dict(planner.work) if planner is not None else {},
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
    parser.add_argument("--output", default="runs/belief_tail_capability")
    run(Path(parser.parse_args().output))
