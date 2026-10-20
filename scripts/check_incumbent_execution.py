"""Freeze saved-plan incumbent selections, then assess all available actual tails."""

import argparse
import hashlib
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions, score
from scripts.export_full_source_results import table
from scripts.incumbent_belief_tail import IncumbentResult
from scripts.incumbent_tail_controller import (
    ContingentExecutor,
    EvaluatedPhase,
    bind_worlds,
    saved_phase,
    select_improvement,
)
from scripts.recover_bounded_tails import preflight
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import sha, write


class BindingView:
    keys = staticmethod(visible_keys)
    legal = staticmethod(KoiTailDynamics.legal)


def inputs_and_artifacts():
    sources, _, inputs = preflight()
    locations = {}
    for folder in ("earlier_belief_tail_capability", "ceiling_tail_recovery"):
        base = Path("runs") / folder
        closed = read(base / "plans_complete.json")
        for record in closed["records"]:
            if record["status"] == "complete":
                index = record["source_index"]
                world_path = base / record["evidence_file"]
                plan_path = Path(f"runs/bounded_tail_equivalence/root_{index:03}.pt")
                locations[index] = (world_path, plan_path, False)
    recovery = Path("runs/bounded_tail_recovery")
    complete, protocol = (
        read(recovery / "complete.json"),
        read(recovery / "protocol.json"),
    )
    assert complete["status"] == read(recovery / "process.json")["status"] == "complete"
    assert complete["protocol_sha256"] == sha(recovery / "protocol.json")
    assert complete["plans_sha256"] == sha(recovery / "plans_complete.json")
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    inputs.update(protocol["inputs"])
    for path in (recovery / "complete.json", recovery / "plans_complete.json"):
        inputs[str(path)] = sha(path)
    closed = read(recovery / "plans_complete.json")
    for record in closed["records"]:
        path = recovery / record["evidence_file"]
        assert sha(path) == closed["artifacts"][path.name]
        inputs[str(path)] = sha(path)
        locations[record["source_index"]] = (path, path, True)
    assert len(locations) == 33
    assert set(locations) == {
        i for i, s in enumerate(sources) if s["selected_root"] is not None
    }
    for path, digest in PARENTS:
        assert sha(path) == digest
    opponents = {}
    for name in ("wp", "point"):
        weights = {}
        for phase in ("discard", "pick", "koikoi"):
            matches = [
                v for p, v in inputs.items() if Path(p).name == f"{phase}_rl_{name}.pt"
            ]
            assert len(set(matches)) == 1
            weights[phase] = matches[0]
        weights["adapter"] = sha("koikoi/upstream.py")
        weights["guard"] = sha("koikoi/agents.py")
        opponents[name] = hashlib.sha256(
            json.dumps(weights, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return sources, locations, inputs, opponents


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sources, locations, inputs, identities = inputs_and_artifacts()
    inputs.update(
        {
            name: sha(name)
            for name in (
                __file__,
                "scripts/incumbent_tail_controller.py",
                "scripts/incumbent_belief_tail.py",
                "tests/test_incumbent_tail_controller.py",
                "docs/INCUMBENT_CONTROLLER.rst",
                "koikoi/upstream.py",
                "koikoi/agents.py",
            )
        }
    )
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "source_indices": list(range(64)),
            "available_indices": sorted(locations),
            "opponent_identities": identities,
            "scope": "All available saved plans; zero new source/history/search; exploratory execution audit.",
        },
    )

    def stage(name):
        write(
            output / "process.json",
            {"status": "running", "pid": os.getpid(), "stage": name},
        )

    try:
        stage("freeze_selections")
        records, selections = [], {}
        for index, source in enumerate(sources):
            if index not in locations:
                records.append(
                    {
                        "source": index,
                        "reason": "missing_root",
                        "mode": "parent",
                        "gain": "0",
                    }
                )
                continue
            world_path, plan_path, combined = locations[index]
            evidence = torch.load(world_path, map_location="cpu", weights_only=False)
            saved = (
                evidence["plans"]
                if combined
                else torch.load(plan_path, map_location="cpu", weights_only=False)
            )
            worlds = evidence["worlds"]
            binding = bind_worlds(
                worlds,
                BindingView(),
                PARENTS[source["parent"]][1],
                {name: identities[name] for name in worlds.model_names},
            )
            root = reference(source["selected_root"]["state"])
            assert binding.viewer == source["seat"]
            assert binding.root_key == visible_keys(root.state[None], source["seat"])[0]
            fixed = (
                saved_phase(binding, "fixed", saved["fixed"])
                if "fixed" in saved
                else EvaluatedPhase(
                    binding, "fixed", IncumbentResult(None, {}, "no_saved_parent_value")
                )
            )
            optimized = None
            if (
                fixed.result.plan is not None
                and fixed.result.plan.values[fixed.result.plan.action] < 1
                and "optimized" in saved
            ):
                optimized = saved_phase(binding, "optimized", saved["optimized"])
            selected = select_improvement(fixed, optimized)
            selections[index] = selected
            records.append(
                {
                    "source": index,
                    "reason": selected.reason,
                    "mode": selected.mode,
                    "parent_q": None
                    if selected.parent_q is None
                    else str(selected.parent_q),
                    "selected_q": None
                    if selected.selected_q is None
                    else str(selected.selected_q),
                    "gain": str(selected.gain or 0),
                    "binding": binding.digest,
                    "phase_optimality_gap": None
                    if selected.plan is None
                    else str(selected.plan.optimality_gap),
                    "world_artifact": str(world_path),
                    "plan_artifact": str(plan_path),
                }
            )
        target = output / "selections.pt"
        torch.save(selections, target)
        selection_sha = sha(target)
        write(
            output / "selections_complete.json",
            {"records": records, "artifact_sha256": selection_sha},
        )
        # Read the frozen artifact, rather than assess mutable planning objects.
        selections = torch.load(target, map_location="cpu", weights_only=False)
        stage("actual_tails")
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        assessments, physical, audit, calls = {}, Counter(), Counter(), Counter()
        for index, selected in selections.items():
            source = sources[index]
            parent, opponent = parents[source["parent"]], opponents[source["opponent"]]
            outcomes = {}
            for arm in ("parent", "controller"):
                env = reference(source["selected_root"]["state"])
                executor = ContingentExecutor(
                    selected, parent, PARENTS[source["parent"]][1]
                )
                actions, work, changed = [], empty_work(), 0
                for _ in range(32):
                    if env.state[PHASE] == TERMINAL:
                        break
                    if env.state[PLAYER] == source["seat"]:
                        if arm == "controller":
                            action = executor.act(env)
                            if selected.plan is None:
                                calls["parent_actor_calls"] += 1
                            else:
                                baseline = int(parent.act(env))
                                calls["parent_actor_calls"] += 1
                                changed += int(action != baseline)
                        else:
                            action = int(parent.act(env))
                            calls["parent_actor_calls"] += 1
                    else:
                        action = int(opponent.act(env))
                        calls["opponent_actor_calls"] += 1
                    actions.append(action)
                    add_step(work, env.step(action))
                assert env.state[PHASE] == TERMINAL
                outcomes[arm] = {
                    "actions": actions,
                    "work": work,
                    "changed_own_decisions": changed,
                    "final": env.state.tolist(),
                    "point_margin": int(
                        env.state[POINTS + source["seat"]]
                        - env.state[POINTS + 1 - source["seat"]]
                    ),
                }
                physical.update(work)
                replay = replay_actions(
                    reference(source["selected_root"]["state"]),
                    actions,
                    outcomes[arm]["final"],
                )
                assert replay == work
                audit.update(replay)
            assert (
                outcomes["parent"]["actions"]
                == source["actions"][source["selected_root"]["decision"] :]
            )
            assert outcomes["parent"]["final"] == source["final"]
            if selected.plan is None:
                assert outcomes["controller"] == outcomes["parent"]
            assessments[str(index)] = outcomes
        summary = {}
        for parent in (0, 1):
            indices = [i for i, s in enumerate(sources) if s["parent"] == parent]
            gains = [Fraction(records[i]["gain"]) for i in indices]
            realized = [
                score(assessments[str(i)]["controller"]["point_margin"])
                - score(assessments[str(i)]["parent"]["point_margin"])
                if str(i) in assessments
                else Fraction(0)
                for i in indices
            ]
            summary[str(parent + 1)] = {
                "sources": len(indices),
                "adoptions": sum(g > 0 for g in gains),
                "conditional_mean_gain": str(sum(gains) / len(indices)),
                "realized_mean_gain": str(sum(realized) / len(indices)),
                "realized_changed_outcomes": sum(g != 0 for g in realized),
            }
        assert sha(target) == selection_sha
        assert all(sha(path) == digest for path, digest in inputs.items())
        write(output / "assessments.json", assessments)
        text = (
            "Saved incumbent execution audit\n==============================\n\n"
            "All64 original sources /16 streams remain included:33 available\n"
            "roots and31 missing roots. Selections were frozen before all66\n"
            "parent/controller actual-tail completions. All parent suffixes\n"
            "reproduce their sources, and explicit parent fallbacks have identical\n"
            "controller traces. Every adopted policy executes through own/public\n"
            "keys with fresh strict opponents; every trace passes separate replay.\n\n"
        )
        text += table(
            "Means over32 sources per parent",
            [
                "Parent",
                "Adoptions",
                "Conditional gain",
                "Realized gain",
                "Changed outcomes",
            ],
            [
                [
                    p,
                    r["adoptions"],
                    r["conditional_mean_gain"],
                    r["realized_mean_gain"],
                    r["realized_changed_outcomes"],
                ]
                for p, r in summary.items()
            ],
        )
        text += table(
            "All source selections",
            ["Source", "Mode", "Reason", "Exact gain"],
            [[r["source"], r["mode"], r["reason"], r["gain"]] for r in records],
        )
        text += table(
            "New conditional continuation work",
            ["Category", "Counter", "Count"],
            [
                [category, k, v]
                for category, counters in (
                    ("actual", physical),
                    ("export_replay", audit),
                    ("actor_calls", calls),
                )
                for k, v in counters.items()
            ],
        )
        text += (
            "There are zero new search/history decisions, training updates or\n"
            "source matches. Existing independent Q evidence is reused through\n"
            "unchanged saved policy maps/values and bound artifacts. This audit\n"
            "does not execute the fresh two-phase search schedule or reconstruct\n"
            "unsaved partial incumbents. Conditional completions are not independent\n"
            "samples; no prospective strength replication or deployment is claimed.\n"
        )
        (output / "report.rst").write_bytes(text.encode("utf-8"))
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(output / "protocol.json"),
                "selections_sha256": sha(output / "selections_complete.json"),
                "assessments_sha256": sha(output / "assessments.json"),
                "report_sha256": sha(output / "report.rst"),
                "summary": summary,
                "actual_work": dict(physical),
                "replay_work": dict(audit),
                "actor_calls": dict(calls),
                "new_source_matches": 0,
                "new_search_decisions": 0,
                "new_history_decisions": 0,
            },
        )
        write(output / "process.json", {"status": "complete", "pid": os.getpid()})
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write(
            output / "process.json",
            {"status": "failed", "pid": os.getpid(), "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/incumbent_saved_execution")
    run(Path(parser.parse_args().output))
