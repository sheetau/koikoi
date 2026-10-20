from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.deferred_belief_tail import deferred_worlds
from scripts.export_prospective_incumbent import audit_arm, audit_source
from scripts.history_conditioned_worlds import enumerate_support
from scripts.prospective_incumbent import describe_selection, evaluate_arm
from scripts.run_arbitration_study import PARENTS
from tests.test_history_conditioned_worlds import Extremal, late_root


def test_arm_audit_rejects_changed_observation_or_choice():
    torch.set_num_threads(2)
    root, actor = late_root(), Extremal()
    root.state[ROUND] = 7
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer] = 200
    root.state[POINTS + 1 - viewer] = 30
    support = enumerate_support(root, {"fixture": actor})
    worlds = deferred_worlds(root, support)
    protocol = {
        "phase_nodes": 500000,
        "phase_decisions": 2000000,
        "depth": 32,
        "opponent_identities": {"fixture": "fixture"},
    }
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"fixture": actor}, actor)
    phases, selection, _ = evaluate_arm(
        worlds,
        dynamics,
        PARENTS[0][1],
        {"fixture": "fixture"},
        "controller",
        protocol,
    )
    evidence = {"worlds": worlds, "phases": phases, "selection": selection}
    row = {
        "selection": describe_selection(selection),
        "oracle": {},
        "phases": {
            name: {"work": phase.result.work, "refusal": phase.result.refusal}
            for name, phase in phases.items()
        },
    }
    source = {
        "parent": 0,
        "seat": viewer,
        "selected_root": {
            "state": root.state.tolist(),
            "public_identification": {"model": "fixture"},
        },
    }
    assert audit_arm(evidence, row, source, "controller", protocol) == selection
    changed = deepcopy(source)
    changed["selected_root"]["state"][POINTS + viewer] += 1
    with pytest.raises(AssertionError):
        audit_arm(evidence, row, changed, "controller", protocol)
    evidence["selection"] = replace(selection, reason="tampered")
    with pytest.raises(AssertionError):
        audit_arm(evidence, row, source, "controller", protocol)


def test_source_replay_checks_root_position_and_final_outcome():
    env, actor = ReferenceEnv(7, "paper8"), Extremal()
    actions, work, root = [], empty_work(), None
    while env.state[PHASE] != TERMINAL:
        if len(actions) == 4:
            root = {"state": env.state.tolist(), "decision": 4}
        action = int(actor.act(env))
        actions.append(action)
        add_step(work, env.step(action))
    source = {
        "seed": 7,
        "actions": actions,
        "work": work,
        "final": env.state.tolist(),
        "selected_root": root,
    }
    assert audit_source(source) == work
    source["selected_root"]["decision"] = 5
    with pytest.raises(AssertionError):
        audit_source(source)
