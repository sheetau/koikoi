from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.deferred_belief_tail import deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.prospective_incumbent import (
    assess_three,
    cases,
    contrasts,
    evaluate_arm,
    verify_adoption,
)
from scripts.run_arbitration_study import sha, write
from tests.test_history_conditioned_worlds import Extremal, late_root
from tests.test_incumbent_belief_tail import IncumbentToy
from tests.test_incumbent_tail_controller import toy_setup


def test_fresh_complete_factorial_and_deal_clusters():
    design = cases()
    assert len(design) == 512
    assert len({c["seed"] for c in design}) == 128
    assert min(c["seed"] for c in design) == 123110001
    assert max(c["seed"] for c in design) == 123111064
    assert len({tuple(c.values()) for c in design}) == 512
    rows = []
    for case in design:
        final = [0] * 240
        # Both seats are wins, regardless of the player-indexed point slots.
        final[POINTS + case["seat"]] = 40
        final[POINTS + 1 - case["seat"]] = 20
        row = {"source": {**case, "final": final}, "status": "missing_root"}
        if case["parent"] == 0 and case["seat"] == 0:
            row.update(
                status="evaluated",
                assessment={
                    "outcomes": {
                        "parent": {"margin": 20},
                        "root_only": {"margin": 0},
                        "controller": {"margin": -20},
                    }
                },
                arms={
                    "controller": {"selection": {"gain": "1/5"}},
                    "root_only": {"selection": {"gain": "1/10"}},
                },
            )
        rows.append(row)
    keys, actual, conditional = contrasts(rows)
    assert len(keys) == 128
    np.testing.assert_array_equal(actual, np.tile([-0.5, -0.25, 0, 0], (128, 1)))
    np.testing.assert_allclose(conditional, np.tile([0.1, 0.05, 0, 0], (128, 1)))


def test_equal_total_allowance_keeps_partial_incumbents():
    protocol = {"phase_nodes": 500000, "phase_decisions": 8, "depth": 32}
    total = {}
    selections = {}
    for arm in ("controller", "root_only"):
        dynamics, worlds = toy_setup()
        phases, selection, timing = evaluate_arm(
            worlds,
            dynamics,
            "parent",
            {"unused": "opponent"},
            arm,
            protocol,
            planner_type=IncumbentToy,
        )
        assert set(timing) == set(phases)
        assert selection.parent_q == Fraction(1, 4)
        assert selection.selected_q == Fraction(3, 4)
        total[arm] = sum(p.result.work["decisions"] for p in phases.values())
        assert total[arm] <= 2 * protocol["phase_decisions"]
        selections[arm] = selection
    assert total["controller"] == 16
    assert total["root_only"] < total["controller"]
    assert phases["fixed"].result.plan.search_complete
    assert selections["controller"].binding == selections["root_only"].binding


def test_native_three_arm_execution_and_independent_value_checks():
    torch.set_num_threads(2)
    root, actor = late_root(), Extremal()
    root.state[ROUND] = 7
    viewer = int(root.state[PLAYER])
    root.state[POINTS + viewer] = 200
    root.state[POINTS + 1 - viewer] = 30
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support)
    protocol = {"phase_nodes": 500000, "phase_decisions": 2000000, "depth": 32}
    selections = {}
    for arm in ("controller", "root_only"):
        dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
        phases, selection, _ = evaluate_arm(
            worlds,
            dynamics,
            "parent",
            {"first": "opponent"},
            arm,
            protocol,
        )
        assert selection.parent_q == 1 and selection.plan is None
        assert verify_adoption(worlds, phases, selection, dynamics.policy_cache) == {}
        # Verify the real parent value through the same two-policy oracle path.
        # No altered comparison or synthetic positive gain is used as evidence.
        plan = phases["fixed"].result.plan
        checks = verify_adoption(
            worlds, phases, replace(selection, plan=plan), dynamics.policy_cache
        )
        assert checks["parent"]["values"] == {str(plan.baseline_action): "1"}
        selections[arm] = selection
    env = root.clone()
    actions, work = [], empty_work()
    while env.state[PHASE] != TERMINAL:
        action = actor.act(env)
        actions.append(action)
        add_step(work, env.step(action))
    source = {
        "seat": viewer,
        "actions": actions,
        "final": env.state.tolist(),
        "selected_root": {"state": root.state.tolist(), "decision": 0},
    }
    result = assess_three(source, selections, actor, actor, "parent")
    assert result["actual_work"] == {k: 3 * v for k, v in work.items()}
    assert result["actual_work"] == result["replay_work"]
    assert result["outcomes"]["parent"] == result["outcomes"]["root_only"]
    assert result["outcomes"]["parent"] == result["outcomes"]["controller"]


def test_complete_only_summary_and_tamper_rejection(tmp_path, monkeypatch):
    from scripts import prospective_incumbent as study

    protocol = {
        "cases": cases(),
        "chunk_size": 16,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 123990001,
        "scope": "Synthetic accounting fixture only.",
    }
    write(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(study, "check", lambda _: protocol)
    for chunk in range(32):
        folder = tmp_path / f"chunk_{chunk:03}"
        folder.mkdir()
        rows = [
            {
                "index": index,
                "status": "missing_root",
                "source": {
                    **case,
                    "final": [30] * 240,
                    "work": {"decisions": 100, "round_endings": 8, "completions": 1},
                    "identification_calls": {"wp": 2, "point": 3},
                },
            }
            for index, case in enumerate(protocol["cases"])
            if index // 16 == chunk
        ]
        write(folder / "records.json", rows)
        write(folder / "process.json", {"status": "complete"})
        write(
            folder / "complete.json",
            {
                "status": "complete",
                "records_sha256": sha(folder / "records.json"),
                "protocol_sha256": sha(tmp_path / "protocol.json"),
                "artifacts": {},
            },
        )
    summary = study.summarize(tmp_path)
    assert summary["actual_mean"] == [0] * 4
    assert summary["actual_simultaneous95_lower"] == [0] * 4
    assert not summary["development_strength_gate"]
    assert summary["work"]["source"]["completions"] == 512
    assert summary["work"]["actual_tails"] == {}
    assert summary["public_identification_calls"] == {"wp": 1024, "point": 1536}
    write(tmp_path / "chunk_031/records.json", [])
    with pytest.raises(AssertionError):
        study.summarize(tmp_path)
