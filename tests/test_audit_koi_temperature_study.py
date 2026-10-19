import copy
import json

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.audit_koi_temperature_study import (
    audit_cell,
    check_optimizer,
    independent_statistics,
    replay,
)
from scripts.koi_temperature_analysis import analyse
from scripts.run_arbitration_study import read, sha, write


def test_independent_paired_statistics_and_primary():
    points = np.zeros((2, 3, 2, 8, 2), dtype=np.int64)
    points[:, 2, :, :, 0] = 10
    points[:, 2, :, :, 1] = -2
    result = independent_statistics(points, 33, 251)
    assert not result["replicated_strength_gain"]
    rows = [r for r in result["contrasts"] if r["left"] == "calibrated"]
    assert all(r["win_change"] == 0 and r["point_change"] == 4 for r in rows)
    assert all(r["win_interval"] == [0.0, 0.0] for r in rows)
    points[:, 2] = 1
    assert independent_statistics(points, 33, 251)["replicated_strength_gain"]
    points[1, 2, 1] = -1
    assert not independent_statistics(points, 33, 251)["replicated_strength_gain"]
    points = np.random.default_rng(41).integers(-3, 4, (2, 3, 2, 8, 2))
    independent = independent_statistics(points, 33, 251)
    produced = analyse(points, 33, 251)
    assert all(independent[k] == produced[k] for k in independent)


def test_optimizer_steps_follow_each_head_and_warmup():
    config = {
        "lr": 3e-6,
        "value_lr": 1e-4,
        "envs": 8,
        "horizon": 8,
        "minibatch": 16,
        "epochs": 3,
        "critic_warmup_updates": 20,
    }
    payload = {
        "config": config,
        "counters": {
            "optimizer_steps_value": 276,
            "optimizer_steps_discard": 36,
            "optimizer_steps_pick": 24,
            "optimizer_steps_koi": 12,
        },
        "optimizer": {
            "param_groups": [
                {"params": [0, 1, 2], "lr": 3e-6},
                {"params": [3], "lr": 1e-4},
            ],
            "state": {
                i: {"step": torch.tensor(float(step)), "exp_avg": torch.zeros(2)}
                for i, step in enumerate((36, 24, 12, 276))
            },
        },
    }
    names = ["models.0.weight", "models.1.weight", "models.3.weight"]
    check_optimizer(payload, names, 23)
    altered = copy.deepcopy(payload)
    altered["optimizer"]["state"][2]["step"] += 1
    with pytest.raises(AssertionError):
        check_optimizer(altered, names, 23)
    altered = copy.deepcopy(payload)
    altered["optimizer"]["state"][3]["exp_avg"][0] = float("nan")
    with pytest.raises(AssertionError):
        check_optimizer(altered, names, 23)
    warm = copy.deepcopy(payload)
    warm["counters"] = {
        k: 240 if k.endswith("value") else 0 for k in payload["counters"]
    }
    warm["optimizer"]["state"] = {3: {"step": torch.tensor(240.0)}}
    check_optimizer(warm, names, 20)
    warm["optimizer"]["state"][0] = {"step": torch.tensor(0.0)}
    with pytest.raises(AssertionError):
        check_optimizer(warm, names, 20)


def test_full_replay_sign_allocation_and_corruption_rejection(tmp_path):
    folder = tmp_path / "evaluation"
    folder.mkdir()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"independent fixture, no neural policy")
    rows = []
    for rep in range(2):
        for seat in (0, 1):
            seed = 88110001 + rep
            env = ReferenceEnv(seed, "paper8")
            actions, rounds = [], 0
            while env.state[PHASE] != TERMINAL:
                legal = np.flatnonzero(env.legal())
                action = int(legal[(seed + len(actions)) % len(legal)])
                rounds += int(env.step(action)[1])
                actions.append(action)
            difference = int(env.state[POINTS] - env.state[POINTS + 1])
            rows.append(
                {
                    "parent": 1,
                    "opponent": "wp",
                    "replicate": rep,
                    "seat": seat,
                    "seed": seed,
                    "points": difference * (1 - 2 * seat),
                    "actions": actions,
                    "endpoint": env.state.tolist(),
                    "work": {
                        "decisions": len(actions),
                        "round_endings": rounds,
                        "full_matches": 1,
                    },
                }
            )
    rows_path = folder / "parent1_fixed_wp.jsonl"
    rows_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    totals = {k: sum(r["work"][k] for r in rows) for k in rows[0]["work"]}
    report_path = folder / "parent1_fixed_wp.json"
    write(
        report_path,
        {
            "status": "complete",
            "parent": 1,
            "policy": "fixed",
            "opponent": "wp",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "rows_sha256": sha(rows_path),
            "work": totals,
        },
    )
    values, work, _ = audit_cell(tmp_path, 1, "fixed", "wp", checkpoint, 2, 88110001)
    assert work == totals and work["full_matches"] == 4
    assert np.array_equal(values[:, 0], -values[:, 1]) and np.any(values != 0)
    nonzero = next(r for r in rows if r["points"])
    for field, value in (
        ("points", -nonzero["points"]),
        ("actions", nonzero["actions"][:-1]),
        ("actions", [-1]),
        ("endpoint", [0]),
        ("work", {**nonzero["work"], "round_endings": 0}),
    ):
        altered = copy.deepcopy(nonzero)
        altered[field] = value
        with pytest.raises(AssertionError):
            replay(altered)
    rows[0]["seed"] += 1
    rows_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    report = read(report_path)
    report["rows_sha256"] = sha(rows_path)
    write(report_path, report)
    with pytest.raises(AssertionError):
        audit_cell(tmp_path, 1, "fixed", "wp", checkpoint, 2, 88110001)
