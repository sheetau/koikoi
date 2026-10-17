"""Reject a broken zero-update evaluation before launching strength training."""

import copy

import pytest
import torch

from scripts.audit_pretrained_ppo_screen import advance_criterion, validate_accounting
from scripts.run_pretrained_ppo_screen import initial_gate


def baseline():
    return {
        "pairs": 1000,
        "draw_adjusted_win_rate": 0.5,
        "records": [
            {
                "seed": 1000000 + i,
                "seat0_score": 1,
                "seat1_score": 0,
                "seat0_point_difference": 4,
                "seat1_point_difference": -4,
            }
            for i in range(1000)
        ],
    }


def test_initial_gate_checks_paired_compatibility():
    report = baseline()
    initial_gate(report)
    for mutate in (
        lambda r: r.update(pairs=999),
        lambda r: r["records"].pop(),
        lambda r: r.update(draw_adjusted_win_rate=0.50025),
        lambda r: r["records"][3].update(seat1_score=0.5),
        lambda r: r["records"][4].update(seat1_point_difference=-2),
    ):
        bad = copy.deepcopy(report)
        mutate(bad)
        with pytest.raises(AssertionError):
            initial_gate(bad)


def accounting_fixture():
    config = {
        "envs": 2,
        "horizon": 4,
        "minibatch": 4,
        "epochs": 2,
        "critic_warmup_updates": 1,
        "seed": 100,
    }
    rows = []
    for i in (1, 2):
        rows.append(
            {
                "updates": i,
                "decisions": 8 * i,
                "rounds": 2 * i - 1,
                "matches": 1,
                "policy_updates": i - 1,
                "warmup_updates": 1,
                "warmup_decisions": 8,
                "warmup_rounds": 1,
                "warmup_matches": 1,
                "optimizer_steps_value": 4 * i,
                "optimizer_steps_discard": 4 * (i - 1),
                "optimizer_steps_pick": 4 * (i - 1),
                "optimizer_steps_koi": 4 * (i - 1),
                "critic_only": i == 1,
            }
        )
    env = torch.zeros((2, 208), dtype=torch.int32)
    env[:, 110] = torch.tensor([100, 101])
    env[0, 111] = 1
    env[1, 103] = 2
    payload = {
        "config": config,
        "counters": {k: v for k, v in rows[-1].items() if k != "critic_only"},
        "env": env,
    }
    return payload, rows


def test_independent_accounting_fixture():
    payload, rows = accounting_fixture()
    validate_accounting(payload, rows, 2)


@pytest.mark.parametrize(
    "fault",
    [
        "decisions",
        "adam",
        "warmup",
        "phase_steps",
        "fractional",
        "nonfinite",
        "row_count",
        "episodes",
        "seed",
        "terminal",
        "rounds",
    ],
)
def test_accounting_rejects_inconsistent_evidence(fault):
    payload, rows = accounting_fixture()
    if fault == "decisions":
        rows[0]["decisions"] = 7
    elif fault == "adam":
        rows[0]["optimizer_steps_value"] = 3
    elif fault == "warmup":
        rows[1]["warmup_rounds"] = 3
    elif fault == "phase_steps":
        rows[0]["optimizer_steps_koi"] = 1
    elif fault == "fractional":
        rows[0]["rounds"] = 1.0
    elif fault == "nonfinite":
        rows[1]["policy_loss"] = float("nan")
    elif fault == "row_count":
        rows.pop()
    elif fault == "episodes":
        payload["env"][0, 111] = 0
    elif fault == "seed":
        payload["env"][1, 110] = 100
    elif fault == "terminal":
        payload["env"][1, 99] = 4
    else:
        payload["env"][1, 103] = 3
    with pytest.raises(AssertionError):
        validate_accounting(payload, rows, 2)


@pytest.mark.parametrize(
    "absolute,paired,expected",
    [
        (0.501, 0.001, True),
        (0.5, 0.001, False),
        (0.501, 0, False),
        (0.49, -0.01, False),
    ],
)
def test_criterion_requires_both_seeds_and_strict_lower_bounds(
    absolute, paired, expected
):
    endpoints = {
        "first": {
            "score_ci95": [0.6, 0.7],
            "versus_initial": {"score_ci95": [0.1, 0.2]},
        },
        "second": {
            "score_ci95": [absolute, 0.7],
            "versus_initial": {"score_ci95": [paired, 0.2]},
        },
    }
    assert advance_criterion(endpoints, ("first", "second")) is expected
