"""Reject a broken zero-update evaluation before launching strength training."""

import copy

import pytest

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
