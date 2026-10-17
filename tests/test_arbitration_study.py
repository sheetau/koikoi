import copy

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.arbitration import ArbitrationAgent, ArbitrationConfig, ArbitrationTrainer
from koikoi.layout import PHASE, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.analyze_arbitration_study import interval, raw_outcomes, tensor_equal
from scripts.run_arbitration_study import PARENTS


def rows():
    return [
        {
            "seed": 17,
            "decisions": 190,
            "seat0_score": 1,
            "seat1_score": 0,
            "seat0_point_difference": 6,
            "seat1_point_difference": -2,
        },
        {
            "seed": 18,
            "decisions": 180,
            "seat0_score": 0.5,
            "seat1_score": 1,
            "seat0_point_difference": 0,
            "seat1_point_difference": 10,
        },
    ]


def test_pair_statistics_and_rejected_corruption():
    scores, points = raw_outcomes(rows(), 17, 2)
    np.testing.assert_array_equal(scores, [0.5, 0.75])
    np.testing.assert_array_equal(points, [2, 5])
    assert interval(scores - scores) == [0, 0]
    for key, value in (
        ("seed", 18),
        ("seat1_score", 1),
        ("decisions", 0),
        ("seat0_point_difference", -6),
    ):
        bad = copy.deepcopy(rows())
        bad[0][key] = value
        with pytest.raises(ValueError):
            raw_outcomes(bad, 17, 2)
    with pytest.raises(ValueError):
        raw_outcomes(rows()[::-1], 17, 2)


def test_embedded_expert_weight_comparison():
    before = [{"weight": torch.tensor([1.0, 2.0])}, {"weight": torch.tensor([3.0])}]
    after = copy.deepcopy(before)
    assert tensor_equal(before, after)
    after[1]["weight"][0] += 1
    assert not tensor_equal(before, after)


@pytest.mark.parametrize("parent", [item[0] for item in PARENTS])
def test_initial_arbitrator_matches_actual_adapted_parent(tmp_path, parent):
    config = ArbitrationConfig(device="cpu", envs=2, horizon=4, minibatch=4)
    trainer = ArbitrationTrainer.from_parent(config, parent)
    path = tmp_path / "initial.pt"
    trainer.save(path)
    controlled, original = (
        ArbitrationAgent(path),
        NeuralAgent(parent, deterministic=True),
    )
    for seed in (18980001, 18980002):
        env = ReferenceEnv(seed, "paper8")
        while env.state[PHASE] != TERMINAL:
            proposed, expected = controlled.act(env), original.act(env)
            assert proposed == expected
            env.step(expected)
