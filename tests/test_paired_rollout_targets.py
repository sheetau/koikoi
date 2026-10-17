import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.paired_rollout_targets import match_split, paired_outcomes


def first_legal(env):
    return env.observe()[1].int().argmax(-1)


def reference_branch(state, first):
    env = ReferenceEnv(1, "paper8")
    env.state[:] = state
    player = int(state[PLAYER])
    decisions = 1
    env.step(first)
    while env.state[PHASE] != TERMINAL:
        env.step(int(np.flatnonzero(env.legal())[0]))
        decisions += 1
    difference = int(env.state[POINTS] - env.state[POINTS + 1]) * (1 - 2 * player)
    return int(np.sign(difference)), difference, decisions


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paired_branches_equal_independent_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = BatchEnv(4, device, "paper8", seed=18970001)
    for _ in range(23):
        env.step(first_legal(env))
    before = env.state.clone()
    legal = env.observe()[1]
    low = legal.int().argmax(-1)
    high = 49 - legal.flip(1).int().argmax(-1)
    proposals = torch.stack([low, high], dim=1)
    wins, points, work = paired_outcomes(env.state, proposals, first_legal)
    total = 0
    for index, state in enumerate(before.cpu().numpy()):
        for branch in (0, 1):
            expected = reference_branch(state, int(proposals[index, branch]))
            assert wins[index, branch] == expected[0]
            assert points[index, branch] == expected[1]
            total += expected[2]
    assert work.decisions == total and work.completed_branches == 8
    assert torch.equal(before, env.state)
    same, _, _ = paired_outcomes(env.state, torch.stack([low, low], dim=1), first_legal)
    assert torch.equal(same[:, 0], same[:, 1])


def test_match_group_split_excludes_cross_decision_leakage():
    groups = torch.tensor([[18, 0], [18, 0], [19, 1], [18, 0], [19, 1]])
    mask = match_split(groups)
    assert mask[0] == mask[1] == mask[3]
    assert mask[2] == mask[4]
