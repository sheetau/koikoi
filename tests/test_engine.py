import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import *
from koikoi.reference import ReferenceEnv, score


def test_scoring_fixtures():
    assert score({0, 8, 28}) == 5
    assert score({0, 8, 40}) == 0
    assert score({0, 8, 28, 40}) == 7
    assert score({0, 8, 28, 44}) == 8
    assert score({0, 8, 28, 40, 44}) == 10
    assert score({8, 32}) == 1
    assert score({8, 32}, 1) == 4
    assert score({1, 5, 9, 21, 33, 37}) == 22
    assert score({1, 5, 9, 21, 33, 37}, 4) == 44


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_random_trajectories(profile, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    n = 12
    env = BatchEnv(n, device=device, profile=profile, seed=123, debug=True)
    refs = [ReferenceEnv(123+i, profile) for i in range(n)]
    rng = np.random.default_rng(42)
    completed = 0
    for _ in range(360):
        np.testing.assert_array_equal(env.state.cpu().numpy(), np.stack([r.state for r in refs]))
        obs, masks = env.observe()
        np.testing.assert_array_equal(obs.cpu().numpy(), np.stack([r.observe() for r in refs]))
        np.testing.assert_array_equal(masks.cpu().numpy(), np.stack([r.legal() for r in refs]))
        actions = [rng.choice(np.flatnonzero(r.legal())) for r in refs]
        expected = np.stack([r.step(a) for r, a in zip(refs, actions)])
        actual = env.step(torch.tensor(actions, device=device)).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        for i, r in enumerate(refs):
            if r.state[PHASE] == TERMINAL:
                completed += 1
                refs[i] = ReferenceEnv(123+i, profile, int(r.state[EPISODE])+1)
        env.reset_done()
    assert completed > 0


def test_hidden_information_noninterference():
    env = BatchEnv(8)
    obs, mask = env.observe()
    s = env.state
    for row in s:
        p = int(row[PLAYER])
        hidden = torch.where((row[:48] == 1-p) | (row[:48] == -1))[0]
        loc = row[hidden].clone().flip(0)
        row[hidden] = loc
        row[72:96] = row[72:96].flip(0)
        row[SEED] += 100
    other, other_mask = env.observe()
    assert torch.equal(obs, other)
    assert torch.equal(mask, other_mask)


def test_invalid_action_is_atomic():
    env = BatchEnv(2, debug=True)
    before = env.state.clone()
    with pytest.raises(ValueError):
        env.step(torch.tensor([-1, 49]))
    assert torch.equal(before, env.state)


def test_reference_card_conservation():
    env = ReferenceEnv()
    rng = np.random.default_rng(17)
    while env.state[PHASE] != TERMINAL:
        s = env.state
        assert all(-1 <= loc <= 5 for loc in s[:48])
        assert np.count_nonzero(s[:48] == -1) == s[STOCK]-23
        env.step(rng.choice(np.flatnonzero(env.legal())))
