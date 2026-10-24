import numpy as np
import pytest
import torch

from koikoi.layout import EPISODE, KOIS, PHASE, PLAYER, POINTS, STOCK, TERMINAL
from koikoi.public_stock import (
    CHOOSE_STOCK,
    PROFILES,
    PUBLIC,
    PublicStockEnv,
    sample_worlds,
)
from koikoi.public_stock_native import PublicStockBackend, PublicStockBatch


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return PublicStockBackend(request.param)


@pytest.mark.parametrize("profile", PROFILES)
def test_complete_trajectories(backend, profile):
    seeds = [720101, 720102, 720103, 720104]
    batch = PublicStockBatch(seeds, profile, backend.device, backend)
    refs = [PublicStockEnv(seed, profile) for seed in seeds]
    rng = np.random.default_rng(83)
    for _ in range(700):
        np.testing.assert_array_equal(
            batch.state.cpu(), np.stack([e.state for e in refs])
        )
        obs, legal = batch.observe()
        np.testing.assert_allclose(
            obs.cpu(), np.stack([e.observe() for e in refs]), atol=1e-7
        )
        np.testing.assert_array_equal(legal.cpu(), np.stack([e.legal() for e in refs]))
        if all(e.state[PHASE] == TERMINAL for e in refs):
            return
        actions, outcomes = [], []
        for env in refs:
            if env.state[PHASE] == TERMINAL:
                actions.append(0)
                outcomes.append(np.zeros(4, dtype=np.int32))
            else:
                action = int(rng.choice(np.flatnonzero(env.legal())))
                actions.append(action)
                outcomes.append(env.step(action))
        np.testing.assert_array_equal(batch.step(actions).cpu(), np.stack(outcomes))
    pytest.fail("Match did not terminate")


@pytest.mark.parametrize("preview", [0, 1])
def test_choice_preserves_known_remainder(preview):
    env = PublicStockEnv(17, f"preview{preview}_choice1_stakes0")
    while env.state[PHASE] != CHOOSE_STOCK:
        env.step(int(np.flatnonzero(env.legal())[0]))
    old = env.state[PUBLIC : PUBLIC + 2].copy()
    env.step(int(old[1] - 1))
    assert env.state[PUBLIC] == old[0]
    assert env.state[48 + env.state[STOCK]] == old[0] - 1
    assert bool(env.state[PUBLIC + 1]) == bool(preview)
    assert env.observe()[(old[0] - 1) * 11 + 4] == 0


@pytest.mark.parametrize("profile", PROFILES)
def test_hidden_resampling_preserves_information(profile):
    env = PublicStockEnv(75, profile)
    rng = np.random.default_rng(98)
    for _ in range(20):
        worlds = sample_worlds(env, rng, 4)
        for world in worlds:
            other = env.clone()
            other.state = world
            np.testing.assert_array_equal(other.observe(), env.observe())
            np.testing.assert_array_equal(other.legal(), env.legal())
            np.testing.assert_array_equal(
                np.sort(world[72 : 49 + world[STOCK]]), np.flatnonzero(world[:48] == -1)
            )
            for j, card in enumerate(world[PUBLIC : PUBLIC + 2]):
                if card:
                    assert world[48 + world[STOCK] - j] == card - 1
            assert np.count_nonzero(world[:48] == 1 - world[PLAYER]) == round(
                env.observe()[541] * 8
            )
        if env.state[PHASE] == TERMINAL:
            break
        env.step(int(rng.choice(np.flatnonzero(env.legal()))))


def test_reset_matches_independent_python(backend):
    profile = "preview1_choice1_stakes1"
    batch = PublicStockBatch([91, 92], profile, backend.device, backend)
    batch.state[0, PHASE] = TERMINAL
    before = batch.state[1].clone()
    batch.reset_done()
    np.testing.assert_array_equal(
        batch.state[0].cpu(), PublicStockEnv(91, profile, 1).state
    )
    assert torch.equal(batch.state[1], before)
    assert batch.state[0, EPISODE] == 1


def test_stakes_is_only_own_double_and_both_yaku():
    env = PublicStockEnv(1, "preview0_choice0_stakes1")
    env.state[:48] = -1
    env.state[[0, 8, 28]] = 3
    env.state[[1, 5, 9]] = 4
    env.state[KOIS : KOIS + 2] = [1, 2]
    assert env.yaku(0) == 6
    assert env.yaku(1) == 7
    assert env.payment(0) == 26  # Neither seven-double nor opponent-koi double.
    env.state[KOIS] = 0
    assert env.payment(0) == 12
    before = env.state[POINTS : POINTS + 2].sum()
    env._settle(0, env.payment(0))
    assert env.state[POINTS : POINTS + 2].sum() == before


@pytest.mark.parametrize("profile", PROFILES)
def test_batched_sampling_and_search(backend, profile):
    from koikoi.public_stock_policy import PublicStockPolicy
    from koikoi.public_stock_search import rollout_actions, sample_batch, view

    torch.manual_seed(8)
    batch = PublicStockBatch([17, 18, 19], profile, backend.device, backend)
    generator = torch.Generator(device=backend.device).manual_seed(100)
    for _ in range(8):
        obs, mask = batch.observe()
        sampled = sample_batch(batch, 5, generator)
        worlds = view(sampled, batch)
        sampled_obs, sampled_mask = worlds.observe()
        torch.testing.assert_close(
            sampled_obs, obs.repeat_interleave(5, 0), rtol=0, atol=0
        )
        assert torch.equal(sampled_mask, mask.repeat_interleave(5, 0))
        for s in sampled.cpu().numpy():
            np.testing.assert_array_equal(
                np.sort(s[72 : 49 + s[STOCK]]), np.flatnonzero(s[:48] == -1)
            )
            start = 48 + 8 * (1 - s[PLAYER])
            assert np.bincount(s[start : start + 8] // 4, minlength=12).max() < 4
        batch.step(mask.float().argmax(1))
    model = PublicStockPolicy(32).to(backend.device).eval()
    before = batch.state.clone()
    actions, counters = rollout_actions(batch, model, 2, 4, generator)
    assert torch.equal(before, batch.state)
    _, mask = batch.observe()
    assert mask[torch.arange(3, device=backend.device), actions].all()
    assert counters["transitions"] >= counters["rollouts"]
