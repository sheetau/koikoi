import numpy as np
import pytest

from koikoi.agents import GreedyAgent
from koikoi.evaluate import duel, paired_interval, seed_agents
from koikoi.layout import PHASE, STOCK, TERMINAL
from koikoi.reference import ReferenceEnv, valid_deal
from koikoi.search import SearchAgent, sample_hidden, sample_hidden_batch


def test_sampling_preserves_observation_and_deck():
    env = ReferenceEnv(1234)
    rng = np.random.default_rng(45)
    for _ in range(40):
        observation = env.observe()
        for __ in range(8):
            world = env.clone()
            world.state = sample_hidden(env, rng)
            np.testing.assert_array_equal(observation, world.observe())
            assert sorted(world.state[48:96]) == list(range(48))
            assert valid_deal(world.state[48:96])
            assert set(world.state[72 : 49 + world.state[STOCK]]) == set(
                np.flatnonzero(world.state[:48] == -1)
            )
        env.step(rng.choice(np.flatnonzero(env.legal())))


def test_sampling_independent_of_real_hidden_world():
    env = ReferenceEnv(321)
    alternate = env.clone()
    alternate.state = sample_hidden(env, np.random.default_rng(9))
    a = sample_hidden(env, np.random.default_rng(10))
    b = sample_hidden(alternate, np.random.default_rng(10))
    np.testing.assert_array_equal(a, b)


def test_deterministic_duplicate_pairs_and_interval():
    a, b = GreedyAgent(), GreedyAgent()
    first = duel(a, b, 987, "paper8")
    second = duel(b, a, 987, "paper8")
    assert (first[0] + 1 - second[0]) / 2 == 0.5
    assert first[1] == second[1]
    assert paired_interval([0.5] * 10) == [0.5, 0.5]
    assert paired_interval([0.5]) is None


def test_pair_replay_is_independent_of_prior_rng_consumption():
    from koikoi.agents import RandomAgent

    a, b = RandomAgent(), GreedyAgent()
    seed_agents(a, b, 4321)
    first = duel(a, b, 4321, "paper8")
    a.rng.random(1000)
    seed_agents(a, b, 4321)
    second = duel(a, b, 4321, "paper8")
    assert first[:2] == second[:2]
    assert first[3] == second[3]


def test_search_is_legal_and_hidden_world_independent():
    import torch

    from koikoi.model import PolicyValue

    class Neural:
        device = "cpu"
        belief_trained = False
        model = PolicyValue(width=32).eval()

    reference = ReferenceEnv(7)
    alternate = reference.clone()
    alternate.state = sample_hidden(reference, np.random.default_rng(22))
    actions = []
    for env in (reference, alternate):
        torch.manual_seed(17)
        search = SearchAgent(
            Neural(), particles=4, horizon=4, seed=33, use_belief=False, koi_only=False
        )
        action = search.act(env)
        assert env.legal()[action]
        assert search.simulated_decisions == env.legal().sum() * 16
        actions.append(action)
    assert actions[0] == actions[1]


@pytest.mark.parametrize("profile", ["paper8", "research12"])
@pytest.mark.parametrize("weighted", [False, True])
def test_batched_proposals_preserve_scalar_stream_and_hidden_independence(
    profile, weighted
):
    env = ReferenceEnv(1234, profile)
    play_rng = np.random.default_rng(123)
    logits = np.linspace(-2, 2, 48) if weighted else None
    for step in range(48):
        before = env.state.copy()
        first, second = np.random.default_rng(step), np.random.default_rng(step)
        worlds = sample_hidden_batch(env, first, 16, logits)
        singles = np.stack([sample_hidden(env, second, logits) for _ in range(16)])
        np.testing.assert_array_equal(worlds, singles)
        assert first.bit_generator.state == second.bit_generator.state
        np.testing.assert_array_equal(env.state, before)
        alternate = env.clone()
        alternate.state = sample_hidden(env, np.random.default_rng(step + 100))
        hidden_changed = sample_hidden_batch(
            alternate, np.random.default_rng(step), 16, logits
        )
        np.testing.assert_array_equal(worlds, hidden_changed)
        for state in worlds:
            world = env.clone()
            world.state = state
            np.testing.assert_array_equal(env.observe(), world.observe())
            assert valid_deal(state[48:96])
            assert sorted(state[48:96]) == list(range(48))
        env.step(play_rng.choice(np.flatnonzero(env.legal())))
        if env.state[PHASE] == TERMINAL:
            env = ReferenceEnv(1235 + step, profile)


def test_batched_proposals_reject_empty_request():
    with pytest.raises(ValueError, match="At least one"):
        sample_hidden_batch(ReferenceEnv(1), np.random.default_rng(1), 0)
