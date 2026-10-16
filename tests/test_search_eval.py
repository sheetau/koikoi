import numpy as np
import torch

from koikoi.evaluate import duel, paired_interval
from koikoi.agents import GreedyAgent
from koikoi.layout import *
from koikoi.reference import ReferenceEnv, valid_deal
from koikoi.search import sample_hidden


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
            assert set(world.state[72:49+world.state[STOCK]]) == set(np.flatnonzero(world.state[:48] == -1))
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
    assert (first[0]+1-second[0])/2 == .5
    assert first[1] == second[1]
    assert paired_interval([.5]*10) == [.5, .5]
    assert paired_interval([.5]) is None
