"""Conditional draws must be fresh, reproducible, and hidden-state invariant."""

import numpy as np
import pytest

from koikoi.layout import PLAYER, STOCK
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from scripts.sample_history_support import sample_supported_worlds


def fixture():
    env = ReferenceEnv(7)
    hidden = sample_hidden_batch(env, np.random.default_rng(35001), 2)
    hands = [np.flatnonzero(s[:48] == 1 - s[PLAYER]).tolist() for s in hidden]
    return env, hidden, hands


def test_hypotheses_and_rng_invariant_to_actual_hidden_world():
    env, hidden, hands = fixture()
    original = env.state.copy()
    rng, alternate_rng = np.random.default_rng(35003), np.random.default_rng(35003)
    worlds, indices = sample_supported_worlds(env.state, hands, rng, 32)
    other, other_indices = sample_supported_worlds(
        hidden[0], hands[::-1], alternate_rng, 32
    )
    np.testing.assert_array_equal(worlds, other)
    np.testing.assert_array_equal(indices, other_indices)
    assert rng.bit_generator.state == alternate_rng.bit_generator.state
    np.testing.assert_array_equal(env.state, original)
    for world in worlds:
        hand = np.flatnonzero(world[:48] == 1 - world[PLAYER]).tolist()
        assert hand in hands
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = world
        np.testing.assert_array_equal(ref.observe(), env.observe())
        assert sorted(world[48:96]) == list(range(48))
    assert len({tuple(w[72 : 49 + w[STOCK]]) for w in worlds}) > 1


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_particle_counts_rejected(count):
    env, _, hands = fixture()
    with pytest.raises(ValueError, match="particle count"):
        sample_supported_worlds(env.state, hands, np.random.default_rng(1), count)


def test_empty_or_duplicate_support_rejected():
    env, _, hands = fixture()
    for invalid in ([], [hands[0], hands[0][::-1]]):
        with pytest.raises(ValueError, match="unique hands"):
            sample_supported_worlds(env.state, invalid, np.random.default_rng(1), 1)
