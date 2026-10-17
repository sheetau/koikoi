"""Enumerated support must derive hypotheses from public context only."""

import numpy as np

from koikoi.layout import PLAYER
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from scripts.enumerate_history_support import canonical_world


def test_canonical_hypothesis_independent_of_actual_hidden_state():
    env = ReferenceEnv(7)
    original = env.state.copy()
    hand = np.flatnonzero(original[:48] == 1 - original[PLAYER]).tolist()
    alternate = sample_hidden_batch(env, np.random.default_rng(34001), 1)[0]
    first = canonical_world(original, hand)
    second = canonical_world(alternate, hand)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(env.state, original)
    np.testing.assert_array_equal(canonical_world(original, hand[::-1]), first)


def test_invalid_initial_month_is_excluded():
    env = ReferenceEnv(7)
    own = list(range(0, 32, 4))
    opponent = list(range(1, 32, 4))
    field = list(range(2, 32, 4))
    deck = own + opponent + field
    deck += sorted(set(range(48)) - set(deck))
    env.deal(deck)
    env.state[PLAYER] = 0
    assert canonical_world(env.state, [36, 37, 38, 39, 32, 33, 40, 41]) is None
