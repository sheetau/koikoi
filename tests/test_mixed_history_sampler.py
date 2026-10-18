"""Exact finite transition checks, complete replay and observation invariance."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from koikoi.search import sample_hidden
from scripts.diagnose_history_support import reference, replay
from scripts.enumerate_history_support import canonical_world
from scripts.mixed_history_sampler import (
    chain_marginals,
    draw_hand_chain,
    sample_public_hand_chain,
)
from tests.test_history_conditioned_worlds import Extremal, late_root


class ScriptedRng:
    def __init__(self, choices, coin=0):
        self.choices = iter(choices)
        self.coin = coin

    def choice(self, values, size=None, replace=True):
        result = next(self.choices)
        if size is not None:
            assert len(result) == size and not replace and set(result) <= set(values)
        else:
            assert result in values
        return result

    def random(self):
        return self.coin


def test_actual_mixed_kernel_preserves_uniform_mass_and_connects_components():
    pool = tuple(range(5))
    support = ((0, 1), (0, 2), (3, 4))
    matrix = {a: dict.fromkeys(support, Fraction(0)) for a in support}
    for start in support:
        for candidate in combinations(pool, 2):
            chain = draw_hand_chain(
                pool,
                2,
                lambda h: h in support,
                ScriptedRng([start, candidate]),
                proposals=2,
            )
            matrix[start][chain.states[-1]] += Fraction(1, 5 * 10)
        for outgoing in start:
            for incoming in set(pool) - set(start):
                chain = draw_hand_chain(
                    pool,
                    2,
                    lambda h: h in support,
                    ScriptedRng([start, outgoing, incoming], coin=0.9),
                    proposals=2,
                )
                matrix[start][chain.states[-1]] += Fraction(4, 5 * 2 * 3)
    for a in support:
        assert sum(matrix[a].values()) == 1
        for b in support:
            assert matrix[a][b] == matrix[b][a]
        assert sum(matrix[b][a] / 3 for b in support) == Fraction(1, 3)
    assert matrix[(0, 1)][(3, 4)] > 0  # No one-card swap connects these states.


def test_rejected_moves_are_retained_and_budget_failures_are_explicit():
    calls = []

    def compatible(hand):
        calls.append(hand)
        return hand == (0, 1)

    chain = draw_hand_chain(
        range(4),
        2,
        compatible,
        ScriptedRng([(2, 3), (0, 1), (2, 3), (0, 1)]),
        proposals=4,
        global_probability=1,
    )
    assert chain.first_supported_attempt == 2
    assert chain.states == ((0, 1),) * 3
    assert chain.cache_hits == 2 and len(calls) == 2
    assert chain_marginals(chain)["independent_sample_size"] is None
    failed = draw_hand_chain(
        range(4), 2, lambda h: False, ScriptedRng([(0, 1)] * 4), proposals=4
    )
    assert failed.status == "no_supported_proposal" and not failed.states
    assert failed.attempts == 4 and failed.cache_hits == 3
    with pytest.raises(ValueError, match="within budget"):
        chain_marginals(failed)


def test_public_wrapper_matches_full_replay_and_hidden_world_substitution():
    root = late_root()
    original = root.state.copy()
    model = Extremal()
    result = sample_public_hand_chain(
        root, {"declared": model}, np.random.default_rng(4), proposals=48
    )
    for hand, compatible in result["chain"].checked:
        world = canonical_world(root.state, hand)
        expected = world is not None and replay(world, model)["compatible"]
        assert compatible == expected
    alternate = reference(sample_hidden(root, np.random.default_rng(71)))
    changed = sample_public_hand_chain(
        alternate, {"declared": model}, np.random.default_rng(4), proposals=48
    )
    assert result == changed
    np.testing.assert_array_equal(root.state, original)


@pytest.mark.parametrize(
    "settings",
    [
        {"proposals": 0},
        {"proposals": True},
        {"global_probability": 0},
        {"global_probability": float("nan")},
        {"global_probability": True},
    ],
)
def test_invalid_controls_do_not_sample_or_call_likelihood(settings):
    def unexpected(*args):
        raise AssertionError("Unexpected evaluation")

    with pytest.raises(ValueError):
        draw_hand_chain(range(5), 2, unexpected, None, **settings)
