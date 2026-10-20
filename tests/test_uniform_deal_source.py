from collections import Counter
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import DEALER, PHASE, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.uniform_deal_source import (
    DeckSample,
    LoggedDealSource,
    permutation,
    replay,
    sample_valid_deck,
)


def choices_for(target):
    deck, choices = list(range(len(target))), []
    for i in range(len(deck) - 1, 0, -1):
        j = deck.index(target[i], 0, i + 1)
        choices.append(j)
        deck[i], deck[j] = deck[j], deck[i]
    assert tuple(deck) == tuple(target)
    return choices


def fixture_deck(seed):
    return tuple(map(int, ReferenceEnv(seed).state[48:96]))


def test_every_bounded_draw_sequence_gives_one_small_permutation():
    counts = Counter()
    for draws in product(range(4), range(3), range(2)):
        iterator = iter(draws)
        counts[permutation(4, lambda _, iterator=iterator: next(iterator))] += 1
    assert len(counts) == 24 and set(counts.values()) == {1}
    assert permutation(0, lambda _: pytest.fail("No draw needed")) == ()
    with pytest.raises(ValueError, match="domain"):
        permutation(48, lambda n: n)


def test_invalid_deals_are_rejected_and_sampling_work_is_separate():
    target = fixture_deck(71)
    # Identity is invalid: each initial group contains a complete month.
    choices = iter([*range(47, 0, -1), *choices_for(target)])
    sampled = sample_valid_deck(lambda _: next(choices))
    assert sampled == DeckSample(target, 2, 94)
    with pytest.raises(RuntimeError, match="no substitute"):
        sample_valid_deck(lambda n: n - 1, max_attempts=1)
    with pytest.raises(ValueError):
        sample_valid_deck(max_attempts=0)


def test_failed_round_sampling_cannot_continue_with_seeded_fallback():
    calls = 0

    def draw():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Synthetic entropy failure")
        return DeckSample(fixture_deck(71), 1, 47)

    source = LoggedDealSource(53, "research12", draw)
    with pytest.raises(RuntimeError, match="entropy failure"):
        for _ in range(100):
            source.step(int(np.flatnonzero(source.env.legal())[0]))
    assert source.failed and len(source.deals) == 1
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        source.step(int(np.flatnonzero(source.env.legal())[0]))


@pytest.mark.parametrize("profile", ["paper8", "research12"])
def test_logged_deals_preserve_native_rules_and_replay_complete_matches(profile):
    fixtures = iter(fixture_deck(71 + i) for i in range(12))
    source = LoggedDealSource(53, profile, lambda: DeckSample(next(fixtures), 1, 47))
    native = BatchEnv(1, profile=profile, debug=True)
    native.state.copy_(torch.from_numpy(source.env.state[None].copy()))
    assert source.env.state[DEALER] == ReferenceEnv(53, profile).state[DEALER]
    assert type(source.env) is ReferenceEnv
    assert not hasattr(source.env, "deals") and not hasattr(source.env, "_draw")
    round_boundaries = 0
    for _ in range(1000):
        if source.env.state[PHASE] == TERMINAL:
            break
        before_round = int(source.env.state[ROUND])
        action = int(np.flatnonzero(source.env.legal())[0])
        native_result = native.step(torch.tensor([action], dtype=torch.int32))[
            0
        ].numpy()
        result = source.step(action)
        np.testing.assert_array_equal(result, native_result)
        if source.env.state[ROUND] != before_round:
            # Native rules auto-deal from their seed. Replace only that chance
            # outcome with the same logged valid deck, after checking metadata.
            native_view = ReferenceEnv.__new__(ReferenceEnv)
            native_view.state = native.state[0].numpy().copy()
            np.testing.assert_array_equal(
                native_view.state[96:128], source.env.state[96:128]
            )
            native_view.deal(source.deals[-1]["deck"])
            native.state.copy_(torch.from_numpy(native_view.state[None]))
            round_boundaries += 1
        np.testing.assert_array_equal(native.state[0].numpy(), source.env.state)
    assert source.env.state[PHASE] == TERMINAL and round_boundaries > 0
    assert len(source.deals) == round_boundaries + 1
    checked = replay(53, profile, source.actions, source.deals)
    np.testing.assert_array_equal(checked.env.state, source.env.state)
    assert checked.work == source.work and checked.deals == source.deals
    assert source.work["completions"] == 1
    with pytest.raises(ValueError, match="Missing"):
        replay(53, profile, source.actions, source.deals[:-1])
    with pytest.raises(ValueError, match="exactly"):
        replay(53, profile, source.actions, [*source.deals, source.deals[-1]])
    altered = deepcopy(source.deals)
    altered[0]["deck"] = list(range(48))
    with pytest.raises(ValueError, match="Invalid initial deal"):
        replay(53, profile, source.actions, altered)
    altered = deepcopy(source.deals)
    altered[0]["round"] = 1
    with pytest.raises(ValueError, match="out of order"):
        replay(53, profile, source.actions, altered)
