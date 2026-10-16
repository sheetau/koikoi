import numpy as np
import pytest
import torch

from koikoi.layout import DEALER, KOI, PHASE, PLAYER, POINTS, STOCK, TERMINAL, TURN
from koikoi.reference import ReferenceEnv, score
from koikoi.search import sample_hidden
from koikoi.upstream import (
    PHASES,
    UPSTREAM_PATH,
    PublishedAgent,
    inject_deal,
    load_source,
    original_action,
    reconstruct,
    skip_forced,
)

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def test_scoring_against_upstream():
    upstream = load_source()
    r = upstream.KoiKoiRoundState(dealer=1)
    rng = np.random.default_rng(111)
    for _ in range(2000):
        cards = rng.choice(48, size=rng.integers(0, 36), replace=False)
        koi = int(rng.integers(0, 8))
        r.pile[1] = [[int(c) // 4 + 1, int(c) % 4 + 1] for c in cards]
        r.koikoi[1] = [1] * koi + [0] * (8 - koi)
        assert score(cards, koi) == r.yaku_point(1)


def test_full_matches_against_original():
    source = load_source()
    rng = np.random.default_rng(222)
    for seed in range(12):
        ref = ReferenceEnv(seed)
        s = ref.state
        game = source.KoiKoiGameState(init_dealer=int(s[DEALER]) + 1)
        inject_deal(game.round_state, s[48:96])
        while s[PHASE] != TERMINAL:
            r = game.round_state
            skip_forced(r)
            assert r.state == PHASES[s[PHASE]]
            assert r.turn_player == s[PLAYER] + 1
            assert r.turn_16 == s[TURN] + 1
            for p in range(2):
                assert r.yaku_point(p + 1) == ref.yaku(p)
                assert (
                    sorted((a - 1) * 4 + b - 1 for a, b in r.hand[p + 1])
                    == np.flatnonzero(s[:48] == p).tolist()
                )
                assert (
                    sorted((a - 1) * 4 + b - 1 for a, b in r.pile[p + 1])
                    == np.flatnonzero(s[:48] == 3 + p).tolist()
                )
            assert (
                sorted((a - 1) * 4 + b - 1 for a, b in r.field)
                == np.flatnonzero(s[:48] == 2).tolist()
            )
            assert r.stock == [
                [int(c) // 4 + 1, int(c) % 4 + 1] for c in s[72 : 49 + s[STOCK]]
            ]
            expected_mask = ref.legal()[48:] if s[PHASE] == KOI else ref.legal()[:48]
            np.testing.assert_array_equal(r.action_mask, expected_mask)
            action = int(rng.choice(np.flatnonzero(ref.legal())))
            phase = int(s[PHASE])
            r.step(original_action(phase, action))
            skip_forced(r)
            result = ref.step(action)
            if result[1]:
                assert r.round_over
                assert r.round_point[1] == result[3]
                game.new_round()
                assert game.point == {1: s[POINTS], 2: s[POINTS + 1]}
                assert game.game_over == bool(result[2])
                if not result[2]:
                    inject_deal(game.round_state, s[48:96])
            else:
                replayed = reconstruct(ref)
                torch.testing.assert_close(
                    replayed.feature_tensor, game.feature_tensor, rtol=0, atol=0
                )


def test_published_weights_load_and_choose_legal():
    torch.set_num_threads(2)
    agent = PublishedAgent()
    ref = ReferenceEnv(77)
    for _ in range(8):
        action = agent.act(ref)
        assert ref.legal()[action]
        ref.step(action)


def test_original_features_do_not_reveal_hidden_world():
    ref = ReferenceEnv(9)
    rng = np.random.default_rng(70)
    for _ in range(40):
        other = ref.clone()
        other.state = sample_hidden(ref, rng)
        torch.testing.assert_close(
            reconstruct(ref).feature_tensor,
            reconstruct(other).feature_tensor,
            rtol=0,
            atol=0,
        )
        ref.step(int(rng.choice(np.flatnonzero(ref.legal()))))
