import itertools

import numpy as np
import pytest

from koikoi.layout import (
    KOI,
    KOIS,
    PHASE,
    PICK_DRAW,
    PICK_HAND,
    PLAYER,
    POINTS,
    ROUND,
    STOCK,
    TERMINAL,
    TURN,
)
from koikoi.reference import ReferenceEnv, score
from scripts.public_score_bounds import outcome_certificates, score_upper


def test_upper_dominates_exhaustive_capture_subsets_and_koi_counts():
    rng = np.random.default_rng(814303)
    # Small available sets permit exhaustive checking of every relaxed capture
    # set, including incompatible yaku maxima and the sake dual classification.
    for player, phase in itertools.product((0, 1), (0, PICK_HAND, PICK_DRAW, KOI)):
        for _ in range(24):
            s = ReferenceEnv(1).state
            s[:48] = 4 - player
            cards = rng.choice(48, 11, replace=False)
            held, free = cards[:5], cards[5:]
            s[held], s[free] = 3 + player, -1
            s[PLAYER], s[TURN], s[PHASE] = player, 13, phase
            s[KOIS + player] = rng.integers(0, 7)
            bound = int(score_upper(s[None], player)[0])
            capacity = {0: 8, PICK_HAND: 6, PICK_DRAW: 2, KOI: 0}[phase]
            for n in range(min(capacity, len(free)) + 1):
                for extra in itertools.combinations(free, n):
                    for koi in (int(s[KOIS + player]), int(s[KOIS + player]) + 1):
                        assert score([*held, *extra], koi) <= bound


def test_koi_three_to_four_is_not_assumed_monotone():
    s = ReferenceEnv(1).state
    s[:48] = 4
    cards = [4, 12, 16, 29, 41]  # One-point tane, no other yaku.
    s[cards] = 3
    s[PLAYER], s[TURN], s[PHASE], s[KOIS] = 0, 12, KOI, 3
    assert score(cards, 3) == 4 and score(cards, 4) == 2
    assert score_upper(s[None], 0)[0] == 4


def test_certificate_uses_public_captures_only_and_strict_final_margin():
    s = ReferenceEnv(1).state
    s[ROUND], s[TURN], s[PLAYER] = 7, 12, 0
    s[POINTS : POINTS + 2] = [59, 1]
    s[:48] = 3
    s[[2, 3, 6, 7, 10, 11]] = -1
    assert outcome_certificates(s[None], 0)[0][0]
    t = s.copy()
    # Preserve public captured identities but scramble every hidden location,
    # stock order and seed/episode field.
    t[:48][t[:48] == -1] = 1
    t[48:96] = np.arange(48)[::-1]
    t[110:112] = 999
    for viewer in (0, 1):
        for a, b in zip(
            outcome_certificates(s[None], viewer),
            outcome_certificates(t[None], viewer),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
    s[TURN], s[PLAYER] = 15, 0
    s[POINTS : POINTS + 2] = [31, 29]
    assert not outcome_certificates(s[None], 0)[0][0]
    s[POINTS] += 1
    assert outcome_certificates(s[None], 0)[0][0]
    s[ROUND] = 6
    assert not outcome_certificates(s[None], 0)[0][0]


@pytest.mark.parametrize("seed", [31010004, 71403, 81430])
def test_every_legal_late_continuation_respects_bounds(seed):
    env = ReferenceEnv(seed)
    # Reproducible legal development trajectory; koi keeps it alive where legal.
    while env.state[TURN] < 14:
        legal = np.flatnonzero(env.legal())
        env.step(49 if 49 in legal else int(legal[0]))
    env.state[ROUND] = 7
    root = env.state.copy()
    upper = [int(score_upper(root[None], p)[0]) for p in (0, 1)]
    visits = 0

    def visit(e):
        nonlocal visits
        for action in np.flatnonzero(e.legal()):
            probe = e.clone()
            result = probe.step(int(action))
            draw = int(e.state[STOCK] - probe.state[STOCK])
            tops = range(72, 49 + int(e.state[STOCK])) if draw else (None,)
            for position in tops:
                child = e.clone()
                if position is not None:
                    top = 48 + int(child.state[STOCK])
                    child.state[position], child.state[top] = (
                        child.state[top],
                        child.state[position],
                    )
                result = child.step(int(action))
                visits += 1
                if child.state[PHASE] == TERMINAL:
                    delta = int(result[3])
                    winner = 0 if delta > 0 else 1
                    assert abs(delta) <= max(1, upper[winner])
                else:
                    # Bounds stay valid after captures, turns and declarations.
                    for p in (0, 1):
                        assert score_upper(child.state[None], p)[0] <= upper[p]
                    visit(child)

    visit(env)
    assert visits >= 10


def test_empty_and_invalid_viewer():
    empty = np.empty((0, len(ReferenceEnv(1).state)), dtype=np.int32)
    assert score_upper(empty, 0).size == 0
    assert all(x.size == 0 for x in outcome_certificates(empty, 1))
    with pytest.raises(ValueError):
        outcome_certificates(empty, 2)
