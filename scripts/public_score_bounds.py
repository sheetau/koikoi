"""Admissible round-score bounds using public captures and remaining turns only."""

import numpy as np

from koikoi.layout import (
    KOI,
    KOIS,
    PHASE,
    PICK_DRAW,
    PICK_HAND,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    TERMINAL,
    TURN,
)
from koikoi.reference import BLUE, KASU, LIGHTS, RED, RIBBONS, TANE

RULE = "public-capture-capacity-score-bound-v1"


def score_upper(states, player):
    """Upper bound on any remaining award TO player (excluding dealer award).

    Opponent captures are unavailable forever. Each remaining discard/draw can
    capture at most four cards; a pending two-way choice captures exactly two.
    We maximize each yaku separately within that capacity, deliberately allowing
    incompatible maximizers. Then maximize over all reachable koi counts. This
    last maximum is necessary: score is not monotone in koi count at 3 -> 4.
    No card's hidden location, stock order or RNG field is consulted.
    """
    if player not in (0, 1):
        raise ValueError("Invalid player")
    held = states[:, :48] == 3 + player
    available = states[:, :48] != 4 - player
    active = states[:, PLAYER] == player
    first = states[:, TURN] + ~active
    turns = np.maximum(0, (17 - first) // 2)
    capacity = 8 * turns
    capacity -= active * np.select(
        [
            states[:, PHASE] == PICK_HAND,
            states[:, PHASE] == PICK_DRAW,
            states[:, PHASE] == KOI,
        ],
        [2, 6, 8],
        default=0,
    )
    capacity = np.maximum(0, capacity)

    def count(cards, mask):
        return mask[:, sorted(cards)].sum(axis=1)

    def attainable_count(cards):
        have = count(cards, held)
        return np.minimum(count(cards, available), have + capacity)

    def attainable_set(cards):
        return (count(cards, available) == len(cards)) & (
            len(cards) - count(cards, held) <= capacity
        )

    # All possible light awards; taking maxima also covers mandatory rain.
    light_count = attainable_count(LIGHTS)
    no_rain = LIGHTS - {40}
    lights = np.maximum.reduce(
        [
            10 * attainable_set(LIGHTS),
            8 * (attainable_set(no_rain) & ~held[:, 40]),
            7 * (light_count >= 4),
            5 * ((attainable_count(no_rain) >= 3) & ~held[:, 40]),
        ]
    )
    ordinary = (
        lights
        + 5 * attainable_set({20, 24, 36})
        + np.maximum(0, attainable_count(TANE) - 4)
        + 5 * attainable_set(RED)
        + 5 * attainable_set(BLUE)
        + 10 * attainable_set(RED | BLUE)
        + np.maximum(0, attainable_count(RIBBONS) - 4)
        + np.maximum(0, attainable_count(KASU) - 9)
    )
    sake = attainable_set({8, 32}).astype(np.int64) + attainable_set({28, 32})
    koi_min = states[:, KOIS + player]
    # Only turn indices <= 13 may declare koi. The current turn is included.
    koi_max = koi_min + np.maximum(0, (15 - first) // 2)
    upper = np.zeros(len(states), dtype=np.int64)
    for koi in range(8):
        base = ordinary + (1 if koi == 0 else 3) * sake
        score = base + koi if koi <= 3 else base * (koi - 2)
        possible = (koi_min <= koi) & (koi <= koi_max)
        upper = np.maximum(upper, np.where(possible, score, 0))
    return np.where(turns > 0, upper, 0)


def outcome_certificates(states, viewer):
    """Return public, strict win/loss masks for the final round only."""
    if viewer not in (0, 1):
        raise ValueError("Invalid viewer")
    live = (states[:, PHASE] != TERMINAL) & (states[:, ROUND] + 1 == states[:, ROUNDS])
    margin = states[:, POINTS + viewer] - states[:, POINTS + 1 - viewer]
    win = np.zeros(len(states), dtype=bool)
    loss = np.zeros(len(states), dtype=bool)
    for mask, player, target, lead in (
        (live & (margin > 2), 1 - viewer, win, margin),
        (live & (margin < -2), viewer, loss, -margin),
    ):
        # No turn remains for this player: only the one-point dealer award can
        # hurt the leading side. Avoid evaluating yaku in this frequent leaf.
        no_turn = (states[:, TURN] == 15) & (states[:, PLAYER] != player)
        target[mask & no_turn] = True
        pending = mask & ~no_turn
        if pending.any():
            target[pending] = lead[pending] > 2 * np.maximum(
                1, score_upper(states[pending], player)
            )
    return win, loss
