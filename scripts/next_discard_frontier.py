"""A phase-aware public frontier with bounded prefix size at the five-draw gate."""

from dataclasses import dataclass
from math import comb, perm

from koikoi.layout import DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND, PLAYER, STOCK, TURN


@dataclass(frozen=True)
class NextDiscardFrontier:
    draws: int
    target_turn: int
    future_draws: int
    maximum_hands: int
    maximum_rows: int
    maximum_transitions: int


def next_discard_frontier(state, viewer, model_count=1):
    """Read public turn/phase/stock and opponent hand COUNT, never card identities.

    At the original five-draw gate the learner acts either after turn 10's draw
    or before turn 11's draw. Its next discard follows respectively one or two
    new chance draws. Forced steps can skip capture/koi decisions, but cannot
    skip that next discard unless the round ends.
    """
    turn, phase = int(state[TURN]), int(state[PHASE])
    expected = {10: (PICK_DRAW, KOI), 11: (DISCARD, PICK_HAND)}
    stock = int(state[STOCK]) - 23
    opponent_count = int((state[:48] == 1 - viewer).sum())
    if (
        viewer not in (0, 1)
        or state[PLAYER] != viewer
        or model_count != 1
        or turn not in expected
        or phase not in expected[turn]
        or stock != 13
        or opponent_count != (3 if turn == 10 else 2)
    ):
        raise ValueError("Expected the identified-model five-draw learner gate")
    draws = 14 - turn
    future_draws = 5 - draws
    hands = comb(stock + opponent_count, opponent_count)
    rows = hands * perm(stock, future_draws)
    own_steps = {DISCARD: 4, PICK_HAND: 3, PICK_DRAW: 2, KOI: 1}[phase]
    transitions = (own_steps + 4) * rows
    return NextDiscardFrontier(draws, turn + 2, future_draws, hands, rows, transitions)
