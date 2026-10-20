"""Conservative final-round outcome certificates using public captures only."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from koikoi.layout import (
    DEALER,
    KOIS,
    PHASE,
    POINTS,
    ROUND,
    ROUNDS,
    STATE_SIZE,
    TERMINAL,
    TURN,
)
from koikoi.reference import score

ALL_CARDS = (1 << 48) - 1


@dataclass(frozen=True)
class PublicTail:
    captured: tuple[int, int]
    kois: tuple[int, int]
    dealer: int
    turn: int
    phase: int
    round: int
    rounds: int
    margin: int  # Seat zero's current point lead.

    def __post_init__(self):
        if (
            len(self.captured) != 2
            or len(self.kois) != 2
            or any(not 0 <= c <= ALL_CARDS for c in self.captured)
            or self.captured[0] & self.captured[1]
            or any(not 0 <= k <= 7 for k in self.kois)
            or self.dealer not in (0, 1)
            or not 0 <= self.turn <= 15
            or not 0 <= self.phase <= TERMINAL
            or not 0 <= self.round < self.rounds
        ):
            raise ValueError("Malformed public round context")


def public_tail(state):
    """Project a simulator state without reading hands, deck, RNG or history."""
    s = np.asarray(state)
    if s.shape != (STATE_SIZE,):
        raise ValueError("Expected one packed state")
    captured = tuple(
        sum(1 << int(c) for c in np.flatnonzero(s[:48] == 3 + p)) for p in (0, 1)
    )
    return PublicTail(
        captured,
        tuple(map(int, s[KOIS : KOIS + 2])),
        int(s[DEALER]),
        int(s[TURN]),
        int(s[PHASE]),
        int(s[ROUND]),
        int(s[ROUNDS]),
        int(s[POINTS] - s[POINTS + 1]),
    )


def public_tail_from_key(key):
    """Read the public portion of a belief_tail.visible_keys entry."""
    row = np.frombuffer(key, dtype=np.int32)
    if row.shape != (160,) or row[0] not in (0, 1):
        raise ValueError("Expected a complete own/public history key")
    state = np.zeros(STATE_SIZE, dtype=np.int32)
    state[:48] = row[1:49]
    state[96:110] = row[49:63]
    return public_tail(state)


@lru_cache(maxsize=65536)
def score_ceiling(excluded, current_kois, future_kois):
    """All possible captures, and every feasible declaration count, upper-bound score.

    Cards already captured by the other player can never be acquired. At a
    fixed declaration count the score is monotone in the captured set. The
    score need not be monotone across declaration counts (notably 3 -> 4),
    hence the explicit maximum. The dealer's exhaustion award is also included.
    """
    cards = [c for c in range(48) if not (excluded >> c) & 1]
    return max(
        1,
        *(score(cards, k) for k in range(current_kois, current_kois + future_kois + 1)),
    )


def final_margin_bounds(context, viewer):
    """Return inclusive terminal point-margin bounds, or None outside final round."""
    if viewer not in (0, 1):
        raise ValueError("Seat must be zero or one")
    if context.phase == TERMINAL or context.round + 1 != context.rounds:
        return None
    ceilings = []
    for p in (0, 1):
        # Turns 14 and 15 force stopping on a score increase. Including the
        # current turn is conservative in every live decision phase.
        future = sum((context.dealer + t) % 2 == p for t in range(context.turn, 14))
        ceilings.append(score_ceiling(context.captured[1 - p], context.kois[p], future))
    margin = context.margin if viewer == 0 else -context.margin
    return margin - 2 * ceilings[1 - viewer], margin + 2 * ceilings[viewer]


def certified_outcome(context, viewer):
    """Signed win/loss only when every legal continuation has that outcome."""
    bounds = final_margin_bounds(context, viewer)
    if bounds is None:
        return None
    low, high = bounds
    if low > 0:
        return 1
    if high < 0:
        return -1
    return None
