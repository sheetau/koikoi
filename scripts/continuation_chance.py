"""Fresh training deals from public match context; not an online policy API."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from koikoi.layout import (
    BANKRUPT,
    DEALER,
    DISCARD,
    EPISODE,
    PHASE,
    POINTS,
    ROUND,
    ROUNDS,
    SEED,
    STATE_SIZE,
    TURN,
)
from koikoi.reference import ReferenceEnv


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class PublicContinuationContext:
    """Public state before a new round's deal, without cards or opponent labels.

    Only rounds2--8 of an unfinished paper8 match are supported. Identification
    provenance and learner-seat metadata remain the caller's responsibility.
    """

    round_index: int
    dealer: int
    points: tuple[int, int]

    def __post_init__(self):
        if not 1 <= integer(self.round_index, "round_index") <= 7:
            raise ValueError("Expected continuation round2--8")
        if integer(self.dealer, "dealer") not in (0, 1):
            raise ValueError("Invalid dealer")
        if not isinstance(self.points, tuple) or len(self.points) != 2:
            raise ValueError("Expected an immutable pair of point totals")
        points = tuple(integer(p, "points") for p in self.points)
        if min(points) <= 0 or sum(points) != 60:
            raise ValueError("Expected an unfinished paper8 match with60 total points")

    @classmethod
    def from_root(cls, state):
        """Read only public context from an already audited fresh-round root."""
        if (
            not isinstance(state, np.ndarray)
            or state.shape != (STATE_SIZE,)
            or state.dtype != np.int32
        ):
            raise ValueError("Expected one packed int32 root")
        if (state[ROUNDS], state[BANKRUPT], state[TURN], state[PHASE]) != (
            8,
            1,
            0,
            DISCARD,
        ):
            raise ValueError("Expected a fresh paper8 continuation boundary")
        return cls(
            int(state[ROUND]),
            int(state[DEALER]),
            tuple(map(int, state[POINTS : POINTS + 2])),
        )


def renew_deal(context, seed, episode=0):
    """Draw a legal full deal and set the future chance key, preserving context.

    Returned packed state is private simulator data. Training policies must use
    the existing observation builders. No existing cards,stock,seed,episode,
    history or RNG state enters this constructor.
    """
    if not isinstance(context, PublicContinuationContext):
        raise TypeError("Expected validated public continuation context")
    seed, episode = integer(seed, "seed"), integer(episode, "episode")
    if not 0 <= seed < 2**31 or not 0 <= episode < 2**31:
        raise ValueError("Chance keys must fit nonnegative int32")
    # Bypass match initialization: it would select a new initial dealer and
    # reset scores. The existing deal routine changes neither score nor dealer.
    env = object.__new__(ReferenceEnv)
    state = env.state = np.zeros(STATE_SIZE, dtype=np.int32)
    state[ROUNDS], state[BANKRUPT] = 8, 1
    state[ROUND], state[DEALER] = context.round_index, context.dealer
    state[POINTS : POINTS + 2] = context.points
    state[SEED], state[EPISODE] = seed, episode
    env.deal()
    return state
