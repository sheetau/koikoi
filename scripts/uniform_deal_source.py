"""Independent valid-deal sampling for a future, explicitly different evaluation law.

Uniformity assumes independent uniform bounded draws. Saved accepted decks,
rather than a small seed, reproduce each source. This is not a study runner.
"""

import secrets
from dataclasses import dataclass

from koikoi.layout import PHASE, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv, valid_deal


def permutation(size, randbelow):
    if not isinstance(size, int) or size < 0:
        raise ValueError("Expected a nonnegative permutation size")
    result = list(range(size))
    for i in range(size - 1, 0, -1):
        j = randbelow(i + 1)
        if not isinstance(j, int) or not 0 <= j <= i:
            raise ValueError("Bounded draw outside its domain")
        result[i], result[j] = result[j], result[i]
    return tuple(result)


@dataclass(frozen=True)
class DeckSample:
    deck: tuple[int, ...]
    attempts: int
    bounded_draws: int


def sample_valid_deck(randbelow=None, max_attempts=10000):
    """Rejection conditions a uniform permutation on the existing deal rule."""
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("Expected a positive attempt limit")
    draw = secrets.randbelow if randbelow is None else randbelow
    for attempt in range(1, max_attempts + 1):
        deck = permutation(48, draw)
        if valid_deal(deck):
            return DeckSample(deck, attempt, 47 * attempt)
    raise RuntimeError("Valid-deal attempt limit reached; no substitute deal")


class LoggedDealSource:
    """Keep chance sampling/logs outside the unchanged actor-facing environment.

    Actors receive ``env``, never this harness. Only newly started rounds get a
    replacement deck; dealer, scores and rules retain their reference values.
    ReferenceEnv's automatic seed deal is discarded before the next observation.
    """

    def __init__(self, seed=1, profile="paper8", draw=sample_valid_deck):
        self.env = ReferenceEnv(seed, profile)
        self._draw = draw
        self.deals = []
        self.actions = []
        self.work = {"decisions": 0, "round_endings": 0, "completions": 0}
        self.failed = False
        self._replace_deal()

    def _replace_deal(self):
        self.failed = True
        sample = self._draw()
        if (
            not isinstance(sample, DeckSample)
            or not isinstance(sample.attempts, int)
            or sample.attempts < 1
            or sample.bounded_draws != 47 * sample.attempts
        ):
            raise ValueError("Invalid deal sampling record")
        self.env.deal(list(sample.deck))
        self.deals.append(
            {
                "round": int(self.env.state[ROUND]),
                "deck": list(sample.deck),
                "attempts": sample.attempts,
                "bounded_draws": sample.bounded_draws,
            }
        )
        self.failed = False

    def step(self, action):
        if self.failed:
            raise RuntimeError("Source sampling failed; seeded fallback is forbidden")
        previous_round = int(self.env.state[ROUND])
        result = self.env.step(action)
        self.actions.append(int(action))
        self.work["decisions"] += 1
        self.work["round_endings"] += int(result[1])
        self.work["completions"] += int(result[2])
        if (
            self.env.state[PHASE] != TERMINAL
            and self.env.state[ROUND] != previous_round
        ):
            self._replace_deal()
        return result


def replay(seed, profile, actions, deals):
    """Replay a complete source from logged decks without sampling fresh chance."""
    position = 0

    def draw():
        nonlocal position
        if position >= len(deals):
            raise ValueError("Missing round deal")
        record = deals[position]
        if record["round"] != position:
            raise ValueError("Round deals out of order")
        position += 1
        return DeckSample(
            tuple(record["deck"]), record["attempts"], record["bounded_draws"]
        )

    source = LoggedDealSource(seed, profile, draw)
    for action in actions:
        source.step(action)
    if source.env.state[PHASE] != TERMINAL or position != len(deals):
        raise ValueError("Expected one complete match and exactly its used deals")
    return source
