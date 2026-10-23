"""Explicit research rule variants, isolated from the frozen paper8 engine.

Names denote modifications of paper8, not faithful reproductions of another
publisher's complete rules. Payoffs are symmetric zero-sum transfers.
"""

from dataclasses import dataclass

import numpy as np

from .layout import (
    BANKRUPT,
    BEFORE,
    DEALER,
    DISCARD,
    EPISODE,
    HISTORY,
    KOIS,
    PENDING,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    SEED,
    STOCK,
    TERMINAL,
    TURN,
)
from .reference import RNG, ReferenceEnv, mix64, score

FLAGS, AUTO_ROUNDS, COMPLETED_ROUNDS = 112, 113, 114
OPENING, SEVEN, OPPONENT_KOI, OWN_KOI = 1, 2, 4, 8
TOTAL_YAKU, DRAW_ZERO, SAKE_FIVE, NO_KOI_BONUS, SINGLE_KOI = 16, 32, 64, 128, 256


@dataclass(frozen=True)
class Variant:
    name: str
    flags: int
    rounds: int = 8
    bankrupt: bool = True


VARIANTS = {
    v.name: v
    for v in (
        Variant("paper_control", 0),
        Variant("opening_six", OPENING),
        Variant("seven_double", SEVEN),
        Variant("koi_cover", OPPONENT_KOI),
        Variant("own_koi_double", OWN_KOI),
        Variant("winner_takes_yaku", TOTAL_YAKU),
        Variant(
            "common_mix",
            OPENING | SEVEN | OPPONENT_KOI | DRAW_ZERO | SAKE_FIVE | NO_KOI_BONUS,
        ),
        Variant(
            "single_koi_mix",
            OPENING
            | SEVEN
            | OPPONENT_KOI
            | DRAW_ZERO
            | SAKE_FIVE
            | NO_KOI_BONUS
            | SINGLE_KOI,
        ),
        Variant(
            "combined_mix",
            OPENING
            | SEVEN
            | OPPONENT_KOI
            | OWN_KOI
            | TOTAL_YAKU
            | DRAW_ZERO
            | SAKE_FIVE
            | NO_KOI_BONUS,
        ),
    )
}


def initial_yaku(cards):
    counts = np.bincount(np.asarray(cards) // 4, minlength=12)
    return bool(np.any(counts == 4) or np.count_nonzero(counts == 2) == 4)


def valid_variant_deal(deck, flags):
    if sorted(deck) != list(range(48)):
        return False
    groups = (16,) if flags & OPENING else (0, 8, 16)
    return all(
        max(np.bincount(np.asarray(deck[a : a + 8]) // 4, minlength=12)) < 4
        for a in groups
    )


class VariantEnv(ReferenceEnv):
    def __init__(self, seed=1, variant="paper_control", episode=0):
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        if self.variant.rounds != 8:
            raise ValueError("This initial comparison fixes eight rounds")
        super().__init__(seed, "paper8", episode)

    def deal(self, deck=None):
        s, flags = self.state, self.variant.flags
        s[FLAGS], s[ROUNDS], s[BANKRUPT] = (
            flags,
            self.variant.rounds,
            self.variant.bankrupt,
        )
        if deck is None:
            rng = RNG(
                mix64(
                    int(s[SEED])
                    ^ mix64(int(s[EPISODE]) + 1)
                    ^ mix64(int(s[ROUND]) + 17)
                )
            )
            while True:
                deck = list(range(48))
                for i in range(47, 0, -1):
                    j = rng.below(i + 1)
                    deck[i], deck[j] = deck[j], deck[i]
                if valid_variant_deal(deck, flags):
                    break
        if not valid_variant_deal(deck, flags):
            raise ValueError("Invalid variant deal")
        s[:48], s[48:96] = -1, deck
        for owner, start in enumerate((0, 8, 16)):
            s[np.asarray(deck[start : start + 8])] = owner
        s[PLAYER], s[TURN], s[PHASE] = s[DEALER], 0, DISCARD
        s[PENDING], s[STOCK], s[BEFORE] = -1, 47, 0
        s[KOIS : KOIS + 2], s[HISTORY:] = 0, 0
        if flags & OPENING:
            # Simultaneous opening hands: dealer priority is an explicit local
            # convention, fixed before evaluation rather than inferred later.
            for player in (int(s[DEALER]), 1 - int(s[DEALER])):
                if initial_yaku(np.flatnonzero(s[:48] == player)):
                    s[AUTO_ROUNDS] += 1
                    self._settle(player, 6)
                    break

    def yaku(self, player):
        s, flags = self.state, self.variant.flags
        cards = set(map(int, np.flatnonzero(s[:48] == 3 + player)))
        koi = 0 if flags & NO_KOI_BONUS else int(s[KOIS + player])
        value = score(cards, koi)
        if flags & SAKE_FIVE:
            # Change sake's base value before the legacy declaration multiplier.
            extra = (5 - (3 if koi else 1)) * (({8, 32} <= cards) + ({28, 32} <= cards))
            value += extra * (koi - 2 if koi > 3 else 1)
        return value

    def payment(self, winner):
        s, f = self.state, self.variant.flags
        value = self.yaku(winner)
        if f & TOTAL_YAKU:
            value += self.yaku(1 - winner)
        if f & SEVEN and value >= 7:
            value *= 2
        if f & OPPONENT_KOI and s[KOIS + 1 - winner] > 0:
            value *= 2
        if f & OWN_KOI and s[KOIS + winner] > 0:
            value *= 2
        return value

    def _settle(self, winner, points):
        s = self.state
        s[POINTS + winner] += points
        s[POINTS + 1 - winner] -= points
        s[COMPLETED_ROUNDS] += 1
        if s[ROUND] + 1 == s[ROUNDS] or (
            s[BANKRUPT] and min(s[POINTS : POINTS + 2]) <= 0
        ):
            s[PHASE] = TERMINAL
        else:
            s[ROUND] += 1
            s[DEALER] = winner
            self.deal()

    def finish(self, winner, points, result):
        self._settle(winner, self.payment(winner))

    def end_turn(self, result):
        s = self.state
        if s[TURN] == 15:
            zero = bool(self.variant.flags & DRAW_ZERO)
            self._settle(
                1 - int(s[DEALER]) if zero else int(s[DEALER]), 0 if zero else 1
            )
        else:
            super().end_turn(result)

    def after_draw(self, result):
        s = self.state
        if (
            self.variant.flags & SINGLE_KOI
            and s[KOIS : KOIS + 2].sum()
            and self.yaku(s[PLAYER]) > s[BEFORE]
        ):
            s[HISTORY + s[TURN] * 5 + 4] = 1
            self._settle(int(s[PLAYER]), self.payment(int(s[PLAYER])))
        else:
            super().after_draw(result)

    def step(self, action):
        before_points, before_rounds = (
            int(self.state[POINTS]),
            int(self.state[COMPLETED_ROUNDS]),
        )
        result = super().step(action)
        s = self.state
        result[1], result[3] = (
            s[COMPLETED_ROUNDS] - before_rounds,
            s[POINTS] - before_points,
        )
        result[2] = int(s[PHASE] == TERMINAL)
        result[0] = int(np.sign(s[POINTS] - s[POINTS + 1])) if result[2] else 0
        return result


def solved_variant_action(env):
    s = env.state
    if s[PHASE] != 3:
        return None
    p, payment = int(s[PLAYER]), env.payment(int(s[PLAYER]))
    own, other = s[POINTS + p] + payment, s[POINTS + 1 - p] - payment
    if s[BANKRUPT] and other <= 0:
        return 48
    if s[ROUND] + 1 == s[ROUNDS]:
        if own > other:
            return 48
        if own < other:
            return 49
    return None
