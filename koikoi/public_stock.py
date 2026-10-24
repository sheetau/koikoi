"""Public-stock factorial rules; isolated from frozen rule-transfer evidence."""

import numpy as np

from .layout import HISTORY, PHASE, PLAYER, POINTS, STOCK, TERMINAL
from .rule_variants import COMPLETED_ROUNDS, Variant, VariantEnv

PREVIEW, CHOICE = 512, 1024
PUBLIC = 115  # Two ordered card IDs plus one; zero means not yet public.
CHOOSE_STOCK = 5
OBS_SIZE = 660
PROFILES = {
    f"preview{a}_choice{b}_stakes{c}": Variant(
        f"preview{a}_choice{b}_stakes{c}", a * PREVIEW | b * CHOICE | c * 24
    )
    for a in range(2)
    for b in range(2)
    for c in range(2)
}


class PublicStockEnv(VariantEnv):
    def __init__(self, seed=1, profile="preview0_choice0_stakes0", episode=0):
        super().__init__(seed, PROFILES[profile], episode)

    def expose(self):
        s = self.state
        s[PUBLIC : PUBLIC + 2] = s[48 + s[STOCK] - np.arange(2)] + 1

    def deal(self, deck=None):
        super().deal(deck)
        self.state[PUBLIC : PUBLIC + 2] = 0
        if self.variant.flags & PREVIEW:
            self.expose()

    def legal(self):
        if self.state[PHASE] != CHOOSE_STOCK:
            return super().legal()
        mask = np.zeros(50, dtype=bool)
        mask[self.state[PUBLIC : PUBLIC + 2] - 1] = True
        return mask

    def draw(self, result):
        if self.variant.flags & CHOICE:
            self.expose()
            self.state[PHASE] = CHOOSE_STOCK
        else:
            self.take_stock(result)

    def take_stock(self, result):
        s = self.state
        s[PUBLIC], s[PUBLIC + 1] = s[PUBLIC + 1], 0
        # Reveal the replenishment before automatic capture/settlement. A new
        # round subsequently resets these slots through deal().
        if self.variant.flags & PREVIEW:
            s[PUBLIC : PUBLIC + 2] = s[48 + s[STOCK] - 1 - np.arange(2)] + 1
        super().draw(result)

    def step(self, action):
        s = self.state
        if s[PHASE] != CHOOSE_STOCK:
            return super().step(action)
        if not 0 <= action < 48 or not self.legal()[action]:
            raise ValueError(f"Illegal stock choice {action}")
        pt, rounds = int(s[POINTS]), int(s[COMPLETED_ROUNDS])
        top = 48 + s[STOCK]
        if s[top] != action:
            s[top], s[top - 1] = s[top - 1], s[top]
            s[PUBLIC], s[PUBLIC + 1] = s[PUBLIC + 1], s[PUBLIC]
        result = np.zeros(4, dtype=np.int32)
        self.take_stock(result)
        result[1], result[3] = s[COMPLETED_ROUNDS] - rounds, s[POINTS] - pt
        result[2] = s[PHASE] == TERMINAL
        result[0] = int(np.sign(s[POINTS] - s[POINTS + 1])) if result[2] else 0
        return result

    def observe(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[:560] = super().observe()
        for slot, card in enumerate(self.state[PUBLIC : PUBLIC + 2]):
            if card:
                obs[560 + slot * 48 + card - 1] = 1
                obs[(card - 1) * 11 + 4] = 0
        obs[656:] = [
            self.state[PHASE] == CHOOSE_STOCK,
            bool(self.variant.flags & PREVIEW),
            bool(self.variant.flags & CHOICE),
            bool(self.variant.flags & 24),
        ]
        return obs


def sample_worlds(env, rng, count):
    """Uniform deal-consistent proposal; no opponent action-likelihood model.

    Only observation-derived unknown cards may be reassigned. Public stock
    positions are fixed. Future deals receive independent simulation seeds.
    """
    from .layout import DEALER, SEED

    root = env.state
    p = int(root[PLAYER])
    obs = env.observe()
    unknown = np.flatnonzero(obs[:528].reshape(48, 11)[:, 4])
    hand_n = round(float(obs[541]) * 8)
    played = [
        int(root[HISTORY + t * 5]) - 1
        for t in range(16)
        if (int(root[DEALER]) + t) % 2 != p and root[HISTORY + t * 5]
    ]
    worlds = np.repeat(root[None], count, axis=0)
    for s in worlds:
        for _ in range(10000):
            hand = rng.choice(unknown, hand_n, replace=False)
            initial = played + hand.tolist()
            if np.bincount(np.asarray(initial, dtype=int) // 4, minlength=12).max() < 4:
                break
        else:
            raise RuntimeError("No deal-consistent hidden hand")
        s[unknown] = -1
        s[hand] = 1 - p
        stock = np.asarray([c for c in unknown if c not in set(hand.tolist())])
        rng.shuffle(stock)
        known_n = np.count_nonzero(s[PUBLIC : PUBLIC + 2])
        s[72 : 49 + s[STOCK] - known_n] = stock
        start = 48 + 8 * (1 - p)
        s[start : start + 8] = sorted(initial)
        s[SEED] = rng.integers(0, 2**31 - 1)
    return worlds
