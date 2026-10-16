"""Independent, deliberately readable rules oracle; no native dependencies."""

from __future__ import annotations

import copy
import numpy as np

from .layout import *

LIGHTS = {0, 8, 28, 40, 44}
TANE = {4, 12, 16, 20, 24, 29, 32, 36, 41}
RIBBONS = {1, 5, 9, 13, 17, 21, 25, 33, 37, 42}
KASU = (set(range(48)) - LIGHTS - TANE - RIBBONS) | {32}
RED, BLUE = {1, 5, 9}, {21, 33, 37}


def score(cards, koi=0):
    cards = set(cards)
    lights = len(cards & LIGHTS)
    result = {5: 10, 4: 7 if 40 in cards else 8}.get(lights, 0)
    if lights == 3 and 40 not in cards:
        result += 5
    result += 5 * ({20, 24, 36} <= cards)
    result += (3 if koi else 1) * (({8, 32} <= cards) + ({28, 32} <= cards))
    result += max(0, len(cards & TANE) - 4)
    result += 5 * (RED <= cards) + 5 * (BLUE <= cards) + 10 * ((RED | BLUE) <= cards)
    result += max(0, len(cards & RIBBONS) - 4) + max(0, len(cards & KASU) - 9)
    return result + koi if koi <= 3 else result * (koi - 2)


def mix(x):
    x &= 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


class RNG:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        return mix64(self.state)

    def below(self, n):
        threshold = (1 << 64) % n
        while True:
            r = self.next()
            if r >= threshold:
                return r % n


def mix64(x):
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def valid_deal(deck):
    return all(max(np.bincount(np.asarray(deck[a:a+8]) // 4, minlength=12)) < 4
               for a in (0, 8, 16))


class ReferenceEnv:
    def __init__(self, seed=1, profile="paper8", episode=0):
        rounds, bankrupt = PROFILES[profile]
        self.state = np.zeros(STATE_SIZE, dtype=np.int32)
        s = self.state
        s[ROUNDS], s[BANKRUPT], s[SEED], s[EPISODE] = rounds, bankrupt, seed, episode
        s[POINTS:POINTS+2] = 30
        s[DEALER] = mix(seed ^ mix(episode + 1)) & 1
        self.deal()

    def clone(self):
        return copy.deepcopy(self)

    def deal(self, deck=None):
        s = self.state
        if deck is None:
            rng = RNG(mix64(int(s[SEED]) ^ mix64(int(s[EPISODE]) + 1) ^ mix64(int(s[ROUND]) + 17)))
            while True:
                deck = list(range(48))
                for i in range(47, 0, -1):
                    j = rng.below(i + 1)
                    deck[i], deck[j] = deck[j], deck[i]
                if valid_deal(deck):
                    break
        if sorted(deck) != list(range(48)) or not valid_deal(deck):
            raise ValueError("Invalid initial deal")
        s[:48] = -1
        s[48:96] = deck
        s[np.asarray(deck[:8])] = 0
        s[np.asarray(deck[8:16])] = 1
        s[np.asarray(deck[16:24])] = 2
        s[PLAYER], s[TURN], s[PHASE] = s[DEALER], 0, DISCARD
        s[PENDING], s[STOCK], s[BEFORE] = -1, 47, 0
        s[KOIS:KOIS+2] = 0
        s[HISTORY:] = 0

    def yaku(self, player):
        return score(np.flatnonzero(self.state[:48] == 3 + player), int(self.state[KOIS+player]))

    def matches(self):
        s = self.state
        return [c for c in range(48) if s[c] == 2 and c // 4 == s[PENDING] // 4]

    def legal(self):
        s = self.state
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        if s[PHASE] == DISCARD:
            mask[:48] = s[:48] == s[PLAYER]
        elif s[PHASE] in (PICK_HAND, PICK_DRAW):
            mask[self.matches()] = True
        elif s[PHASE] == KOI:
            mask[48:] = True
        return mask

    def capture(self, chosen=-1):
        s = self.state
        matches = self.matches()
        if not matches:
            s[s[PENDING]] = 2
        else:
            selected = [chosen] if len(matches) == 2 else matches
            s[selected + [int(s[PENDING])]] = 3 + s[PLAYER]
        s[PENDING] = -1

    def finish(self, winner, points, result):
        s = self.state
        delta = points if winner == 0 else -points
        s[POINTS] += delta
        s[POINTS+1] -= delta
        result[1], result[3] = 1, delta
        if s[ROUND] + 1 == s[ROUNDS] or (s[BANKRUPT] and min(s[POINTS:POINTS+2]) <= 0):
            s[PHASE] = TERMINAL
            result[0] = int(np.sign(s[POINTS] - s[POINTS+1]))
            result[2] = 1
        else:
            s[ROUND] += 1
            s[DEALER] = winner
            self.deal()

    def end_turn(self, result):
        s = self.state
        if s[TURN] == 15:
            self.finish(int(s[DEALER]), 1, result)
        else:
            s[TURN] += 1
            s[PLAYER] = 1 - s[PLAYER]
            s[PHASE] = DISCARD

    def after_draw(self, result):
        s = self.state
        if self.yaku(s[PLAYER]) > s[BEFORE]:
            if s[TURN] >= 14:
                s[HISTORY+s[TURN]*5+4] = 1
                self.finish(int(s[PLAYER]), self.yaku(s[PLAYER]), result)
            else:
                s[PHASE] = KOI
        else:
            self.end_turn(result)

    def draw(self, result):
        s = self.state
        card = s[48+s[STOCK]]
        s[STOCK] -= 1
        s[PENDING], s[card] = card, 5
        s[HISTORY+s[TURN]*5+2] = card + 1
        if len(self.matches()) == 2:
            s[PHASE] = PICK_DRAW
        else:
            self.capture()
            self.after_draw(result)

    def step(self, action):
        if not 0 <= action < 50 or not self.legal()[action]:
            raise ValueError(f"Illegal action {action}")
        s = self.state
        result = np.zeros(4, dtype=np.int32)
        phase, h = s[PHASE], HISTORY+s[TURN]*5
        if phase == DISCARD:
            s[BEFORE] = self.yaku(s[PLAYER])
            s[PENDING], s[action], s[h] = action, 5, action + 1
            if len(self.matches()) == 2:
                s[PHASE] = PICK_HAND
            else:
                self.capture()
                self.draw(result)
        elif phase in (PICK_HAND, PICK_DRAW):
            s[h + (1 if phase == PICK_HAND else 3)] = action + 1
            self.capture(action)
            if phase == PICK_HAND:
                self.draw(result)
            else:
                self.after_draw(result)
        else:
            s[h+4] = 1 if action == STOP else 2
            if action == STOP:
                self.finish(int(s[PLAYER]), self.yaku(s[PLAYER]), result)
            else:
                s[KOIS+s[PLAYER]] += 1
                self.end_turn(result)
        return result

    def observe(self):
        s, p = self.state, int(self.state[PLAYER])
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        cards = obs[:528].reshape(48, 11)
        cards[:, 0] = s[:48] == p
        cards[:, 1] = s[:48] == 2
        cards[:, 2] = s[:48] == 3+p
        cards[:, 3] = s[:48] == 4-p
        cards[:, 4] = (s[:48] == 1-p) | (s[:48] == -1)
        cards[:, 5] = s[:48] == 5
        cards[s[64:72], 6] = 1
        for turn in range(16):
            for offset, feature in ((0, 7), (2, 8), (1, 9), (3, 10)):
                c = s[HISTORY+turn*5+offset] - 1
                if c >= 0:
                    cards[c, feature] = (turn+1)/16
        g = obs[528:]
        if s[PHASE] < 4:
            g[s[PHASE]] = 1
        g[4:16] = [s[TURN]/16, float(s[DEALER] == p), s[ROUND]/12, s[ROUNDS]/12,
                   s[POINTS+p]/64, s[POINTS+1-p]/64, s[KOIS+p]/8, s[KOIS+1-p]/8,
                   (s[STOCK]-23)/24, np.count_nonzero(s[:48] == 1-p)/8,
                   s[BEFORE]/32, s[BANKRUPT]]
        g[16:] = s[HISTORY+4::5]/2
        return obs
