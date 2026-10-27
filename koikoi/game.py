"""Presentation events for the desktop game. Research engines stay unchanged."""

from __future__ import annotations

import random
import secrets

import numpy as np

from .layout import (
    BEFORE,
    CONTINUE,
    DEALER,
    DISCARD,
    KOI,
    KOIS,
    PENDING,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    STOCK,
    TERMINAL,
    TURN,
)
from .reference import BLUE, KASU, LIGHTS, RED, RIBBONS, TANE, ReferenceEnv
from .rule_variants import OWN_KOI, TOTAL_YAKU, Variant, VariantEnv


def yaku_rows(cards, koi=0):
    """Scoring breakdown and physical card membership, in display priority order."""
    cards = set(cards)
    rows = []

    def add(key, members, points):
        rows.append({"key": key, "cards": sorted(members), "points": points})

    lights = cards & LIGHTS
    if len(lights) == 5:
        add("five", lights, 10)
    elif len(lights) == 4:
        add("rain_four" if 40 in lights else "four", lights, 7 if 40 in lights else 8)
    elif len(lights) == 3 and 40 not in lights:
        add("three", lights, 5)
    for group, key, points in (
        ({20, 24, 36}, "animals", 5),
        (RED, "red", 5),
        (BLUE, "blue", 5),
        (RED | BLUE, "red_blue", 10),
        ({8, 32}, "flower", 3 if koi else 1),
        ({28, 32}, "moon", 3 if koi else 1),
    ):
        if group <= cards:
            add(key, group, points)
    for group, minimum, key in (
        (TANE, 5, "tane"),
        (RIBBONS, 5, "ribbons"),
        (KASU, 10, "kasu"),
    ):
        members = cards & group
        if len(members) >= minimum:
            add(key, members, len(members) - minimum + 1)
    if koi and koi <= 3:
        add("koi_bonus", [], koi)
    elif koi > 3:
        add("koi_multiplier", [], sum(r["points"] for r in rows) * (koi - 3))
    return rows


def card_value(card):
    return 20 if card in LIGHTS else 10 if card in TANE else 5 if card in RIBBONS else 1


def choose_dealer(rng):
    # Equal month and value: reveal another pair, rather than arbitrary seat priority.
    reveals = []
    while True:
        a, b = rng.sample(range(48), 2)
        ka, kb = (a // 4, -card_value(a)), (b // 4, -card_value(b))
        winner = -1 if ka == kb else int(kb < ka)
        reveals.append({"cards": [a, b], "winner": winner})
        if winner >= 0:
            return winner, reveals


class GameEnv(VariantEnv):
    """Use existing variant transitions with a twelve-round casual configuration.

    The variant research constructor deliberately fixes eight rounds. This local
    adapter initializes the reference engine directly, without changing that gate.
    Events contain only human/public information, never the packed engine state.
    """

    def __init__(self, seed, dealer, flags=0):
        if flags not in (0, OWN_KOI, TOTAL_YAKU, OWN_KOI | TOTAL_YAKU):
            raise ValueError("Unsupported game rules")
        self.events = []
        self.variant = Variant("desktop12", flags, rounds=12, bankrupt=False)
        ReferenceEnv.__init__(self, seed, "research12")
        self.state[DEALER] = self.state[PLAYER] = dealer
        self.events = []

    def view(self):
        s = self.state
        captures = [np.flatnonzero(s[:48] == 3 + p).tolist() for p in range(2)]
        # Slots are fixed for the entire round. Only seat zero's identities leave
        # the service; the opponent is represented by eight anonymous slot IDs.
        initial = s[48:56].tolist()
        return {
            "hand": [c if s[c] == 0 else -1 for c in initial],
            "opponent_slots": [i for i, c in enumerate(s[56:64]) if s[c] == 1],
            "field": np.flatnonzero(s[:48] == 2).tolist(),
            "captured": captures,
            "yaku": [yaku_rows(captures[p], int(s[KOIS + p])) for p in range(2)],
            "yaku_points": [self.yaku(p) for p in range(2)],
            "koi": s[KOIS : KOIS + 2].tolist(),
            "points": s[POINTS : POINTS + 2].tolist(),
            "dealer": int(s[DEALER]),
            "player": int(s[PLAYER]),
            "round": int(s[ROUND]) + 1,
            "phase": int(s[PHASE]),
            "pending": int(s[PENDING]),
            "stock_count": max(0, int(s[STOCK]) - 23),
            "legal": np.flatnonzero(self.legal()).tolist() if s[PLAYER] == 0 else [],
        }

    def emit(self, kind, **extra):
        self.events.append({"kind": kind, **extra, "view": self.view()})

    def capture(self, chosen=-1):
        card, player = int(self.state[PENDING]), int(self.state[PLAYER])
        matches = self.matches()
        taken = ([chosen] if len(matches) == 2 else matches) if matches else []
        before = self.yaku(player)
        super().capture(chosen)
        self.emit(
            "capture" if taken else "place",
            card=card,
            targets=taken,
            player=player,
            new_yaku=self.yaku(player) > before,
        )

    def draw(self, result):
        # Publish just the actual top card when the rules reveal it.
        card = int(self.state[48 + self.state[STOCK]])
        self.emit("draw", card=card, player=int(self.state[PLAYER]))
        super().draw(result)

    def _settle(self, winner, points):
        # Save the final table before VariantEnv automatically prepares a new deal.
        final_view = self.view()
        exhausted = (
            self.state[TURN] == 15
            and self.yaku(self.state[PLAYER]) <= self.state[BEFORE]
        )
        super()._settle(winner, points)
        final_view["points"] = self.state[POINTS : POINTS + 2].tolist()
        final_view["legal"] = []
        self.events.append(
            {
                "kind": "round_end",
                "winner": winner,
                "payment": points,
                "total_yaku": bool(self.variant.flags & TOTAL_YAKU) and not exhausted,
                "reason": "exhausted" if exhausted else "stop",
                "finished": bool(self.state[PHASE] == TERMINAL),
                "view": final_view,
            }
        )


class GameSession:
    def __init__(self, agent, seed=None, flags=0):
        seed = seed if seed is not None else secrets.randbelow(90_000_000) + 1
        dealer, self.reveals = choose_dealer(random.Random(seed))
        self.env = GameEnv(seed, dealer, flags)
        self.agent = agent
        self.paused = False
        self.end = None
        self.events = [
            {"kind": "opening", "reveals": self.reveals, "view": self.env.view()}
        ]

    def act(self, action=None):
        if self.paused or self.env.state[PHASE] == TERMINAL:
            raise ValueError("Round is over")
        player, phase = int(self.env.state[PLAYER]), int(self.env.state[PHASE])
        if (action is None) != (player == 1):
            raise ValueError("Wrong player")
        if action is None:
            action = int(self.agent.act(self.env))
        if (
            type(action) is not int
            or not 0 <= action < 50
            or not self.env.legal()[action]
        ):
            raise ValueError("Illegal action")
        self.env.events = []
        if phase == DISCARD:
            initial = self.env.state[48 + 8 * player : 56 + 8 * player].tolist()
            self.env.emit(
                "play", card=action, slot=initial.index(action), player=player
            )
        result = self.env.step(action)
        if phase == KOI and action == CONTINUE:
            self.env.emit("koi", player=player)
        self.events = self.env.events
        if result[1]:
            self.paused = True
            self.end = self.events[-1]

    def next_round(self):
        if not self.paused or self.env.state[PHASE] == TERMINAL:
            raise ValueError("No next round")
        self.paused, self.end = False, None
        self.events = [{"kind": "deal", "view": self.env.view()}]

    def response(self):
        return {
            "view": self.end["view"] if self.paused else self.env.view(),
            "events": self.events,
            "paused": self.paused,
            "finished": bool(self.env.state[PHASE] == TERMINAL),
            "end": self.end,
        }
