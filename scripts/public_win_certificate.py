"""Conservative, observation-only certificates for wins within the current turn.

The unknown-card set overapproximates possible next draws. Hand capture choices
precede the universal draw quantifier; draw capture choices follow it. Apply the
wrapper at EVERY subsequent decision to execute the certified continuation.
This research wrapper does not modify the simulator or the deployed policy.
"""

from dataclasses import dataclass

import numpy as np

from koikoi.layout import DISCARD, KOI, PICK_DRAW, PICK_HAND, STOP
from koikoi.reference import score


@dataclass(frozen=True)
class WinCertificate:
    actions: tuple[int, ...]
    draw_cases: int
    leaf_checks: int


def _captures(field, taken, card):
    matches = field & frozenset(range(card // 4 * 4, card // 4 * 4 + 4))
    if not matches:
        return ((field | {card}, taken),)
    choices = ({c} for c in sorted(matches)) if len(matches) == 2 else (matches,)
    return tuple((field - choice, taken | choice | {card}) for choice in choices)


def certify_turn(observation, legal) -> WinCertificate:
    """Return legal first actions with an observation-based immediate-win plan.

    Inputs must be a valid engine observation (560) and legal mask (50).
    An empty certificate is inconclusive, not proof that winning is impossible.
    No sampling, hidden state, opponent model or value function is used.
    """
    obs = np.asarray(observation)
    mask = np.asarray(legal, dtype=bool)
    if obs.shape != (560,) or mask.shape != (50,) or not np.isfinite(obs).all():
        raise ValueError("Expected a finite public observation and legal mask")
    if not mask.any():
        return WinCertificate((), 0, 0)
    cards, g = obs[:528].reshape(48, 11), obs[528:]
    phase = int(g[:4].argmax())
    final = round(float(g[6]) * 12) + 1 == round(float(g[7]) * 12)
    bankrupt = bool(g[15])
    if not final and not bankrupt:
        return WinCertificate((), 0, 0)
    field = frozenset(map(int, np.flatnonzero(cards[:, 1])))
    taken = frozenset(map(int, np.flatnonzero(cards[:, 2])))
    unknown = tuple(map(int, np.flatnonzero(cards[:, 4])))
    own, other = (round(float(g[i]) * 64) for i in (8, 9))
    koi = round(float(g[10]) * 8)
    turn = round(float(g[4]) * 16)
    before = score(taken, koi) if phase == DISCARD else round(float(g[14]) * 32)
    draw_cases = leaf_checks = 0

    def winning_finish(points, own_winner=True):
        delta = points if own_winner else -points
        own_after, other_after = own + delta, other - delta
        terminal = final or (bankrupt and min(own_after, other_after) <= 0)
        return terminal and own_after > other_after

    def after_draw(captured):
        nonlocal leaf_checks
        leaf_checks += 1
        points = score(captured, koi)
        if points > before:
            # Either automatic stop on turns 14/15, or a certified STOP next.
            return winning_finish(points)
        # No increase on the last turn awards one point to the initial dealer.
        return turn == 15 and winning_finish(1, bool(g[5]))

    def all_draws(next_field, captured):
        nonlocal draw_cases
        if not unknown:
            return False  # No vacuous certificate from an invalid empty stock.
        for card in unknown:
            draw_cases += 1
            if not any(after_draw(c) for _, c in _captures(next_field, captured, card)):
                return False
        return True

    actions = []
    if phase in (PICK_HAND, PICK_DRAW):
        pending = np.flatnonzero(cards[:, 5])
        if len(pending) != 1:
            raise ValueError("Capture phase requires one public pending card")
        pending = int(pending[0])
    for action in map(int, np.flatnonzero(mask)):
        if phase == DISCARD:
            wins = any(all_draws(f, c) for f, c in _captures(field, taken, action))
        elif phase in (PICK_HAND, PICK_DRAW):
            next_field, captured = field - {action}, taken | {pending, action}
            wins = (
                all_draws(next_field, captured)
                if phase == PICK_HAND
                else after_draw(captured)
            )
        elif phase == KOI:
            leaf_checks += 1
            wins = action == STOP and winning_finish(score(taken, koi))
        else:
            wins = False
        if wins:
            actions.append(action)
    return WinCertificate(tuple(actions), draw_cases, leaf_checks)


class CertifiedWinAgent:
    """Keep the base action if certified; otherwise choose the first certificate.

    Always invoke the base, preserving its call/RNG schedule until the first
    deviation. The win guarantee requires using this wrapper throughout a match.
    """

    def __init__(self, base):
        self.base = base
        self.certified_decisions = 0
        self.overrides = 0
        self.draw_cases = 0
        self.leaf_checks = 0

    def act(self, env):
        legal = env.legal()
        base_action = int(self.base.act(env))
        if not 0 <= base_action < 50 or not legal[base_action]:
            raise ValueError("Base policy returned an illegal action")
        certificate = certify_turn(env.observe(), legal)
        self.draw_cases += certificate.draw_cases
        self.leaf_checks += certificate.leaf_checks
        if not certificate.actions:
            return base_action
        self.certified_decisions += 1
        if base_action in certificate.actions:
            return base_action
        self.overrides += 1
        return certificate.actions[0]
