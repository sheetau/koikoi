"""Retrospective actor inputs from public snapshots and eight public discards.

Reconstruction is available only after the hand revelation. This module does
not classify opponents, select a response policy or access simulator states.
"""

from dataclasses import dataclass

import numpy as np

from koikoi.reference import score
from scripts.check_seat_feature_reconstruction import reconstruct_actor


@dataclass(frozen=True)
class PublicActorSnapshot:
    features: np.ndarray
    decision: int
    match_key: str


def public_snapshot(observation, actor_seat, decision, *, match_key):
    """Erase BOTH hand membership channels; retain only public information.

    Input uses the acting player's ordinary observation layout. The resulting
    snapshot can be stored by either player: its private own-hand and unknown
    complement channels have been removed. Counts, scores, seat, board,
    captures, pending card and played-card histories are public.
    """
    obs = np.asarray(observation)
    if obs.shape != (560,) or not np.isfinite(obs).all():
        raise ValueError("Expected a finite ordinary observation")
    if actor_seat not in (0, 1) or type(decision) is not int or decision < 0:
        raise ValueError("Invalid public seat or decision address")
    if not isinstance(match_key, str) or not match_key:
        raise ValueError("A public match identity is required")
    features = np.zeros(562, dtype=np.float32)
    features[:560] = obs
    cards = features[:528].reshape(48, 11)
    cards[:, [0, 4]] = 0
    features[560:] = (1, actor_seat)
    return PublicActorSnapshot(features, decision, match_key)


def reconstruct_revealed(snapshot, revelation, available_decision):
    """Return raw actor features, legal mask and public-assisted legal mask.

    revelation must come from public events, e.g. revealed_hands(). Caller time
    may be in a later round, but it must not precede the eighth public discard.
    """
    x = np.asarray(snapshot.features)
    if x.shape != (562,) or not np.isfinite(x).all():
        raise ValueError("Invalid public snapshot")
    if np.any(x[:528].reshape(48, 11)[:, [0, 4]]):
        raise ValueError("Snapshot must have private hand channels erased")
    if x[560] != 1 or x[561] not in (0, 1):
        raise ValueError("Missing public actor seat")
    if revelation.get("match_key") != snapshot.match_key:
        raise ValueError("Revelation and snapshot belong to different matches")
    if revelation.get("actor_seat") != int(x[561]):
        raise ValueError("Revelation and snapshot name different actors")
    after = revelation["available_after_decision"]
    if (
        type(available_decision) is not int
        or type(after) is not int
        or available_decision < after
        or not 0 <= snapshot.decision < after
    ):
        raise ValueError("Hand revelation is not yet available for this snapshot")
    initial = revelation["initial_opponent_hand"]
    if (
        len(initial) != 8
        or len(set(initial)) != 8
        or any(type(c) is not int or c not in range(48) for c in initial)
    ):
        raise ValueError("Expected eight distinct publicly revealed hand cards")
    restored = x.copy()
    cards, g = restored[:528].reshape(48, 11), restored[528:560]
    if round(float(g[6]) * 12) != revelation["round"]:
        raise ValueError("Revelation and snapshot belong to different rounds")
    if not np.isclose(g[7] * 12, 8) or g[15] != 1:
        raise ValueError("Expected paper8 snapshot")
    phase = int(np.argmax(g[:4]))
    if not np.array_equal(g[:4], np.eye(4, dtype=np.float32)[phase]):
        raise ValueError("Expected a live decision phase")
    actor = int(x[561])
    dealer = actor if g[5] == 1 else 1 - actor
    current_turn = round(float(g[4]) * 16)
    used = set()
    for card in np.flatnonzero(cards[:, 7]):
        turn = round(float(cards[card, 7]) * 16) - 1
        if not 0 <= turn <= current_turn:
            raise ValueError("Invalid public history timestamp")
        if (dealer + turn) % 2 == actor:
            used.add(int(card))
    hand = set(initial)
    if not used <= hand:
        raise ValueError("Revealed hand contradicts earlier public discards")
    hand -= used
    visible = set(np.flatnonzero(cards[:, [1, 2, 3, 5]].any(axis=1)))
    if hand & visible:
        raise ValueError("Remaining hand overlaps publicly exposed cards")
    cards[list(hand), 0] = 1
    cards[list(set(range(48)) - visible - hand), 4] = 1
    features = reconstruct_actor(restored)
    legal = np.zeros(50, dtype=bool)
    if phase == 0:
        legal[list(hand)] = True
    elif phase in (1, 2):
        pending = np.flatnonzero(cards[:, 5])
        if len(pending) != 1:
            raise ValueError("Expected one public pending card")
        legal[:48] = cards[:, 1].astype(bool) & (np.arange(48) // 4 == pending[0] // 4)
    else:
        legal[48:] = True
    if not legal.any():
        raise ValueError("Reconstructed decision has no legal action")
    assisted = legal.copy()
    if phase == 3:
        points = score(np.flatnonzero(cards[:, 2]), round(float(g[10]) * 8))
        own_after = round(float(g[8]) * 64) + points
        other_after = round(float(g[9]) * 64) - points
        solved = None
        if other_after <= 0:
            solved = 48
        elif round(float(g[6]) * 12) == 7:
            solved = (
                48
                if own_after > other_after
                else 49
                if own_after < other_after
                else None
            )
        if solved is not None:
            assisted[:] = False
            assisted[solved] = True
    return features, legal, assisted
