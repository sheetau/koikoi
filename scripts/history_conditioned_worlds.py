"""Exact finite-model history conditioning for bounded paper8 hand spaces.

Opponent models must be fixed deterministic observation-only policies. This is
Bayesian conditioning under that declared finite model class, not a claim that
the class contains an arbitrary real opponent. No candidate action is selected.
"""

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from koikoi.layout import BANKRUPT, DEALER, PHASE, PLAYER, ROUNDS, SEED, STOCK, TERMINAL
from scripts.diagnose_history_support import replay
from scripts.enumerate_history_support import canonical_world, context


def observation_key(reference):
    """Bind support to the root's own/public information, including seat labels."""
    state = reference.state
    return hashlib.sha256(
        reference.observe().tobytes()
        + np.asarray(state[[PLAYER, DEALER]], dtype=np.int32).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class HistorySupport:
    root_key: str
    model_names: tuple[str, ...]
    status: str
    candidate_hands: int
    valid_hands: int
    invalid_hands: int
    hands: tuple[tuple[int, ...], ...]
    # A hand can be compatible with multiple models; keep their joint mass.
    compatible: tuple[tuple[bool, ...], ...]
    counts: dict[str, int]


@torch.no_grad()
def enumerate_support(reference, opponents, max_hands=100):
    """Enumerate every eligible hand; never inspect the actual opponent hand.

    Replay applies all observed actions even after a model disagrees. The
    computation cap depends only on public combinatorics, before any inference.
    No existing diagnostic corpus or policy is selected implicitly.
    """
    if type(max_hands) is not int or max_hands < 1:
        raise ValueError("max_hands must be a positive integer")
    if not opponents or not all(isinstance(k, str) and k for k in opponents):
        raise ValueError("Supply named deterministic opponent models")
    state = reference.state
    if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
        raise ValueError("Expected a live paper8 root")
    names = tuple(sorted(opponents))
    unknown, count, _ = context(state)
    candidates = math.comb(len(unknown), count)
    root_key = observation_key(reference)
    if candidates > max_hands:
        return HistorySupport(
            root_key, names, "too_many_hands", candidates, 0, 0, (), (), {}
        )
    hands, compatible, counts = [], [], {}
    invalid = 0
    for hand in combinations(unknown, count):
        world = canonical_world(state, hand)
        if world is None:
            invalid += 1
            continue
        row = []
        for name in names:
            result = replay(world, opponents[name])
            row.append(bool(result["compatible"]))
            for key, value in result["counts"].items():
                counts[key] = counts.get(key, 0) + value
        hands.append(tuple(hand))
        compatible.append(tuple(row))
    assert len(hands) + invalid == candidates
    assert root_key == observation_key(reference)
    return HistorySupport(
        root_key,
        names,
        "complete",
        candidates,
        len(hands),
        invalid,
        tuple(hands),
        tuple(compatible),
        counts,
    )


def joint_probabilities(support, prior=None):
    """P(hand, model | observed history) under a shared uniform valid-hand prior.

    Each compatible pair has unnormalized mass equal to its model prior.
    Normalizing each model's hand set separately would incorrectly discard the
    history's evidence about which opponent model is more plausible.
    A supplied prior must precede this round's observations; feeding back the
    posterior of an earlier root in the same round would double-count evidence.
    """
    if support.status != "complete":
        raise ValueError("Support enumeration did not complete")
    n = len(support.model_names)
    if prior is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        if set(prior) != set(support.model_names):
            raise ValueError("Prior must name exactly the declared models")
        weights = np.array(
            [prior[name] for name in support.model_names], dtype=np.float64
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Model prior weights must be finite and positive")
    weights /= weights.max()
    matrix = np.asarray(support.compatible, dtype=bool).reshape(support.valid_hands, n)
    mass = matrix * weights[None, :]
    total = mass.sum()
    if total == 0:
        raise ValueError("No declared opponent model is compatible with this history")
    return mass / total


def sample_joint_worlds(reference, support, rng, count, prior=None):
    """Draw joint hand/model particles and fresh stock/future-deal randomness.

    Canonical stocks are used only to replay past evidence, never as future
    rollouts. Returned model indices must accompany their respective worlds.
    Empty or unavailable support is an explicit refusal, never silently uniform.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if observation_key(reference) != support.root_key:
        raise ValueError("Support belongs to a different public observation")
    mass = joint_probabilities(support, prior)
    keys = rng.choice(mass.size, size=count, p=mass.ravel())
    hand_indices, model_indices = np.divmod(keys, len(support.model_names))
    templates = {}
    worlds = []
    for hand_index in hand_indices:
        if int(hand_index) not in templates:
            template = canonical_world(reference.state, support.hands[hand_index])
            if template is None:
                raise ValueError("Support contains an invalid hand")
            templates[int(hand_index)] = template
        world = templates[int(hand_index)].copy()
        rng.shuffle(world[72 : 49 + int(world[STOCK])])
        world[SEED] = rng.integers(0, 2**31 - 1)
        worlds.append(world)
    return np.stack(worlds), model_indices.astype(np.int32)
