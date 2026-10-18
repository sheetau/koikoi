"""Fixed-budget hand proposals conditioned on deterministic opponent actions.

Large spaces use uniform proposals with replacement and retain their multiplicity.
Small spaces are enumerated exactly. This component estimates beliefs only;it
does not select an action or read the real hidden hand as likelihood evidence.
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from koikoi.layout import BANKRUPT, PHASE, ROUNDS, TERMINAL
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import observation_key
from scripts.pruned_history_support import replay_joint


@dataclass(frozen=True)
class HistoryProposalBatch:
    root_key: str
    model_names: tuple[str, ...]
    mode: str
    candidate_hands: int
    proposed_draws: int
    invalid_draws: int
    hands: tuple[tuple[int, ...], ...]
    compatible: tuple[tuple[bool, ...], ...]
    counts: dict[str, int]


@torch.no_grad()
def propose_history_support(reference, opponents, rng, *, max_draws=256):
    """Fixed public proposal budget,with no retries after rejection/zero support.

    Candidate models and any later prior must precede this round's evidence.
    A prior-round public identification can supply a singleton model dictionary;
    the caller owns that public provenance,not this sampler.
    """
    if type(max_draws) is not int or max_draws < 1:
        raise ValueError("Expected positive integer max_draws")
    if not opponents or not all(isinstance(name, str) and name for name in opponents):
        raise ValueError("Supply named deterministic opponent models")
    state = reference.state
    if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
        raise ValueError("Expected live paper8 root")
    unknown, size, _ = context(state)
    candidates = math.comb(len(unknown), size)
    key = observation_key(reference)
    names = tuple(sorted(opponents))
    models = tuple(opponents[name] for name in names)
    if candidates <= max_draws:
        mode, draws = "enumerated", candidates
        proposals = combinations(unknown, size)
    else:
        mode, draws = "sampled", max_draws
        proposals = (
            tuple(sorted(int(c) for c in rng.choice(unknown, size, replace=False)))
            for _ in range(draws)
        )
    hands, compatible, work = [], [], {}
    invalid = 0
    for hand in proposals:
        world = canonical_world(state, hand)
        if world is None:
            invalid += 1
            continue
        matched, counts = replay_joint(world, models)
        hands.append(tuple(hand))
        compatible.append(matched)
        for name, count in counts.items():
            work[name] = work.get(name, 0) + count
    assert len(hands) + invalid == draws
    assert observation_key(reference) == key
    return HistoryProposalBatch(
        key,
        names,
        mode,
        candidates,
        draws,
        invalid,
        tuple(hands),
        tuple(compatible),
        work,
    )


def proposal_probabilities(batch, prior=None):
    """Joint mass over proposal occurrences and models,not deduplicated hands.

    This is exact only in enumerated mode. In sampled mode it is a normalized
    importance estimate under a uniform valid-hand prior. Zero sampled mass is
    an explicit failure,not proof that the exact posterior is empty.
    """
    if batch.mode not in ("enumerated", "sampled"):
        raise ValueError("Unknown proposal mode")
    count = len(batch.model_names)
    if prior is None:
        weights = np.ones(count)
    else:
        if set(prior) != set(batch.model_names):
            raise ValueError("Prior must name exactly the declared models")
        weights = np.array([prior[name] for name in batch.model_names], dtype=float)
    if not count or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Prior weights must be finite and positive")
    weights /= weights.max()
    compatible = np.asarray(batch.compatible, dtype=bool).reshape(
        len(batch.hands), count
    )
    mass = compatible * weights[None, :]
    if mass.sum() == 0:
        raise ValueError("Zero compatible proposal mass")
    return mass / mass.sum()


def marginal_summary(batch, prior=None):
    joint = proposal_probabilities(batch, prior)
    proposal_mass = joint.sum(axis=1)
    cards = np.zeros(48)
    for hand, weight in zip(batch.hands, proposal_mass):
        cards[list(hand)] += weight
    supported = [hand for hand, mass in zip(batch.hands, proposal_mass) if mass > 0]
    return {
        "card_probabilities": cards.tolist(),
        "model_probabilities": dict(zip(batch.model_names, joint.sum(axis=0).tolist())),
        "effective_proposals": float(1 / np.square(proposal_mass).sum()),
        "supported_proposal_occurrences": len(supported),
        "distinct_supported_hands": len(set(supported)),
        "exact_enumeration": batch.mode == "enumerated",
    }
