"""Bounded global/local proposals for one publicly selected deterministic model.

Rejection initialization gives a uniform supported hand. Both subsequent kernels
are symmetric, so accepted moves preserve that distribution. Chain states are
correlated: their count is not an effective independent sample size.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from koikoi.layout import BANKRUPT, PHASE, ROUNDS, TERMINAL
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import observation_key
from scripts.pruned_history_support import replay_joint


@dataclass(frozen=True)
class HandChain:
    status: str
    attempts: int
    first_supported_attempt: int | None
    global_proposals: int
    local_proposals: int
    accepted_proposals: int
    changed_hands: int
    cache_hits: int
    states: tuple[tuple[int, ...], ...]
    checked: tuple[tuple[tuple[int, ...], bool], ...]


def draw_hand_chain(
    pool, size, compatible, rng, *, proposals=128, global_probability=0.2
):
    """One fixed attempt budget, including initialization and all rejections.

    ``compatible`` must be a deterministic Boolean function of a proposed hand.
    Its results are cached only within this call. Failure to initialize returns
    no estimate; it does not imply an empty exact posterior.
    """
    if type(proposals) is not int or proposals < 1:
        raise ValueError("Expected positive integer proposal budget")
    if isinstance(global_probability, bool) or not (
        math.isfinite(global_probability) and 0 < global_probability <= 1
    ):
        raise ValueError("Global probability must be in (0, 1]")
    pool = tuple(sorted(pool))
    if len(set(pool)) != len(pool) or not all(
        isinstance(c, (int, np.integer)) and not isinstance(c, bool) and 0 <= c < 48
        for c in pool
    ):
        raise ValueError("Expected unique card indices")
    if type(size) is not int or not 0 < size < len(pool):
        raise ValueError("Expected a nonempty proper-subset hand size")
    current, first = None, None
    cache, states = {}, []
    global_count = local_count = accepted = changed = hits = 0
    for attempt in range(1, proposals + 1):
        if current is None or rng.random() < global_probability:
            candidate = tuple(
                sorted(int(c) for c in rng.choice(pool, size, replace=False))
            )
            global_count += 1
        else:
            outgoing = int(rng.choice(current))
            incoming = int(rng.choice(tuple(c for c in pool if c not in current)))
            candidate = tuple(sorted((set(current) - {outgoing}) | {incoming}))
            local_count += 1
        if candidate in cache:
            hits += 1
        else:
            value = compatible(candidate)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("Compatibility must be Boolean")
            cache[candidate] = bool(value)
        if cache[candidate]:
            accepted += 1
            changed += int(current is not None and current != candidate)
            current = candidate
            if first is None:
                first = attempt
        if current is not None:
            # Holding after a rejected move is essential for stationarity.
            states.append(current)
    return HandChain(
        "sampled" if states else "no_supported_proposal",
        proposals,
        first,
        global_count,
        local_count,
        accepted,
        changed,
        hits,
        tuple(states),
        tuple(cache.items()),
    )


def chain_marginals(chain):
    if not chain.states:
        raise ValueError("No supported proposal within budget")
    cards = np.zeros(48)
    for hand in chain.states:
        cards[list(hand)] += 1
    return {
        "card_probabilities": (cards / len(chain.states)).tolist(),
        "chain_states": len(chain.states),
        "distinct_hands": len(set(chain.states)),
        "independent_sample_size": None,
    }


@torch.no_grad()
def sample_public_hand_chain(
    reference, opponents, rng, *, proposals=128, global_probability=0.2
):
    """Use a singleton model selected before this round's evidence.

    The caller must establish public prior-round provenance. Routing labels or
    the true hidden hand must not select the model. This function selects no
    game action and never supplies chain simulator states to a future rollout.
    """
    if len(opponents) != 1 or not all(isinstance(k, str) and k for k in opponents):
        raise ValueError("Expected one named publicly selected opponent model")
    state = reference.state
    if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
        raise ValueError("Expected live paper8 root")
    key = observation_key(reference)
    unknown, size, _ = context(state)
    name, model = next(iter(opponents.items()))
    work = {"invalid_hand_checks": 0}

    def compatible(hand):
        world = canonical_world(state, hand)
        if world is None:
            work["invalid_hand_checks"] += 1
            return False
        matches, counts = replay_joint(world, (model,))
        for counter, value in counts.items():
            work[counter] = work.get(counter, 0) + value
        return matches[0]

    chain = draw_hand_chain(
        unknown,
        size,
        compatible,
        rng,
        proposals=proposals,
        global_probability=global_probability,
    )
    assert key == observation_key(reference)
    return {"root_key": key, "model_name": name, "chain": chain, "history_work": work}
