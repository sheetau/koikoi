"""Exact zero-likelihood pruning for deterministic, stateless opponent models.

For a valid current-round history, a disagreeing model/hand pair has likelihood
zero. Later factors cannot restore it. Replay the observed actions once per
hand and stop invoking each model at its first disagreement. This changes work,
not the joint posterior. It is not valid for stochastic/stateful policy APIs.
"""

import math
from itertools import combinations

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import (
    BANKRUPT,
    HISTORY,
    KOI,
    PHASE,
    PLAYER,
    ROUND,
    ROUNDS,
    STOP,
    TERMINAL,
    TURN,
)
from scripts.diagnose_history_support import reference as make_reference
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import HistorySupport, observation_key


@torch.no_grad()
def replay_joint(target, opponents):
    """Score a canonical valid hand; policies receive ordinary acting observations.

    This routine assumes a legal recorded history, just like canonical_world.
    An all-rejected prefix is not a validator for its unvisited suffix. Surviving
    pairs always replay to the root and verify the complete reconstructed state.
    Counts distinguish shared transitions from individual model invocations.
    """
    target = np.asarray(target, dtype=np.int32)
    env = make_reference(target)
    env.deal(deck=target[48:96].copy())
    active = [True] * len(opponents)
    counts = dict.fromkeys(
        ("replay_decisions", "candidate_decisions", "opponent_decisions",
         "model_calls", "guarded_model_calls", "unguarded_model_calls",
         "rejected_models", "pruned_hands", "completed_hands"), 0,
    )
    while (env.state[TURN], env.state[PHASE]) != (target[TURN], target[PHASE]):
        assert env.state[ROUND] == target[ROUND]
        assert counts["replay_decisions"] < 64
        phase, turn = int(env.state[PHASE]), int(env.state[TURN])
        encoded = int(target[HISTORY + turn * 5 + (0, 1, 3, 4)[phase]])
        assert encoded > 0
        observed = STOP + encoded - 1 if phase == KOI else encoded - 1
        assert env.legal()[observed]
        if env.state[PLAYER] != target[PLAYER]:
            counts["opponent_decisions"] += 1
            guarded = solved_koi_action(env) is not None
            for i, opponent in enumerate(opponents):
                if active[i]:
                    predicted = opponent.act(env)
                    assert env.legal()[predicted]
                    counts["model_calls"] += 1
                    counts["guarded_model_calls" if guarded else
                           "unguarded_model_calls"] += 1
                    if predicted != observed:
                        active[i] = False
                        counts["rejected_models"] += 1
        else:
            counts["candidate_decisions"] += 1
        result = env.step(observed)
        assert not result[1], "Historical replay crossed the current round"
        counts["replay_decisions"] += 1
        if not any(active):
            counts["pruned_hands"] = 1
            return tuple(active), counts
    np.testing.assert_array_equal(env.state, target)
    counts["completed_hands"] = 1
    return tuple(active), counts


@torch.no_grad()
def enumerate_pruned_support(reference, opponents, max_hands=100):
    """Same support ordering and public cap as full enumeration, less replay."""
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
            root_key, names, "too_many_hands", candidates, 0, 0, (), (), {},
        )
    models = tuple(opponents[name] for name in names)
    hands, compatible, counts = [], [], {}
    invalid = 0
    for hand in combinations(unknown, count):
        world = canonical_world(state, hand)
        if world is None:
            invalid += 1
            continue
        row, work = replay_joint(world, models)
        for key, value in work.items():
            counts[key] = counts.get(key, 0) + value
        hands.append(tuple(hand))
        compatible.append(row)
    assert len(hands) + invalid == candidates
    assert root_key == observation_key(reference)
    return HistorySupport(
        root_key, names, "complete", candidates, len(hands), invalid,
        tuple(hands), tuple(compatible), counts,
    )
