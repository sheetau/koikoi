"""Renewed starts agree with natural native boundaries and complete rollouts."""

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import (
    DEALER,
    DISCARD,
    EPISODE,
    PHASE,
    POINTS,
    ROUND,
    SEED,
    TERMINAL,
    TURN,
)
from koikoi.reference import ReferenceEnv, valid_deal
from scripts.continuation_chance import PublicContinuationContext, renew_deal


def reference(state):
    env = object.__new__(ReferenceEnv)
    env.state = state.copy()
    return env


def test_renewal_matches_naturally_dealt_native_rounds():
    env = BatchEnv(8, device="cpu", seed=301, profile="paper8", debug=True)
    rng = np.random.default_rng(193)
    seen = set()
    boundaries = 0
    for _ in range(448):
        states = env.state.numpy()
        for lane, row in enumerate(states):
            key = (lane, int(row[ROUND]))
            if (
                row[PHASE] == DISCARD
                and row[TURN] == 0
                and row[ROUND] > 0
                and key not in seen
            ):
                seen.add(key)
                context = PublicContinuationContext.from_root(row)
                rebuilt = renew_deal(context, int(row[SEED]), int(row[EPISODE]))
                np.testing.assert_array_equal(rebuilt, row)
                boundaries += 1
        if bool((env.state[:, PHASE] == TERMINAL).all()):
            break
        _, mask = env.observe()
        actions = [
            int(rng.choice(np.flatnonzero(m))) if m.any() else 48 for m in mask.numpy()
        ]
        env.step(torch.tensor(actions), freeze_done=True)
    assert bool((env.state[:, PHASE] == TERMINAL).all())
    assert boundaries >= 20
    assert {r for _, r in seen} == set(range(1, 8))


def test_public_context_ignores_old_cards_and_future_key():
    context = PublicContinuationContext(3, 1, (17, 43))
    old = renew_deal(context, 701)
    changed = old.copy()
    changed[:96] = -123
    changed[SEED], changed[EPISODE] = 999, 42
    np.testing.assert_array_equal(
        renew_deal(PublicContinuationContext.from_root(old), 702),
        renew_deal(PublicContinuationContext.from_root(changed), 702),
    )
    changed_deck = renew_deal(context, 702)
    assert not np.array_equal(old[48:96], changed_deck[48:96])
    assert changed_deck[DEALER] == 1
    np.testing.assert_array_equal(changed_deck[POINTS : POINTS + 2], [17, 43])
    assert valid_deal(changed_deck[48:96])
    np.testing.assert_array_equal(old, renew_deal(context, 701))
    with pytest.raises(ValueError):
        PublicContinuationContext.from_root(old.astype(np.int64))
    for bad in [
        (0, 0, (30, 30)),
        (8, 0, (30, 30)),
        (1, 2, (30, 30)),
        (1, 0, (0, 60)),
        (1, 0, (30, 29)),
    ]:
        with pytest.raises(ValueError):
            PublicContinuationContext(*bad)
    for bad in [-1, 2**31, 1.5, True]:
        with pytest.raises((ValueError, TypeError)):
            renew_deal(context, bad)
    old[TURN] = 1
    with pytest.raises(ValueError):
        PublicContinuationContext.from_root(old)


def test_renewed_multiround_rollouts_match_reference():
    contexts = [
        PublicContinuationContext(r, d, p)
        for r, d, p in [
            (1, 0, (30, 30)),
            (3, 1, (17, 43)),
            (5, 0, (59, 1)),
            (7, 1, (1, 59)),
        ]
    ]
    states = np.stack([renew_deal(c, 811 + i) for i, c in enumerate(contexts)])
    env = BatchEnv(4, device="cpu", seed=811, profile="paper8", debug=True)
    env.state.copy_(torch.from_numpy(states))
    refs = [reference(s) for s in states]
    rng = np.random.default_rng(197)
    rounds = np.zeros(4, dtype=int)
    completed = np.zeros(4, dtype=int)
    for _ in range(448):
        np.testing.assert_array_equal(
            env.state.numpy(), np.stack([r.state for r in refs])
        )
        obs, masks = env.observe()
        np.testing.assert_array_equal(
            obs.numpy(), np.stack([r.observe() for r in refs])
        )
        np.testing.assert_array_equal(
            masks.numpy(), np.stack([r.legal() for r in refs])
        )
        if all(r.state[PHASE] == TERMINAL for r in refs):
            break
        actions = [
            int(rng.choice(np.flatnonzero(r.legal())))
            if r.state[PHASE] != TERMINAL
            else 48
            for r in refs
        ]
        expected = np.stack(
            [
                r.step(a) if r.state[PHASE] != TERMINAL else [0, 0, 0, 0]
                for r, a in zip(refs, actions)
            ]
        )
        actual = env.step(torch.tensor(actions), freeze_done=True).numpy()
        np.testing.assert_array_equal(actual, expected)
        rounds += actual[:, 1]
        completed += actual[:, 2]
    assert all(r.state[PHASE] == TERMINAL for r in refs)
    np.testing.assert_array_equal(completed, np.ones(4, dtype=int))
    assert rounds[0] > 1 and rounds[3] == 1
