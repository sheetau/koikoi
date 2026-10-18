import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import reconstruct
from scripts.check_public_hand_revelation import revealed_hands
from scripts.revealed_opponent_inputs import public_snapshot, reconstruct_revealed
from tests.test_final_round_actual_match import ContinueFirst, StopFirst


def fixtures():
    result = []
    for seed in range(31010001, 31010013):
        env = ReferenceEnv(seed, "paper8")
        states, events = [], []
        while env.state[PHASE] != TERMINAL:
            states.append(env.state.copy())
            # Continued rounds expose both hand history and koi decisions.
            actor = ContinueFirst() if seed % 2 else StopFirst()
            action = actor.act(env)
            events.append(
                (
                    int(env.state[ROUND]),
                    int(env.state[PHASE]),
                    int(env.state[PLAYER]),
                    action,
                )
            )
            env.step(action)
        for viewer in (0, 1):
            reveals = revealed_hands(events, viewer)
            for reveal in reveals:
                reveal = {
                    **reveal,
                    "match_key": f"fixture-{seed}",
                    "actor_seat": 1 - viewer,
                }
                selected = [
                    (i, s)
                    for i, s in enumerate(states)
                    if s[ROUND] == reveal["round"]
                    and s[PLAYER] != viewer
                    and i < reveal["available_after_decision"]
                ]
                result.extend((i, s, reveal) for i, s in selected)
    return result


def test_exact_all_phase_actor_features_and_masks_from_revealed_hand():
    native = BatchEnv(1, "cpu", "paper8")
    seen = set()
    examples = fixtures()
    assert examples
    for decision, state, reveal in examples:
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state
        snapshot = public_snapshot(
            ref.observe(), int(state[PLAYER]), decision, match_key=reveal["match_key"]
        )
        actual, legal, assisted = reconstruct_revealed(
            snapshot, reveal, reveal["available_after_decision"]
        )
        native.state.copy_(torch.from_numpy(state[None]))
        expected, expected_legal = native.published_observe()
        np.testing.assert_array_equal(actual, expected[0].numpy())
        # The source opponents use strict upstream CPU inputs, not an assumed
        # numerical equivalence to a native or dynamically batched policy.
        strict = reconstruct(ref).feature_tensor.numpy()
        np.testing.assert_array_equal(actual[:, : strict.shape[1]], strict)
        np.testing.assert_array_equal(legal, expected_legal[0].numpy())
        solved = int(native.solved_actions()[0])
        if solved >= 0:
            expected_legal.zero_()
            expected_legal[0, solved] = True
        np.testing.assert_array_equal(assisted, expected_legal[0].numpy())
        seen.add((int(state[PLAYER]), int(state[PHASE])))
    assert seen == {(seat, phase) for seat in (0, 1) for phase in range(4)}


def test_hidden_reassignment_invariance_and_time_round_refusal():
    decision, state, reveal = next(x for x in fixtures() if x[1][PHASE] == 0)
    ref = ReferenceEnv.__new__(ReferenceEnv)
    ref.state = state.copy()
    actor = int(state[PLAYER])
    snapshot = public_snapshot(
        ref.observe(), actor, decision, match_key=reveal["match_key"]
    )
    # Change either private hand assignment and unobserved stock order. Public
    # snapshots must be identical, even though actor-private observations differ.
    for seat in (0, 1):
        altered = state.copy()
        hand = int(np.flatnonzero(altered[:48] == seat)[0])
        hidden = int(np.flatnonzero(altered[:48] == -1)[0])
        altered[hand], altered[hidden] = altered[hidden], altered[hand]
        a, b = [
            48 + int(np.flatnonzero(altered[48:96] == c)[0]) for c in (hand, hidden)
        ]
        altered[a], altered[b] = altered[b], altered[a]
        ref.state = altered
        other = public_snapshot(
            ref.observe(), actor, decision, match_key=reveal["match_key"]
        )
        np.testing.assert_array_equal(snapshot.features, other.features)
        decoded = reconstruct_revealed(
            other, reveal, reveal["available_after_decision"]
        )
        expected = reconstruct_revealed(
            snapshot, reveal, reveal["available_after_decision"]
        )
        for left, right in zip(decoded, expected):
            np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="not yet available"):
        reconstruct_revealed(snapshot, reveal, reveal["available_after_decision"] - 1)
    with pytest.raises(ValueError, match="different rounds"):
        reconstruct_revealed(
            snapshot,
            {**reveal, "round": reveal["round"] + 1},
            reveal["available_after_decision"],
        )
    with pytest.raises(ValueError, match="eight distinct"):
        reconstruct_revealed(
            snapshot,
            {**reveal, "initial_opponent_hand": [0] * 8},
            reveal["available_after_decision"],
        )

    with pytest.raises(ValueError, match="different matches"):
        reconstruct_revealed(
            snapshot,
            {**reveal, "match_key": "other-match"},
            reveal["available_after_decision"],
        )
    with pytest.raises(ValueError, match="different actors"):
        reconstruct_revealed(
            snapshot,
            {**reveal, "actor_seat": 1 - actor},
            reveal["available_after_decision"],
        )
