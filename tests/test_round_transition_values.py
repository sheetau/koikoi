import itertools
from dataclasses import asdict

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import DEALER, HISTORY, PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from koikoi.published_actor import PublishedActor
from koikoi.reference import valid_deal
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import MixtureContinuation
from scripts.round_transition_values import (
    RoundWorkspace,
    backup_round,
    backward_values,
    fresh_round_states,
)
from scripts.run_arbitration_study import PARENTS


def test_fresh_context_grid_keeps_deals_coupled_and_seats_explicit():
    contexts = np.array([[3, 1, 0], [3, 59, 1], [7, 30, 1]])
    states = fresh_round_states(contexts, [32110001] * 3)
    np.testing.assert_array_equal(states[0, 48:96], states[1, 48:96])
    for state, (round_index, points, dealer) in zip(states, contexts):
        assert state[ROUND] == round_index
        assert state[PLAYER] == state[DEALER] == dealer
        assert state[TURN] == state[PHASE] == 0
        assert state[POINTS : POINTS + 2].tolist() == [points, 60 - points]
        assert not state[HISTORY:].any() and valid_deal(state[48:96])
    for bad in ([[8, 30, 0]], [[0, 0, 0]], [[0, 60, 1]], [[0, 30, 2]], [[0.0, 30, 0]]):
        with pytest.raises(ValueError):
            fresh_round_states(bad, [1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("player", [0, 1])
def test_first_round_endpoints_match_reference_and_freeze(device, player):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = NeuralAgent(PARENTS[0][0], deterministic=True)
    actor = PublishedActor(parent.model.actor.checkpoint(), device)
    continuation = MixtureContinuation(actor, device)
    continuation.prepare(player, [0, 1, 0])
    contexts = [[0, 30, 0], [2, 1, 1], [7, 30, 0], [7, 59, 1], [6, 59, 0], [4, 1, 0]]
    states = fresh_round_states(contexts, np.arange(32120001, 32120007))
    refs = [reference(s) for s in states]
    done = [False] * 6
    counts = {"decisions": 0, "round_endings": 0, "terminal_match_endings": 0}
    first = [int(np.flatnonzero(ref.legal())[-1]) for ref in refs]

    def advance(actions):
        for i, (ref, action) in enumerate(zip(refs, actions)):
            if done[i]:
                continue
            result = ref.step(int(action))
            done[i] = bool(result[1])
            counts["decisions"] += 1
            counts["round_endings"] += int(result[1])
            counts["terminal_match_endings"] += int(result[2])

    advance(first)
    saw_frozen = []

    def checked(env):
        features, masks = env.published_observe()
        for i, ref in enumerate(refs):
            expected = ref.state.copy()
            if done[i]:
                expected[PHASE] = TERMINAL
                assert not features[i].any() and not masks[i].any()
                saw_frozen.append(i)
            np.testing.assert_array_equal(env.state[i].cpu().numpy(), expected)
        actions = continuation(env)
        advance(actions.tolist())
        return actions

    workspace = RoundWorkspace(6, device)
    tensor = torch.from_numpy(states).to(device)
    endpoints, work = workspace.run(tensor, checked, torch.tensor(first, device=device))
    np.testing.assert_array_equal(tensor.cpu().numpy(), states)
    np.testing.assert_array_equal(
        endpoints.cpu().numpy(), np.stack([r.state for r in refs])
    )
    assert asdict(work) == counts and work.round_endings == 6
    assert 0 < work.terminal_match_endings < 6 and saw_frozen
    for state, original in zip(endpoints.cpu().numpy(), states):
        if state[PHASE] != TERMINAL:
            assert state[ROUND] == original[ROUND] + 1 and state[TURN] == 0
    torch.testing.assert_close(workspace.env.state, endpoints, rtol=0, atol=0)
    # No-first-action mode must start the policy at the root and retain endpoints.
    direct = RoundWorkspace(6, device)
    result, used = direct.run(tensor, continuation)
    assert used.round_endings == 6 and result.shape == tensor.shape


def test_kernel_backup_matches_exhaustive_path_expectations():
    shape = (8, 2, 2, 59, 2, 2)
    points, dealers = np.empty(shape, np.int32), np.empty(shape, np.int32)
    terminal = np.empty(shape, bool)
    for r, player, model, p, dealer, draw in itertools.product(
        range(8), range(2), range(2), range(1, 60), range(2), range(2)
    ):
        delta = ((-1 - dealer, 1 + model)[draw]) * (1 - 2 * player)
        after = p + delta
        key = (r, player, model, p - 1, dealer, draw)
        points[key], dealers[key] = after, int(delta < 0)
        terminal[key] = r == 7 or not 1 <= after <= 59
    values = backward_values(points, dealers, terminal)
    for r, player, model, p, dealer in itertools.product(
        (0, 3, 7), range(2), range(2), (1, 15, 29, 30, 31, 45, 59), range(2)
    ):
        outcomes = []
        for draws in itertools.product((0, 1), repeat=8 - r):
            score, d = p, dealer
            for draw in draws:
                amount = (-1 - d) if draw == 0 else (1 + model)
                amount *= 1 - 2 * player
                score += amount
                d = int(amount < 0)
                if score <= 0 or score >= 60:
                    break
            outcomes.append(np.sign(score - 30) * (1 - 2 * player))
        assert values[r, player, model, p - 1, dealer] == np.mean(outcomes)
    broken = terminal.copy()
    broken[7, 0, 0, 0, 0, 0] = False
    with pytest.raises(ValueError, match="eighth round"):
        backward_values(points, dealers, broken)


def test_terminal_draw_and_bankruptcy_override_lookup_and_invalid_live_refused():
    points = np.zeros((59, 2, 3), np.int32)
    points[:] = [-4, 30, 64]
    dealers = np.zeros_like(points)
    terminal = np.ones_like(points, dtype=bool)
    np.testing.assert_array_equal(
        backup_round(points, dealers, terminal, np.ones((59, 2)), 0), 0
    )
    terminal[0, 0, 0] = False
    with pytest.raises(ValueError, match="live point"):
        backup_round(points, dealers, terminal, np.zeros((59, 2)), 0)
