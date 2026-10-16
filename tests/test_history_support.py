"""Historical-support diagnostics must retain observed trajectories and empty sets."""

import numpy as np
import pytest

from koikoi.layout import HISTORY, PHASE, PLAYER, TURN
from koikoi.reference import ReferenceEnv
from scripts.diagnose_history_support import conditional, replay


class RecordedOpponent:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def act(self, env):
        key = int(env.state[TURN]), int(env.state[PHASE])
        self.calls.append(key)
        return self.actions[key]


def history():
    env = ReferenceEnv(7)
    actions, alternatives, players = {}, {}, {}
    while env.state[TURN] < 6:
        key = int(env.state[TURN]), int(env.state[PHASE])
        legal = np.flatnonzero(env.legal())
        actions[key], alternatives[key] = int(legal[-1]), int(legal[0])
        players[key] = int(env.state[PLAYER])
        assert env.step(actions[key])[1] == 0
    return env.state.copy(), actions, alternatives, players


def test_actual_replay_scores_only_opponent_actions():
    target, actions, _, players = history()
    agent = RecordedOpponent(actions)
    result = replay(target, agent)
    expected = [key for key in actions if players[key] != target[PLAYER]]
    assert agent.calls == expected
    assert result["compatible"] and result["first_mismatch"] is None
    assert result["counts"]["replay_decisions"] == len(actions)
    assert result["counts"]["opponent_decisions"] == len(expected)


def test_mismatch_does_not_stop_or_change_observed_replay():
    target, actions, alternatives, players = history()
    expected = [key for key in actions if players[key] != target[PLAYER]]
    changed = next(key for key in expected if actions[key] != alternatives[key])
    predictions = actions | {changed: alternatives[changed]}
    agent = RecordedOpponent(predictions)
    result = replay(target, agent)
    assert agent.calls == expected
    assert not result["compatible"]
    assert result["first_mismatch"] == expected.index(changed)
    assert sum(e["observed"] != e["predicted"] for e in result["events"]) == 1
    assert result["counts"]["replay_decisions"] == len(actions)


def test_empty_history_is_compatible_without_teacher_calls():
    agent = RecordedOpponent({})
    result = replay(ReferenceEnv(7).state, agent)
    assert result["compatible"]
    assert not agent.calls and not result["events"]
    assert result["counts"]["replay_decisions"] == 0


def test_missing_history_rejected():
    target, actions, _, _ = history()
    target[HISTORY] = 0
    with pytest.raises(AssertionError):
        replay(target, RecordedOpponent(actions))


def test_empty_support_is_null_and_single_survivor_uses_only_its_values():
    values = [[1, -1], [-1, 1], [0, -1], [1, 0]]
    assert conditional(values, [False, False]) is None
    result = conditional(values, [False, True])
    assert result == {
        "arm_means": [-1.0, 1.0, -1.0, 0.0],
        "self_gain": 2.0,
        "published_gain": 1.0,
        "gain_shift": -1.0,
    }
