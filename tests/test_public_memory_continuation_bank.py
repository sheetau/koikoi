"""Natural source replay, public stopping times and suffix independence."""

import copy

import numpy as np
import pytest

from koikoi.layout import PHASE, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.build_public_memory_continuation_bank import extract_root
from tests.test_final_round_actual_match import StopFirst


def history(alternate_suffix=False):
    env = ReferenceEnv(31010001, "paper8")
    actor = StopFirst()
    actions, boundaries = [], []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        previous_round = int(env.state[ROUND])
        action = actor.act(env)
        if alternate_suffix and boundaries:
            action = int(np.flatnonzero(env.legal())[-1])
        result = env.step(action)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
        if int(env.state[ROUND]) != previous_round and env.state[PHASE] != TERMINAL:
            boundaries.append((len(actions), previous_round, env.state.copy()))
    return {
        "seed": 31010001,
        "actions": actions,
        "endpoint": env.state.tolist(),
        "work": work,
    }, boundaries


def test_root_is_first_natural_boundary_and_independent_of_suffix():
    row, boundaries = history()
    decision, previous_round, expected = boundaries[0]
    first = {
        "round": previous_round,
        "available_after_decision": decision,
        "model": "point",
    }
    root, event, work = extract_root(row, first)
    np.testing.assert_array_equal(root, expected)
    assert event == {
        "decision": decision,
        "round": previous_round + 1,
        "candidate": "point",
    }
    assert work == row["work"]
    changed, _ = history(alternate_suffix=True)
    assert changed["actions"] != row["actions"]
    other_root, other_event, _ = extract_root(changed, first)
    np.testing.assert_array_equal(other_root, root)
    assert other_event == event
    # Future or absent public identification cannot select any earlier boundary.
    future = {**first, "available_after_decision": len(row["actions"]) + 1}
    for unavailable in (None, future):
        absent, event, replay = extract_root(row, unavailable)
        assert absent is None and event is None and replay == work
    # A public identification in a later round selects that round's successor.
    decision, previous_round, expected = boundaries[2]
    root, event, _ = extract_root(
        row,
        {
            "round": previous_round,
            "available_after_decision": decision,
            "model": "wp",
        },
    )
    np.testing.assert_array_equal(root, expected)
    assert event["candidate"] == "wp"


def test_replay_rejects_corrupt_source_and_post_terminal_actions():
    row, _ = history()
    corrupted = copy.deepcopy(row)
    corrupted["endpoint"][0] ^= 1
    with pytest.raises(AssertionError):
        extract_root(corrupted, None)
    corrupted = copy.deepcopy(row)
    corrupted["work"]["completed_matches"] += 1
    with pytest.raises(ValueError, match="work"):
        extract_root(corrupted, None)
    corrupted = copy.deepcopy(row)
    corrupted["actions"].append(0)
    with pytest.raises(ValueError, match="after match termination"):
        extract_root(corrupted, None)
