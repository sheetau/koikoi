"""Complete one strict-policy match with its previously selected single root action."""

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.midround_search import is_midround


def play_intervention(case, parent, opponent, native, expected_root, selected_action):
    if (expected_root is None) != (selected_action is None):
        raise ValueError("Supply both expected root and action, or neither")
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    actions, intervention_indices = [], []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        assert len(actions) < 1024
        ours = int(env.state[PLAYER]) == case["seat"]
        if ours and is_midround(env):
            assert expected_root is not None and not intervention_indices
            np.testing.assert_array_equal(env.state, expected_root)
            action = int(selected_action)
            assert env.legal()[action]
            intervention_indices.append(len(actions))
        else:
            action = int((parent if ours else opponent).act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action]))
        np.testing.assert_array_equal(actual[0].numpy(), result)
        np.testing.assert_array_equal(native.state[0].numpy(), env.state)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    assert len(intervention_indices) == int(expected_root is not None)
    return {
        **case,
        "actions": actions,
        "intervention_indices": intervention_indices,
        "selected_action": selected_action,
        "work": work,
        "terminal_points": env.state[POINTS : POINTS + 2].tolist(),
    }
