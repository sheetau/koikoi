"""Collect natural final-round roots and replay a single declared intervention.

Packed roots are verification artifacts only. Actors receive the ordinary
reference environment through their existing observation-only adapters.
This helper selects no root and defines no strength experiment.
"""

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv


def eligible(env, seat):
    return bool(
        env.state[ROUND] == 7
        and env.state[PLAYER] == seat
        and env.state[PHASE] in (0, 1, 2)
        and np.count_nonzero(env.legal()) > 1
    )


def play_final_round_match(case, parent, opponent, native, *, intervention=None):
    """Return all encountered eligible roots and a complete-match record.

    intervention, when supplied, is (decision_index, expected_state, action).
    The full state is checked only by this evaluator, never a policy input.
    Missing opportunities are valid baseline matches, not excluded samples.
    """
    seat = case["seat"]
    if seat not in (0, 1):
        raise ValueError("Learner seat must be zero or one")
    if intervention is not None:
        index, expected, selected = intervention
        if not isinstance(index, int) or index < 0:
            raise ValueError("Intervention index must be a nonnegative integer")
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    roots, actions, applied = [], [], []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise AssertionError("Match exceeded the decision bound")
        candidate = eligible(env, seat)
        if candidate:
            roots.append((len(actions), env.state.copy()))
        if intervention is not None and len(actions) == index:
            if not candidate:
                raise ValueError("Intervention must be an unforced final-round card")
            np.testing.assert_array_equal(env.state, expected)
            action = int(selected)
            if not 0 <= action < 50 or not env.legal()[action]:
                raise ValueError("Intervention action is illegal")
            applied.append(len(actions))
        else:
            actor = parent if env.state[PLAYER] == seat else opponent
            action = int(actor.act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action]))
        np.testing.assert_array_equal(actual[0].cpu().numpy(), result)
        np.testing.assert_array_equal(native.state[0].cpu().numpy(), env.state)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    if intervention is not None and applied != [index]:
        raise ValueError("Declared intervention was never reached")
    return roots, {
        **case,
        "actions": actions,
        "eligible_decision_indices": [index for index, _ in roots],
        "intervention_indices": applied,
        "selected_action": int(selected) if intervention is not None else None,
        "terminal_round": int(env.state[ROUND]),
        "terminal_points": env.state[POINTS : POINTS + 2].tolist(),
        "work": work,
    }
