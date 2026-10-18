"""Strict complete-match evaluation with an explicit public-history observer."""

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.revealed_opponent_inputs import public_snapshot


def play_memory_match(case, actor, opponent, native):
    seat = case["seat"]
    if seat not in (0, 1):
        raise ValueError("Invalid learner seat")
    observes = hasattr(actor, "start_match")
    key = f"match-{case['seed']}-seat-{seat}"
    if observes:
        actor.start_match(key, seat)
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    actions, prefix_length = [], None
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise RuntimeError("Exceeded match decision bound")
        if env.state[ROUND] == 7 and prefix_length is None:
            prefix_length = len(actions)
        own = int(env.state[PLAYER]) == seat
        observe = observes and not own and env.state[ROUND] < 7
        if observe:
            snapshot = public_snapshot(
                env.observe(), 1 - seat, len(actions), match_key=key
            )
        action = int((actor if own else opponent).act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action]))
        np.testing.assert_array_equal(actual[0].cpu().numpy(), result)
        np.testing.assert_array_equal(native.state[0].cpu().numpy(), env.state)
        # Even if the action crosses a round boundary, submit the pre-action
        # public snapshot only after the action has been executed and revealed.
        if observe:
            actor.observe_opponent(snapshot, action)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    return {
        **case,
        "actions": actions,
        "prefix_length": prefix_length,
        "terminal_round": int(env.state[ROUND]),
        "terminal_points": env.state[POINTS : POINTS + 2].tolist(),
        "work": work,
        "memory": {
            "candidates": sorted(actor.tracker.candidates),
            "tail_code": actor.tail_code,
            "events": actor.tracker.events,
            "prediction_calls": actor.tracker.prediction_calls,
        }
        if observes
        else None,
    }
