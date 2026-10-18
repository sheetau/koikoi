"""Online public stopping rule and strict complete-match continuation evaluation."""

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from scripts.check_public_memory_activation import eligible_boundary
from scripts.final_round_training import SnapshotAgent
from scripts.public_memory_continuation import ContinuationConfig
from scripts.public_memory_tail import candidate_code, condition_observation
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet


class ContinuationAgent:
    """Keep the parent until the first fresh round after public identification.

    The caller announces public counters before EVERY decision, including the
    opponent's. Opponent snapshots are submitted only after their action. This
    makes activation independent of which player deals the next round.
    """

    def __init__(self, path, predictors):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "public_memory_continuation_v1":
            raise ValueError("Expected public-memory continuation checkpoint")
        self.config = ContinuationConfig(**payload["config"])
        self.prefix = SnapshotAgent(self.config, payload["prefix_actor"])
        self.tail = SnapshotAgent(self.config, payload["actor"])
        if set(predictors) != {"wp", "point"}:
            raise ValueError("Expected both fixed candidate predictors")
        self.predictors = predictors
        self.tracker = None
        self.activation = self.context = None
        self.next_decision = 0

    def start_match(self, match_key, learner_seat):
        self.tracker = RevealedModelSet(match_key, learner_seat, self.predictors)
        self.activation = self.context = None
        self.next_decision = 0

    def begin_decision(self, *, match_key, decision, round_index, turn, phase):
        if self.tracker is None or match_key != self.tracker.match_key:
            raise ValueError("Start the correct match before announcing decisions")
        if any(type(x) is not int for x in (decision, round_index, turn, phase)):
            raise ValueError("Public counters must be integers")
        if (
            decision != self.next_decision
            or round_index not in range(8)
            or turn not in range(16)
            or phase not in range(4)
        ):
            raise ValueError("Invalid or nonsequential public decision")
        if decision == 0 and (round_index, turn, phase) != (0, 0, 0):
            raise ValueError("A match must start at its first public decision")
        if self.context is not None and round_index < self.context[0]:
            raise ValueError("Public rounds cannot go backwards")
        self.context = (round_index, turn, phase)
        self.next_decision += 1
        first = self.tracker.first_identification
        if self.activation is None and eligible_boundary(
            round_index, turn, phase, decision, first
        ):
            code = candidate_code(self.tracker.candidates)
            if code >= 0 and code == candidate_code({first["model"]}):
                self.activation = {
                    "decision": decision,
                    "round": round_index,
                    "candidate": first["model"],
                    "code": code,
                }

    @property
    def wants_observation(self):
        return (
            self.context is not None and self.activation is None and self.context[0] < 7
        )

    def observe_opponent(self, snapshot, action):
        if self.tracker is None or not self.wants_observation:
            raise ValueError("No opponent observation expected now")
        x = np.asarray(snapshot.features)
        if x.shape != (562,) or not np.isfinite(x).all():
            raise ValueError("Invalid public opponent snapshot")
        context = (
            round(float(x[534]) * 12),
            round(float(x[532]) * 16),
            int(np.argmax(x[528:532])),
        )
        if snapshot.decision != self.next_decision - 1 or context != self.context:
            raise ValueError("Opponent event does not match the announced decision")
        return self.tracker.observe(snapshot, action)

    @torch.no_grad()
    def act(self, env):
        if self.tracker is None or int(env.state[PLAYER]) != self.tracker.viewer:
            raise ValueError("Start the match and act on the learner turn")
        if self.context != tuple(int(env.state[k]) for k in (ROUND, TURN, PHASE)):
            raise ValueError("Announce the current public decision before acting")
        if self.activation is None:
            return self.prefix.act(env)
        solved = solved_koi_action(env)
        if solved is not None:
            return solved
        self.tail.published_env.state.copy_(torch.from_numpy(env.state[None]))
        obs, mask = packed_observation(self.tail.published_env)
        obs = condition_observation(
            obs,
            torch.tensor([self.activation["code"]]),
            self.config.public_memory,
        )
        logits, _, _ = self.tail.model(obs, mask)
        return int(logits.argmax(-1))


def play_continuation_match(case, actor, opponent, native):
    seat = case["seat"]
    if seat not in (0, 1):
        raise ValueError("Invalid learner seat")
    observes = isinstance(actor, ContinuationAgent)
    key = f"match-{case['seed']}-seat-{seat}"
    if observes:
        actor.start_match(key, seat)
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    actions = []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise RuntimeError("Exceeded match decision bound")
        if observes:
            actor.begin_decision(
                match_key=key,
                decision=len(actions),
                round_index=int(env.state[ROUND]),
                turn=int(env.state[TURN]),
                phase=int(env.state[PHASE]),
            )
        own = int(env.state[PLAYER]) == seat
        observe = observes and not own and actor.wants_observation
        if observe:
            snapshot = public_snapshot(
                env.observe(), 1 - seat, len(actions), match_key=key
            )
        action = int((actor if own else opponent).act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action]))
        np.testing.assert_array_equal(actual[0].cpu().numpy(), result)
        np.testing.assert_array_equal(native.state[0].cpu().numpy(), env.state)
        if observe:
            actor.observe_opponent(snapshot, action)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    return {
        **case,
        "actions": actions,
        "terminal_round": int(env.state[ROUND]),
        "terminal_points": env.state[POINTS : POINTS + 2].tolist(),
        "work": work,
        "memory": {
            "candidates": sorted(actor.tracker.candidates),
            "first_identification": actor.tracker.first_identification,
            "activation": actor.activation,
            "events": actor.tracker.events,
            "prediction_calls": actor.tracker.prediction_calls,
        }
        if observes
        else None,
    }
