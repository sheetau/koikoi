"""Finite, observation-consistent tail planning under deterministic models.

The planner optimizes one common action per visible history. Stock prefixes and
opponent hypotheses are internal simulator worlds, never independent decision
makers. Exactness is conditional on the supplied finite model and root belief.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import (
    BANKRUPT,
    EPISODE,
    HISTORY,
    KOI,
    PHASE,
    PICK_DRAW,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    SEED,
    STOCK,
    TERMINAL,
    TURN,
)
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import observation_key


class TailBudgetExceeded(RuntimeError):
    """An incomplete tree must not be reported as an exact solution."""


def visible_keys(states, viewer):
    """Exact integer own/public histories, including absolute seat identities."""
    states = np.asarray(states, dtype=np.int32)
    columns = np.r_[0:48, 96:110, HISTORY:208, 64:72]
    public = states[:, columns].copy()
    hidden = (public[:, :48] == -1) | (public[:, :48] == 1 - viewer)
    public[:, :48][hidden] = -1
    own_initial = np.sort(states[:, 48 + 8 * viewer : 56 + 8 * viewer], axis=1)
    counts = (states[:, :48] == 1 - viewer).sum(1).astype(np.int32)[:, None]
    seats = np.full((len(states), 1), viewer, dtype=np.int32)
    public = np.concatenate((seats, public, own_initial, counts), axis=1)
    return [row.tobytes() for row in public]


@dataclass
class WorldSet:
    states: np.ndarray
    models: np.ndarray
    weights: np.ndarray
    model_names: tuple[str, ...]
    viewer: int
    draws: int


def enumerate_tail_worlds(
    reference_env, support, prior=None, max_draws=3, max_worlds=50000
):
    """Enumerate every future draw prefix; unused stock permutations cancel.

    The support must be the complete current-round deterministic likelihood
    table. Integer prior weights precede that round's observations; zeros may
    represent an already identified model from an earlier round.
    """
    state = reference_env.state
    if (
        state[PHASE] == TERMINAL
        or state[ROUNDS] != 8
        or not state[BANKRUPT]
        or state[ROUND] != state[ROUNDS] - 1
    ):
        raise ValueError("Expected a live final-round paper8 root")
    if support.status != "complete" or support.root_key != observation_key(
        reference_env
    ):
        raise ValueError("Complete support for this public root is required")
    if type(max_draws) is not int or max_draws < 0:
        raise ValueError("Invalid draw cap")
    if type(max_worlds) is not int or max_worlds < 1:
        raise ValueError("Invalid world cap")
    draws = 16 - int(state[TURN]) - int(state[PHASE] in (PICK_DRAW, KOI))
    if draws > max_draws:
        raise TailBudgetExceeded("Public remaining-draw cap exceeded")
    unknown, count, _ = context(state)
    templates = {}
    invalid = 0
    for hand in combinations(unknown, count):
        world = canonical_world(state, hand)
        if world is None:
            invalid += 1
        else:
            templates[hand] = world
    if (
        len(support.model_names) != len(set(support.model_names))
        or not support.model_names
        or len(support.hands) != len(set(support.hands))
        or set(support.hands) != set(templates)
        or support.candidate_hands != math.comb(len(unknown), count)
        or support.valid_hands != len(templates)
        or support.invalid_hands != invalid
        or len(support.compatible) != len(support.hands)
        or any(
            len(row) != len(support.model_names)
            or any(type(value) is not bool for value in row)
            for row in support.compatible
        )
    ):
        raise ValueError("Malformed or incomplete support table")
    prior = dict.fromkeys(support.model_names, 1) if prior is None else prior
    if set(prior) != set(support.model_names) or any(
        type(value) is not int or value < 0 for value in prior.values()
    ):
        raise ValueError("Prior must contain nonnegative integer model weights")
    pairs = [
        (hand, model, prior[name])
        for hand, row in zip(support.hands, support.compatible, strict=True)
        for model, name in enumerate(support.model_names)
        if row[model] and prior[name] > 0
    ]
    if not pairs:
        raise ValueError("No compatible positive-prior model")
    stock_count = len(unknown) - count
    prefixes = math.perm(stock_count, draws)
    world_count = len(pairs) * prefixes
    if world_count > max_worlds:
        raise TailBudgetExceeded("Exact stock-prefix world cap exceeded")
    if prefixes * sum(weight for _, _, weight in pairs) >= 2**63:
        raise ValueError("Exact integer mass exceeds int64 capacity")
    worlds, models, weights = [], [], []
    for hand, model, weight in pairs:
        template = templates[hand]
        stock = sorted(set(unknown) - set(hand))
        assert stock_count == int(template[STOCK]) - 23
        for prefix in permutations(stock, draws):
            world = template.copy()
            remainder = sorted(set(stock) - set(prefix))
            world[72 : 49 + int(world[STOCK])] = remainder + list(reversed(prefix))
            world[SEED] = world[EPISODE] = 0
            worlds.append(world)
            models.append(model)
            weights.append(weight)
    return WorldSet(
        np.stack(worlds),
        np.asarray(models, dtype=np.int64),
        np.asarray(weights, dtype=np.int64),
        support.model_names,
        int(state[PLAYER]),
        draws,
    )


class KoiTailDynamics:
    """Native CPU batched transitions and strict, observation-only policy adapters."""

    def __init__(self, capacity, model_names, opponents, baseline):
        if set(model_names) != set(opponents):
            raise ValueError("Every declared opponent model needs a policy")
        self.model_names, self.opponents, self.baseline = (
            model_names,
            opponents,
            baseline,
        )
        self.env = BatchEnv(capacity, "cpu", debug=True)
        self.storage = self.env.state
        self.capacity = capacity
        self.policy_cache = {}
        self.baseline_cache = {}
        self.policy_calls = self.baseline_calls = 0

    keys = staticmethod(visible_keys)

    @staticmethod
    def players(states):
        return states[:, PLAYER]

    @staticmethod
    def terminal(states):
        return states[:, PHASE] == TERMINAL

    @staticmethod
    def outcomes(states, viewer):
        return np.sign(states[:, POINTS + viewer] - states[:, POINTS + 1 - viewer])

    @staticmethod
    def legal(state):
        return tuple(int(action) for action in np.flatnonzero(reference(state).legal()))

    def baseline_action(self, state):
        key = visible_keys(state[None], int(state[PLAYER]))[0]
        if key not in self.baseline_cache:
            legal = self.legal(state)
            self.baseline_cache[key] = (
                legal[0]
                if len(legal) == 1
                else int(self.baseline.act(reference(state)))
            )
            self.baseline_calls += int(len(legal) > 1)
        return self.baseline_cache[key]

    def opponent_actions(self, states, models):
        groups = defaultdict(list)
        if len(set(states[:, PLAYER])) != 1:
            raise ValueError("Opponent node mixes acting players")
        keys = visible_keys(states, int(states[0, PLAYER]))
        for i, (visible, model) in enumerate(zip(keys, models, strict=True)):
            key = (int(model), visible)
            groups[key].append(i)
        actions = np.empty(len(states), dtype=np.int32)
        for key, indices in groups.items():
            if key not in self.policy_cache:
                env = reference(states[indices[0]])
                legal = tuple(np.flatnonzero(env.legal()))
                self.policy_cache[key] = (
                    int(legal[0])
                    if len(legal) == 1
                    else int(self.opponents[self.model_names[key[0]]].act(env))
                )
                self.policy_calls += int(len(legal) > 1)
                if self.policy_cache[key] not in legal:
                    raise ValueError("Opponent chose an illegal action")
            actions[indices] = self.policy_cache[key]
        return actions

    def advance(self, states, actions):
        if len(states) > self.capacity:
            raise TailBudgetExceeded("Native workspace capacity exceeded")
        env = self.env
        env.count = len(states)
        env.state = self.storage[: len(states)]
        env.state.copy_(torch.from_numpy(np.ascontiguousarray(states)))
        result = env.step(
            torch.from_numpy(np.asarray(actions, dtype=np.int32)), freeze_done=True
        )
        after = env.state.numpy().copy()
        if not np.array_equal(states[:, ROUND], after[:, ROUND]):
            raise ValueError("Tail crossed a round boundary")
        return after, {
            "decisions": len(states),
            "round_endings": int(result[:, 1].sum()),
            "completions": int(result[:, 2].sum()),
        }


@dataclass
class TailPlan:
    action: int
    baseline_action: int
    values: dict[int, Fraction]
    # Complete chosen contingent policy; keep it for the rest of this tail.
    policy: dict[bytes, int]
    root_policies: dict[int, dict[bytes, int]]
    work: dict[str, int]
    optimized_continuation: bool


class FiniteBeliefPlanner:
    def __init__(self, dynamics, max_nodes=100000, max_depth=32):
        if (
            type(max_nodes) is not int
            or max_nodes < 1
            or type(max_depth) is not int
            or max_depth < 1
        ):
            raise ValueError("Positive integer tree budgets required")
        self.dynamics, self.max_nodes, self.max_depth = dynamics, max_nodes, max_depth

    @staticmethod
    def merge(target, addition):
        for key, action in addition.items():
            if key in target and target[key] != action:
                raise ValueError("Conflicting actions at one visible history")
            target[key] = action

    def advance(self, states, actions):
        after, work = self.dynamics.advance(states, actions)
        for key, count in work.items():
            self.work[key] += count
        return after

    def continuation(self, states, models, weights, depth):
        if depth > self.max_depth:
            raise TailBudgetExceeded("Decision depth cap exceeded")
        groups = defaultdict(list)
        for index, key in enumerate(self.dynamics.keys(states, self.viewer)):
            groups[key].append(index)
        total, policy = 0, {}
        for key, indices in groups.items():
            self.work["observation_nodes"] += 1
            if self.work["observation_nodes"] > self.max_nodes:
                raise TailBudgetExceeded("Observation-node cap exceeded")
            s, m, w = states[indices], models[indices], weights[indices]
            terminal = self.dynamics.terminal(s)
            if terminal.all():
                total += int(np.dot(w, self.dynamics.outcomes(s, self.viewer)))
                continue
            if terminal.any() or len(set(self.dynamics.players(s))) != 1:
                raise ValueError("Observable terminal/acting-player states were merged")
            if self.dynamics.players(s)[0] != self.viewer:
                after = self.advance(s, self.dynamics.opponent_actions(s, m))
                value, tail = self.continuation(after, m, w, depth + 1)
            else:
                self.work["own_decision_nodes"] += 1
                legal = self.dynamics.legal(s[0])
                fallback = self.dynamics.baseline_action(s[0])
                if fallback not in legal:
                    raise ValueError("Baseline chose an illegal action")
                values, policies = {}, {}
                for action in legal if self.optimize else (fallback,):
                    after = self.advance(s, np.full(len(s), action, dtype=np.int32))
                    values[action], policies[action] = self.continuation(
                        after, m, w, depth + 1
                    )
                best = max(values.values())
                action = (
                    fallback
                    if values.get(fallback) == best
                    else next(action for action in legal if values.get(action) == best)
                )
                value, tail = values[action], policies[action]
                tail[key] = action
            total += value
            self.merge(policy, tail)
        return total, policy

    def solve(self, worlds, optimize_continuation=True):
        states, models, weights = worlds.states, worlds.models, worlds.weights
        if (
            states.ndim != 2
            or not len(states)
            or models.shape != (len(states),)
            or weights.shape != (len(states),)
            or weights.dtype != np.int64
            or np.any(weights <= 0)
            or sum(map(int, weights)) >= 2**63
        ):
            raise ValueError("Expected positive exact integer world masses")
        self.viewer, self.optimize = worlds.viewer, bool(optimize_continuation)
        keys = self.dynamics.keys(states, self.viewer)
        if (
            len(set(keys)) != 1
            or self.dynamics.terminal(states).any()
            or not np.all(self.dynamics.players(states) == self.viewer)
        ):
            raise ValueError("Root worlds must share one live learner information set")
        self.work = dict.fromkeys(
            (
                "decisions",
                "round_endings",
                "completions",
                "observation_nodes",
                "own_decision_nodes",
            ),
            0,
        )
        legal = self.dynamics.legal(states[0])
        fallback = self.dynamics.baseline_action(states[0])
        if fallback not in legal:
            raise ValueError("Illegal root baseline")
        values, policies = {}, {}
        for action in legal:
            after = self.advance(states, np.full(len(states), action, dtype=np.int32))
            values[action], policies[action] = self.continuation(
                after, models, weights, 1
            )
        best = max(values.values())
        action = (
            fallback
            if values[fallback] == best
            else next(a for a in legal if values[a] == best)
        )
        for first_action, policy in policies.items():
            policy[keys[0]] = first_action
        mass = sum(map(int, weights))
        return TailPlan(
            action,
            fallback,
            {a: Fraction(value + mass, 2 * mass) for a, value in values.items()},
            policies[action],
            policies,
            dict(self.work),
            self.optimize,
        )
