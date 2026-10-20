"""Exact tail planning that expands stock chance only when a draw is reached.

Each row holds a hand/model and the integer multiplicity of its as-yet-unseen
draw prefixes. Decisions still share one action per observable history.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from koikoi.layout import (
    BANKRUPT,
    DISCARD,
    EPISODE,
    KOI,
    PHASE,
    PICK_DRAW,
    PICK_HAND,
    PLAYER,
    ROUND,
    ROUNDS,
    SEED,
    STOCK,
    TERMINAL,
    TURN,
)
from scripts.belief_tail import (
    FiniteBeliefPlanner,
    TailBudgetExceeded,
    TailPlan,
    WorldSet,
)
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import observation_key


@dataclass
class DeferredWorldSet(WorldSet):
    """Remaining stocks are uniform sets, not deterministic future orders."""


def deferred_worlds(root, support, prior=None, max_draws=5, max_hands=1000):
    """Construct complete current-hand hypotheses without any future draws."""
    state = root.state
    if (
        state[PHASE] == TERMINAL
        or state[ROUNDS] != 8
        or not state[BANKRUPT]
        or state[ROUND] != 7
    ):
        raise ValueError("Expected a live final-round paper8 root")
    if support.status != "complete" or support.root_key != observation_key(root):
        raise ValueError("Complete support for this root is required")
    if type(max_draws) is not int or max_draws < 0:
        raise ValueError("Invalid draw cap")
    if type(max_hands) is not int or max_hands < 1:
        raise ValueError("Invalid hand cap")
    draws = 16 - int(state[TURN]) - int(state[PHASE] in (PICK_DRAW, KOI))
    if draws > max_draws:
        raise TailBudgetExceeded("Public remaining-draw cap exceeded")
    unknown, count, _ = context(state)
    candidate_count = math.comb(len(unknown), count)
    if candidate_count > max_hands:
        raise TailBudgetExceeded("Public candidate-hand cap exceeded")
    templates = {
        hand: world
        for hand in combinations(unknown, count)
        if (world := canonical_world(state, hand)) is not None
    }
    if (
        not support.model_names
        or len(set(support.model_names)) != len(support.model_names)
        or len(set(support.hands)) != len(support.hands)
        or set(support.hands) != set(templates)
        or support.candidate_hands != candidate_count
        or support.valid_hands != len(templates)
        or support.invalid_hands != candidate_count - len(templates)
        or len(support.compatible) != len(support.hands)
        or any(
            len(row) != len(support.model_names)
            or any(type(value) is not bool for value in row)
            for row in support.compatible
        )
    ):
        raise ValueError("Malformed or incomplete support")
    prior = dict.fromkeys(support.model_names, 1) if prior is None else prior
    if set(prior) != set(support.model_names) or any(
        type(weight) is not int or weight < 0 for weight in prior.values()
    ):
        raise ValueError("Expected nonnegative integer model prior weights")
    stock_count = len(unknown) - count
    if stock_count - draws != 8:
        raise ValueError("Unexpected final-round stock count")
    multiplicity = math.perm(stock_count, draws)
    states, models, weights = [], [], []
    for hand, row in zip(support.hands, support.compatible, strict=True):
        for model, name in enumerate(support.model_names):
            if row[model] and prior[name]:
                world = templates[hand].copy()
                world[SEED] = world[EPISODE] = 0
                states.append(world)
                models.append(model)
                weights.append(prior[name] * multiplicity)
    if not states:
        raise ValueError("No compatible positive-prior model")
    if sum(weights) >= 2**63:
        raise ValueError("Exact integer mass exceeds int64 capacity")
    return DeferredWorldSet(
        np.stack(states),
        np.asarray(models, dtype=np.int64),
        np.asarray(weights, dtype=np.int64),
        support.model_names,
        int(state[PLAYER]),
        draws,
    )


def expand_draws(states, models, weights, actions, max_rows):
    """Distribute a row's mass uniformly across each next stock card.

    This mirrors only whether a transition needs a draw. The native/reference
    engines still decide captures, yaku, scores, termination and legal actions.
    """
    indices, positions = [], []
    for index, (state, action) in enumerate(zip(states, actions, strict=True)):
        draws = state[PHASE] == PICK_HAND or (
            state[PHASE] == DISCARD
            and np.count_nonzero(state[(action // 4) * 4 : (action // 4) * 4 + 4] == 2)
            != 2
        )
        count = int(state[STOCK]) - 23 if draws else 1
        if draws and (count < 9 or int(weights[index]) % count):
            raise ValueError("Invalid remaining-prefix multiplicity at a draw")
        if len(indices) + count > max_rows:
            raise TailBudgetExceeded("Expanded chance frontier cap exceeded")
        indices.extend([index] * count)
        positions.extend(range(72, 72 + count) if draws else [-1])
    indices = np.asarray(indices, dtype=np.int64)
    expanded = states[indices].copy()
    masses = weights[indices].copy()
    for i, position in enumerate(positions):
        if position >= 0:
            top = 48 + int(expanded[i, STOCK])
            expanded[i, position], expanded[i, top] = (
                expanded[i, top],
                expanded[i, position],
            )
            masses[i] //= int(expanded[i, STOCK]) - 23
    assert sum(map(int, masses)) == sum(map(int, weights))
    return expanded, models[indices], masses, actions[indices]


class DeferredBeliefPlanner(FiniteBeliefPlanner):
    """Same finite information tree as the eager solver, with deferred chance."""

    def transition(self, states, models, weights, actions):
        expanded, models, weights, actions = expand_draws(
            states, models, weights, actions, self.dynamics.capacity
        )
        self.work["chance_rows_added"] += len(expanded) - len(states)
        self.work["maximum_frontier_rows"] = max(
            self.work["maximum_frontier_rows"], len(expanded)
        )
        after = self.advance(expanded, actions)
        terminal = self.dynamics.terminal(after)
        self.work["represented_decisions"] += sum(map(int, weights))
        ended = sum(map(int, weights[terminal]))
        self.work["represented_round_endings"] += ended
        self.work["represented_completions"] += ended
        return after, models, weights

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
                raise ValueError("Merged observable terminal/acting-player states")
            if self.dynamics.players(s)[0] != self.viewer:
                successor = self.transition(
                    s, m, w, self.dynamics.opponent_actions(s, m)
                )
                value, tail = self.continuation(*successor, depth + 1)
            else:
                self.work["own_decision_nodes"] += 1
                legal = self.dynamics.legal(s[0])
                fallback = self.dynamics.baseline_action(s[0])
                if fallback not in legal:
                    raise ValueError("Illegal baseline action")
                values, policies = {}, {}
                for action in legal if self.optimize else (fallback,):
                    successor = self.transition(
                        s, m, w, np.full(len(s), action, dtype=np.int32)
                    )
                    values[action], policies[action] = self.continuation(
                        *successor, depth + 1
                    )
                best = max(values.values())
                action = (
                    fallback
                    if values.get(fallback) == best
                    else next(a for a in legal if values.get(a) == best)
                )
                value, tail = values[action], policies[action]
                tail[key] = action
            total += value
            self.merge(policy, tail)
        return total, policy

    def solve(self, worlds, optimize_continuation=True):
        if not isinstance(worlds, DeferredWorldSet):
            raise TypeError("Deferred solver requires unordered-stock worlds")
        states, models, weights = worlds.states, worlds.models, worlds.weights
        if (
            states.ndim != 2
            or not len(states)
            or models.shape != (len(states),)
            or models.dtype != np.int64
            or weights.shape != (len(states),)
            or weights.dtype != np.int64
            or np.any(weights <= 0)
            or sum(map(int, weights)) >= 2**63
        ):
            raise ValueError("Expected positive integer hypothesis masses")
        self.viewer, self.optimize = worlds.viewer, bool(optimize_continuation)
        keys = self.dynamics.keys(states, self.viewer)
        if (
            len(set(keys)) != 1
            or self.dynamics.terminal(states).any()
            or not np.all(self.dynamics.players(states) == self.viewer)
        ):
            raise ValueError("Root must be one live learner information set")
        self.work = dict.fromkeys(
            (
                "decisions",
                "round_endings",
                "completions",
                "observation_nodes",
                "own_decision_nodes",
                "chance_rows_added",
                "maximum_frontier_rows",
                "represented_decisions",
                "represented_round_endings",
                "represented_completions",
            ),
            0,
        )
        legal = self.dynamics.legal(states[0])
        fallback = self.dynamics.baseline_action(states[0])
        if fallback not in legal:
            raise ValueError("Illegal root baseline")
        values, policies = {}, {}
        for action in legal:
            successor = self.transition(
                states, models, weights, np.full(len(states), action, dtype=np.int32)
            )
            values[action], policies[action] = self.continuation(*successor, 1)
        best = max(values.values())
        action = (
            fallback
            if values[fallback] == best
            else next(a for a in legal if values[a] == best)
        )
        for first, policy in policies.items():
            policy[keys[0]] = first
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
