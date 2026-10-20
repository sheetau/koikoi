"""Exact parent-first action search with the terminal utility ceiling."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from scripts.belief_tail import TailBudgetExceeded, TailPlan
from scripts.deferred_belief_tail import DeferredWorldSet
from scripts.terminal_reduced_belief_tail import TerminalReducedPlanner


@dataclass
class CeilingPlan(TailPlan):
    unevaluated_actions: tuple[int, ...]
    root_ceiling_attained: bool


class CeilingBeliefPlanner(TerminalReducedPlanner):
    """Never assign an exact value to a skipped action.

    Nonnegative integer masses and signed terminal utilities in [-1, 1] imply
    value <= total mass. Parent-first ordering preserves the existing tie rule.
    """

    def choose(self, states, models, weights, depth, root=False):
        legal = self.dynamics.legal(states[0])
        fallback = self.dynamics.baseline_action(states[0])
        if fallback not in legal:
            raise ValueError("Illegal baseline action")
        order = (
            (fallback, *[a for a in legal if a != fallback])
            if root or self.optimize
            else (fallback,)
        )
        mass = sum(map(int, weights))
        values, policies = {}, {}
        best_action = fallback
        for index, action in enumerate(order):
            successor = self.transition(
                states, models, weights, np.full(len(states), action, dtype=np.int32)
            )
            value, policy = self.continuation(*successor, depth + 1)
            if not -mass <= value <= mass:
                raise ValueError("Continuation violates the signed utility bound")
            values[action], policies[action] = value, policy
            if value > values[best_action]:
                best_action = action
            if value == mass:
                skipped = len(order) - index - 1
                self.work["ceiling_cutoffs"] += int(skipped > 0)
                self.work["pruned_actions"] += skipped
                break
        return best_action, fallback, legal, values, policies

    def continuation(self, states, models, weights, depth):
        if depth > self.max_depth:
            raise TailBudgetExceeded("Decision depth cap exceeded")
        terminal = self.dynamics.terminal(states)
        self.work["terminal_rows_valued"] += int(terminal.sum())
        self.work["terminal_mass_valued"] += sum(map(int, weights[terminal]))
        total = (
            int(
                np.dot(
                    weights[terminal],
                    self.dynamics.outcomes(states[terminal], self.viewer),
                )
            )
            if terminal.any()
            else 0
        )
        if terminal.all():
            return total, {}
        states, models, weights = (
            states[~terminal],
            models[~terminal],
            weights[~terminal],
        )
        groups = defaultdict(list)
        for index, key in enumerate(self.dynamics.keys(states, self.viewer)):
            groups[key].append(index)
        policy = {}
        for key, indices in groups.items():
            self.work["live_observation_nodes"] += 1
            if self.work["live_observation_nodes"] > self.max_nodes:
                raise TailBudgetExceeded("Live observation-node cap exceeded")
            s, m, w = states[indices], models[indices], weights[indices]
            if len(set(self.dynamics.players(s))) != 1:
                raise ValueError("Merged acting-player states")
            if self.dynamics.players(s)[0] != self.viewer:
                successor = self.transition(
                    s, m, w, self.dynamics.opponent_actions(s, m)
                )
                value, tail = self.continuation(*successor, depth + 1)
            else:
                self.work["own_decision_nodes"] += 1
                action, _, _, values, policies = self.choose(s, m, w, depth)
                value, tail = values[action], policies[action]
                tail[key] = action
            total += value
            self.merge(policy, tail)
        return total, policy

    def solve(self, worlds, optimize_continuation=True):
        self.work = {}
        if not isinstance(worlds, DeferredWorldSet):
            raise TypeError("Ceiling solver requires unordered-stock worlds")
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
                "live_observation_nodes",
                "own_decision_nodes",
                "chance_rows_added",
                "maximum_frontier_rows",
                "represented_decisions",
                "represented_round_endings",
                "represented_completions",
                "terminal_rows_valued",
                "terminal_mass_valued",
                "ceiling_cutoffs",
                "pruned_actions",
            ),
            0,
        )
        action, fallback, legal, values, policies = self.choose(
            states, models, weights, 0, root=True
        )
        for first, policy in policies.items():
            policy[keys[0]] = first
        mass = sum(map(int, weights))
        skipped = tuple(a for a in legal if a not in values)
        assert not skipped or values[action] == mass
        assert self.work["terminal_rows_valued"] == self.work["completions"]
        assert self.work["terminal_mass_valued"] == self.work["represented_completions"]
        return CeilingPlan(
            action,
            fallback,
            {a: Fraction(value + mass, 2 * mass) for a, value in values.items()},
            policies[action],
            policies,
            dict(self.work),
            self.optimize,
            skipped,
            values[action] == mass,
        )
