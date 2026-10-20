"""Exact policy selection with upper bounds on unfinished action continuations."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from scripts.belief_tail import TailBudgetExceeded, TailPlan
from scripts.ceiling_belief_tail import CeilingBeliefPlanner
from scripts.deferred_belief_tail import DeferredWorldSet


@dataclass
class BoundedPlan(TailPlan):
    action_upper_bounds: dict[int, Fraction]


@dataclass
class ContinuationValue:
    value: int  # Exact signed value, or a certified upper bound.
    exact: bool
    policy: dict


class BoundedBeliefPlanner(CeilingBeliefPlanner):
    """Discard an action only when its upper bound cannot beat an exact incumbent.

    Every returned policy and chosen value is complete and exact. Bounds never
    choose different learner actions inside one own/public information set.
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
        values, policies, bounds = {}, {}, {}
        best_action = fallback
        for index, action in enumerate(order):
            cutoff = values[best_action] if values else None
            successor = self.transition(
                states, models, weights, np.full(len(states), action, dtype=np.int32)
            )
            result = self.continuation(*successor, depth + 1, cutoff)
            if not -mass <= result.value <= mass:
                raise ValueError("Continuation violates the signed utility bound")
            if not result.exact:
                assert cutoff is not None and result.value <= cutoff
                assert not result.policy
                bounds[action] = result.value
                self.work["bounded_actions"] += 1
                continue
            values[action], policies[action] = result.value, result.policy
            if result.value > values[best_action]:
                best_action = action
            if result.value == mass:
                skipped = order[index + 1 :]
                bounds.update(dict.fromkeys(skipped, mass))
                self.work["ceiling_cutoffs"] += int(bool(skipped))
                self.work["pruned_actions"] += len(skipped)
                break
        return best_action, fallback, legal, values, policies, bounds

    def continuation(self, states, models, weights, depth, cutoff=None):
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
            return ContinuationValue(total, True, {})
        states, models, weights = (
            states[~terminal],
            models[~terminal],
            weights[~terminal],
        )
        remaining = sum(map(int, weights))
        if cutoff is not None and total + remaining <= cutoff:
            self.work["continuation_cutoffs"] += 1
            return ContinuationValue(total + remaining, False, {})
        groups = defaultdict(list)
        for index, key in enumerate(self.dynamics.keys(states, self.viewer)):
            groups[key].append(index)
        policy = {}
        for key, indices in groups.items():
            self.work["live_observation_nodes"] += 1
            if self.work["live_observation_nodes"] > self.max_nodes:
                raise TailBudgetExceeded("Live observation-node cap exceeded")
            s, m, w = states[indices], models[indices], weights[indices]
            mass = sum(map(int, w))
            if len(set(self.dynamics.players(s))) != 1:
                raise ValueError("Merged acting-player states")
            if self.dynamics.players(s)[0] != self.viewer:
                successor = self.transition(
                    s, m, w, self.dynamics.opponent_actions(s, m)
                )
                local_cutoff = (
                    None if cutoff is None else cutoff - total - (remaining - mass)
                )
                result = self.continuation(*successor, depth + 1, local_cutoff)
                if not result.exact:
                    upper = total + result.value + remaining - mass
                    assert cutoff is not None and upper <= cutoff
                    return ContinuationValue(upper, False, {})
                value, tail = result.value, result.policy
            else:
                # Solve each learner information set exactly. A local action
                # cutoff may discard alternatives, never the selected policy.
                self.work["own_decision_nodes"] += 1
                action, _, _, values, policies, _ = self.choose(s, m, w, depth)
                value, tail = values[action], policies[action]
                tail[key] = action
            total += value
            remaining -= mass
            self.merge(policy, tail)
            if remaining and cutoff is not None and total + remaining <= cutoff:
                self.work["continuation_cutoffs"] += 1
                return ContinuationValue(total + remaining, False, {})
        assert remaining == 0
        return ContinuationValue(total, True, policy)

    def solve(self, worlds, optimize_continuation=True):
        self.work = {}
        if not isinstance(worlds, DeferredWorldSet):
            raise TypeError("Bounded solver requires unordered-stock worlds")
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
                "continuation_cutoffs",
                "bounded_actions",
            ),
            0,
        )
        action, fallback, legal, values, policies, bounds = self.choose(
            states, models, weights, 0, root=True
        )
        for first, policy in policies.items():
            policy[keys[0]] = first
        mass = sum(map(int, weights))
        assert set(values).isdisjoint(bounds) and set(values) | set(bounds) == set(
            legal
        )
        assert all(upper <= values[action] for upper in bounds.values())
        assert self.work["terminal_rows_valued"] == self.work["completions"]
        assert self.work["terminal_mass_valued"] == self.work["represented_completions"]
        return BoundedPlan(
            action,
            fallback,
            {a: Fraction(value + mass, 2 * mass) for a, value in values.items()},
            policies[action],
            policies,
            dict(self.work),
            self.optimize,
            {a: Fraction(value + mass, 2 * mass) for a, value in bounds.items()},
        )
