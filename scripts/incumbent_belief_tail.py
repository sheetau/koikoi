"""Retain complete executable root policies when an exact search hits its cap."""

from dataclasses import dataclass
from fractions import Fraction

from scripts.belief_tail import TailBudgetExceeded, TailPlan
from scripts.bounded_belief_tail import BoundedBeliefPlanner
from scripts.deferred_belief_tail import DeferredWorldSet


@dataclass
class IncumbentPlan(TailPlan):
    action_upper_bounds: dict[int, Fraction]
    search_complete: bool
    refusal: str | None

    @property
    def optimality_gap(self):
        upper = max((*self.values.values(), *self.action_upper_bounds.values()))
        return upper - self.values[self.action]


@dataclass
class IncumbentResult:
    plan: IncumbentPlan | None
    work: dict[str, int]
    refusal: str | None


class IncumbentBeliefPlanner(BoundedBeliefPlanner):
    """A budget refusal may retain completed policies, never a partial policy.

    An interrupted optimized continuation is not automatically safe relative
    to a fixed parent's value. A controller must compare the retained exact Q
    against an independently completed parent-policy evaluation before use.
    """

    def solve(self, worlds, optimize_continuation=True):
        self._root_states = (
            worlds.states if isinstance(worlds, DeferredWorldSet) else None
        )
        self._exact, self._policies, self._bounds = {}, {}, {}
        self._root_action, self._fallback, self._legal = None, None, None
        self._root_key = None
        self._mass = 0
        refusal = None
        try:
            complete = super().solve(worlds, optimize_continuation)
        except TailBudgetExceeded as error:
            refusal = str(error)
        else:
            plan = IncumbentPlan(
                complete.action,
                complete.baseline_action,
                complete.values,
                complete.policy,
                complete.root_policies,
                complete.work,
                complete.optimized_continuation,
                complete.action_upper_bounds,
                True,
                None,
            )
            assert plan.optimality_gap == 0
            return IncumbentResult(plan, dict(self.work), None)
        if not self._exact:
            return IncumbentResult(None, dict(self.work), refusal)
        best = max(self._exact.values())
        action = (
            self._fallback
            if self._exact.get(self._fallback) == best
            else next(a for a in self._legal if self._exact.get(a) == best)
        )
        bounds = {
            a: self._bounds.get(a, Fraction(1))
            for a in self._legal
            if a not in self._exact
        }
        plan = IncumbentPlan(
            action,
            self._fallback,
            dict(self._exact),
            self._policies[action],
            dict(self._policies),
            dict(self.work),
            self.optimize,
            bounds,
            False,
            refusal,
        )
        assert set(plan.values).isdisjoint(bounds)
        assert set(plan.values) | set(bounds) == set(self._legal)
        return IncumbentResult(plan, dict(self.work), refusal)

    def transition(self, states, models, weights, actions):
        if states is self._root_states:
            self._root_action = int(actions[0])
            if self._fallback is None:
                self._fallback = self._root_action
                self._legal = self.dynamics.legal(states[0])
                self._root_key = self.dynamics.keys(states, self.viewer)[0]
                self._mass = sum(map(int, weights))
        return self._transition(states, models, weights, actions)

    def _transition(self, states, models, weights, actions):
        return super().transition(states, models, weights, actions)

    def continuation(self, states, models, weights, depth, cutoff=None):
        result = super().continuation(states, models, weights, depth, cutoff)
        if depth == 1:
            value = Fraction(result.value + self._mass, 2 * self._mass)
            if result.exact:
                policy = dict(result.policy)
                policy[self._root_key] = self._root_action
                self._exact[self._root_action] = value
                self._policies[self._root_action] = policy
            else:
                self._bounds[self._root_action] = value
        return result
