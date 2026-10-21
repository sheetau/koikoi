"""Complete belief-tail programs with public score-bound outcome certificates."""

from dataclasses import dataclass, fields
from fractions import Fraction

from scripts.belief_tail import visible_keys
from scripts.bounded_belief_tail import BoundedBeliefPlanner, ContinuationValue
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner, IncumbentPlan
from scripts.incumbent_tail_controller import ContingentExecutor
from scripts.public_score_bounds import RULE, outcome_certificates


@dataclass
class ScoreBoundPlan(IncumbentPlan):
    leaf_rule: str = RULE


class ScoreBoundPlanner(IncumbentBeliefPlanner):
    def solve(self, worlds, optimize_continuation=True):
        result = super().solve(worlds, optimize_continuation)
        if result.plan is not None:
            result.plan = ScoreBoundPlan(
                **{f.name: getattr(result.plan, f.name) for f in fields(IncumbentPlan)}
            )
        return result

    def continuation(self, states, models, weights, depth, cutoff=None):
        win, loss = outcome_certificates(states, self.viewer)
        certified = win | loss
        if not certified.any():
            return super().continuation(states, models, weights, depth, cutoff)
        signed = 0
        for label, mask, sign in (("win", win, 1), ("loss", loss, -1)):
            mass = sum(map(int, weights[mask]))
            for suffix, count in (("rows", int(mask.sum())), ("mass", mass)):
                key = f"certified_score_{label}_{suffix}"
                self.work[key] = self.work.get(key, 0) + count
            signed += sign * mass
        rest = BoundedBeliefPlanner.continuation(
            self,
            states[~certified],
            models[~certified],
            weights[~certified],
            depth,
            None if cutoff is None else cutoff - signed,
        )
        result = ContinuationValue(rest.value + signed, rest.exact, rest.policy)
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


class ScoreBoundExecutor(ContingentExecutor):
    """The unchanged parent completes both public win and public loss regions."""

    def __init__(self, selection, parent, parent_identity):
        super().__init__(selection, parent, parent_identity)
        if selection.plan is not None and (
            not isinstance(selection.plan, ScoreBoundPlan)
            or selection.plan.leaf_rule != RULE
        ):
            raise ValueError("Expected the versioned public score-bound program")
        self.symbolic_parent_calls = 0

    def act(self, env):
        viewer = self.selection.binding.viewer
        if self.started and self.policy is not None:
            key = visible_keys(env.state[None], viewer)[0]
            if key not in self.policy:
                win, loss = outcome_certificates(env.state[None], viewer)
                if win[0] or loss[0]:
                    self.policy[key] = int(self.parent.act(env))
                    self.symbolic_parent_calls += 1
        return super().act(env)
