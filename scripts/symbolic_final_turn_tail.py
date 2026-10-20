"""Complete tail programs with a certified parent continuation on final own wins."""

from dataclasses import dataclass, fields
from fractions import Fraction

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, ROUNDS, TERMINAL, TURN
from scripts.belief_tail import visible_keys
from scripts.bounded_belief_tail import BoundedBeliefPlanner, ContinuationValue
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner, IncumbentPlan
from scripts.incumbent_tail_controller import ContingentExecutor
from scripts.opponent_last_turn_tail import opponent_last_turn_loss

RULE = "final-own-turn-margin-above-two-v1"


def final_own_turn_win(states, viewer):
    """Only learner scoring or the dealer's one-point award remains: margin >= D-2."""
    if viewer not in (0, 1):
        raise ValueError("Invalid viewer")
    margin = states[:, POINTS + viewer] - states[:, POINTS + 1 - viewer]
    return (
        (states[:, PHASE] != TERMINAL)
        & (states[:, ROUND] + 1 == states[:, ROUNDS])
        & (states[:, TURN] == 15)
        & (states[:, PLAYER] == viewer)
        & (margin > 2)
    )


@dataclass
class SymbolicPlan(IncumbentPlan):
    """Explicit map plus the unchanged parent at certified final-turn own wins."""

    leaf_rule: str = RULE


class SymbolicFinalTurnPlanner(IncumbentBeliefPlanner):
    def solve(self, worlds, optimize_continuation=True):
        result = super().solve(worlds, optimize_continuation)
        if result.plan is not None:
            result.plan = SymbolicPlan(
                **{
                    field.name: getattr(result.plan, field.name)
                    for field in fields(IncumbentPlan)
                }
            )
        return result

    def continuation(self, states, models, weights, depth, cutoff=None):
        win = final_own_turn_win(states, self.viewer)
        loss = opponent_last_turn_loss(states, self.viewer)
        certified = win | loss
        if not certified.any():
            return super().continuation(states, models, weights, depth, cutoff)
        signed = 0
        for label, mask, sign in (("own_win", win, 1), ("opponent_loss", loss, -1)):
            mass = sum(map(int, weights[mask]))
            self.work[f"certified_{label}_rows"] = self.work.get(
                f"certified_{label}_rows", 0
            ) + int(mask.sum())
            self.work[f"certified_{label}_mass"] = (
                self.work.get(f"certified_{label}_mass", 0) + mass
            )
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


class SymbolicExecutor(ContingentExecutor):
    """Resolve a missing map key only inside the declared own/public win region."""

    def __init__(self, selection, parent, parent_identity):
        super().__init__(selection, parent, parent_identity)
        if selection.plan is not None and (
            not isinstance(selection.plan, SymbolicPlan)
            or selection.plan.leaf_rule != RULE
        ):
            raise ValueError("A recognized symbolic leaf rule is required")
        self.symbolic_parent_calls = 0

    def act(self, env):
        viewer = self.selection.binding.viewer
        if self.started and self.policy is not None:
            key = visible_keys(env.state[None], viewer)[0]
            if (
                key not in self.policy
                and final_own_turn_win(env.state[None], viewer)[0]
            ):
                # The superclass still validates actor, phase and legality.
                self.policy[key] = int(self.parent.act(env))
                self.symbolic_parent_calls += 1
        return super().act(env)
