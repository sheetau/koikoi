"""Adopt and execute a completed policy only after a same-belief parent comparison."""

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.belief_tail import visible_keys
from scripts.incumbent_belief_tail import (
    IncumbentBeliefPlanner,
    IncumbentPlan,
    IncumbentResult,
)


@dataclass(frozen=True)
class TailBinding:
    digest: str
    parent_identity: str
    opponent_identities: tuple
    viewer: int
    root_key: object
    legal: tuple[int, ...]


def bind_worlds(worlds, dynamics, parent_identity, opponent_identities):
    if (
        not isinstance(parent_identity, str)
        or not parent_identity
        or set(opponent_identities) != set(worlds.model_names)
    ):
        raise ValueError("Explicit parent and every opponent identity are required")
    if any(not isinstance(v, str) or not v for v in opponent_identities.values()):
        raise ValueError("Opponent identities must be nonempty strings")
    keys = dynamics.keys(worlds.states, worlds.viewer)
    if not keys or len(set(keys)) != 1:
        raise ValueError("One root information set is required")
    identities = tuple((name, opponent_identities[name]) for name in worlds.model_names)
    digest = hashlib.sha256(
        json.dumps(
            {
                "parent": parent_identity,
                "opponents": identities,
                "viewer": worlds.viewer,
                "draws": worlds.draws,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for array in (worlds.states, worlds.models, worlds.weights):
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return TailBinding(
        digest.hexdigest(),
        parent_identity,
        identities,
        worlds.viewer,
        keys[0],
        tuple(dynamics.legal(worlds.states[0])),
    )


@dataclass
class EvaluatedPhase:
    binding: TailBinding
    mode: str
    result: IncumbentResult


@dataclass
class TailSelection:
    binding: TailBinding
    mode: str
    parent_q: Fraction | None
    selected_q: Fraction | None
    plan: IncumbentPlan | None
    reason: str

    @property
    def gain(self):
        return None if self.parent_q is None else self.selected_q - self.parent_q


def validate_phase(phase):
    if phase.mode not in ("fixed", "optimized"):
        raise ValueError("Unknown evaluation mode")
    plan = phase.result.plan
    if plan is None:
        return
    if plan.optimized_continuation != (phase.mode == "optimized"):
        raise ValueError("Evaluation mode does not match the completed policy")
    values, bounds = plan.values, plan.action_upper_bounds
    if (
        not values
        or plan.action not in values
        or plan.baseline_action not in values
        or set(values) & set(bounds)
        or set(values) | set(bounds) != set(phase.binding.legal)
        or set(plan.root_policies) != set(values)
        or any(
            not isinstance(q, Fraction) or not 0 <= q <= 1
            for q in (*values.values(), *bounds.values())
        )
        or any(
            policy.get(phase.binding.root_key) != a
            for a, policy in plan.root_policies.items()
        )
        or plan.policy != plan.root_policies[plan.action]
        or values[plan.action] != max(values.values())
        or plan.work != phase.result.work
    ):
        raise ValueError("Malformed or mismatched completed root policies")


def saved_phase(binding, mode, plan):
    """Bind a previously audited complete plan; provenance is checked by the caller."""
    if not isinstance(plan, IncumbentPlan):
        bounds = getattr(plan, "action_upper_bounds", None)
        if bounds is None:
            bounds = {a: Fraction(1) for a in binding.legal if a not in plan.values}
        plan = IncumbentPlan(
            plan.action,
            plan.baseline_action,
            plan.values,
            plan.policy,
            plan.root_policies,
            plan.work,
            plan.optimized_continuation,
            bounds,
            True,
            None,
        )
    phase = EvaluatedPhase(
        binding, mode, IncumbentResult(plan, dict(plan.work), plan.refusal)
    )
    validate_phase(phase)
    return phase


def select_improvement(fixed, optimized=None):
    if fixed.mode != "fixed":
        raise ValueError(
            "The unchanged parent reference must come from fixed continuation"
        )
    validate_phase(fixed)
    if optimized is not None:
        if optimized.mode != "optimized" or optimized.binding != fixed.binding:
            raise ValueError(
                "Candidate and parent must share the exact evaluation context"
            )
        validate_phase(optimized)
    parent_plan = fixed.result.plan
    if parent_plan is None:
        return TailSelection(
            fixed.binding, "parent", None, None, None, "parent_unvalued"
        )
    parent_q = parent_plan.values[parent_plan.baseline_action]
    best, mode = parent_plan, "fixed"
    if optimized is not None and optimized.result.plan is not None:
        candidate = optimized.result.plan
        if candidate.values[candidate.action] > best.values[best.action]:
            best, mode = candidate, "optimized"
    value = best.values[best.action]
    if value <= parent_q:
        return TailSelection(
            fixed.binding, "parent", parent_q, parent_q, None, "no_strict_gain"
        )
    return TailSelection(
        fixed.binding, mode, parent_q, value, best, "strict_exact_gain"
    )


def evaluate_controller(
    worlds,
    dynamics,
    parent_identity,
    opponent_identities,
    *,
    max_nodes=500000,
    max_depth=32,
    max_decisions_per_phase=2000000,
    planner_type=IncumbentBeliefPlanner,
):
    """Fixed then optimized, at most two independently capped phases.

    Skip optimization if the parent value is unavailable or fixed value is1.
    Never transfer unused resource allowance between phases.
    """
    binding = bind_worlds(worlds, dynamics, parent_identity, opponent_identities)

    def evaluate(mode):
        planner = planner_type(dynamics, max_nodes, max_depth, max_decisions_per_phase)
        result = planner.solve(worlds, mode == "optimized")
        if (
            bind_worlds(worlds, dynamics, parent_identity, opponent_identities)
            != binding
        ):
            raise ValueError("Search mutated its root belief")
        phase = EvaluatedPhase(binding, mode, result)
        validate_phase(phase)
        return phase

    fixed = evaluate("fixed")
    plan = fixed.result.plan
    optimized = (
        evaluate("optimized")
        if plan is not None and plan.values[plan.action] < 1
        else None
    )
    return fixed, optimized, select_improvement(fixed, optimized)


class ContingentExecutor:
    """Use own/public keys only; a missing adopted-policy key is an error."""

    def __init__(self, selection, parent, parent_identity):
        if parent_identity != selection.binding.parent_identity:
            raise ValueError("Execution parent identity differs from the evaluation")
        if selection.plan is not None and (
            selection.reason != "strict_exact_gain"
            or selection.parent_q is None
            or selection.selected_q is None
            or not 0 <= selection.parent_q < selection.selected_q <= 1
            or selection.selected_q != selection.plan.values[selection.plan.action]
            or selection.plan.policy.get(selection.binding.root_key)
            != selection.plan.action
        ):
            raise ValueError("Execution requires a strict completed-policy improvement")
        self.selection, self.parent = selection, parent
        self.policy = None if selection.plan is None else dict(selection.plan.policy)
        self.started = False

    def act(self, env):
        viewer = self.selection.binding.viewer
        if int(env.state[PLAYER]) != viewer or int(env.state[PHASE]) == TERMINAL:
            raise ValueError("Executor requires a live learner decision")
        key = visible_keys(env.state[None], viewer)[0]
        if not self.started and key != self.selection.binding.root_key:
            raise ValueError("Execution root differs from the certified root")
        if self.policy is None:
            action = int(self.parent.act(env))
        else:
            if key not in self.policy:
                raise KeyError(
                    "Adopted policy has no action for this own/public history"
                )
            action = int(self.policy[key])
        if not 0 <= action < len(env.legal()) or not env.legal()[action]:
            raise ValueError("Selected action is illegal")
        self.started = True
        return action
