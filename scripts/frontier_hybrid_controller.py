"""Preserve the symbolic controller and patch only its unvalued-parent fallback."""

from dataclasses import dataclass
from fractions import Fraction

from scripts.frontier_policy_patch import FrontierExecutor, solve_frontier_patch
from scripts.incumbent_tail_controller import evaluate_controller, select_improvement
from scripts.symbolic_final_turn_tail import SymbolicExecutor, SymbolicFinalTurnPlanner


@dataclass
class FrontierHybrid:
    fixed: object
    optimized: object
    original: object
    patch: object
    decisions: int
    phase_decisions: int

    @property
    def gain_lower(self):
        return (
            self.original.gain or Fraction(0)
            if self.patch is None
            else self.patch.gain_lower
        )


def validate_hybrid(result):
    if select_improvement(result.fixed, result.optimized) != result.original:
        raise ValueError("Original controller selection changed")
    used = result.fixed.result.work["decisions"]
    if used > result.phase_decisions:
        raise ValueError("Original fixed phase exceeds its allowance")
    if result.optimized is not None:
        second = result.optimized.result.work["decisions"]
        if second > result.phase_decisions:
            raise ValueError("Original optimized phase exceeds its allowance")
        used += second
    if result.patch is not None:
        if (
            result.original.reason != "parent_unvalued"
            or result.optimized is not None
            or result.patch.frontier.parent_program.binding != result.fixed.binding
            or result.patch.decisions > result.phase_decisions
        ):
            raise ValueError(
                "Patch must use the skipped phase of the same parent refusal"
            )
        used += result.patch.decisions
    if used != result.decisions or used > 2 * result.phase_decisions:
        raise ValueError("Integrated work differs or exceeds the two-phase maximum")


def evaluate_frontier_hybrid(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    phase_decisions=2000000,
    phase_nodes=500000,
    patch_draws=3,
    prefix_decisions=250000,
    local_phase_decisions=50000,
    local_nodes=25000,
    max_rows=65536,
):
    """Actually execute the old first phase; historical work is not imported.

    Original strict improvements and no-gain decisions retain their programs
    exactly. Only a missing parent value activates the alternative second phase.
    Both phases have the same physical-transition allowance; no borrowing.
    """
    if type(phase_decisions) is not int or phase_decisions < 1:
        raise ValueError("Positive phase transition allowance required")
    fixed, optimized, original = evaluate_controller(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        max_nodes=phase_nodes,
        max_depth=32,
        max_decisions_per_phase=phase_decisions,
        planner_type=SymbolicFinalTurnPlanner,
    )
    used = fixed.result.work["decisions"]
    if optimized is not None:
        used += optimized.result.work["decisions"]
    patch = None
    if original.reason == "parent_unvalued":
        patch = solve_frontier_patch(
            worlds,
            dynamics,
            parent_id,
            opponent_ids,
            draws=patch_draws,
            max_decisions=phase_decisions,
            prefix_decisions=min(prefix_decisions, phase_decisions),
            per_phase_decisions=local_phase_decisions,
            max_nodes_per_phase=local_nodes,
            max_rows=max_rows,
        )
        used += patch.decisions
    result = FrontierHybrid(fixed, optimized, original, patch, used, phase_decisions)
    validate_hybrid(result)
    return result


class FrontierHybridExecutor:
    def __init__(self, result, parent, parent_id):
        validate_hybrid(result)
        self.executor = (
            SymbolicExecutor(result.original, parent, parent_id)
            if result.patch is None
            else FrontierExecutor(result.patch, parent, parent_id)
        )

    def act(self, env):
        return self.executor.act(env)
