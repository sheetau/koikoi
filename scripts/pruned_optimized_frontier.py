"""Apply public-outcome pruning before the original optimized proposal search."""

from fractions import Fraction

from scripts.certified_frontier import CertifiedSkipResult
from scripts.frontier_policy_patch import FrontierPatch, parent_frontier
from scripts.proposal_interval_tail import propose_then_compare
from scripts.public_score_bounds import outcome_certificates


def solve_pruned_optimized_frontier(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    per_phase_decisions=50000,
    max_nodes_per_phase=25000,
    max_rows=65536,
):
    if any(
        type(v) is not int or v < 1
        for v in (max_decisions, prefix_decisions, per_phase_decisions)
    ):
        raise ValueError("Positive shared and phase transition budgets required")
    frontier = parent_frontier(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used, comparisons, skipped, gain = frontier.work["decisions"], [], [], Fraction(0)
    if frontier.complete:
        for key, node in frontier.nodes:
            allowance = min(per_phase_decisions, (max_decisions - used) // 2)
            if allowance == 0:
                break
            win, loss = outcome_certificates(node.states, node.viewer)
            if (win & loss).any():
                raise ValueError("Conflicting public certificates")
            if (win | loss).all():
                winning = sum(map(int, node.weights[win]))
                losing = sum(map(int, node.weights[loss]))
                if winning + losing != sum(map(int, node.weights)):
                    raise ValueError("Certificate mass does not cover the node")
                skipped.append((key, winning, losing))
                continue
            result = propose_then_compare(
                node,
                dynamics,
                parent_id,
                opponent_ids,
                proposal_decisions=allowance,
                reference_decisions=allowance,
                max_nodes_per_phase=max_nodes_per_phase,
            )
            used += result["decisions"]
            comparisons.append((key, result))
            gain += (
                Fraction(sum(map(int, node.weights)), frontier.total_mass)
                * result["selection"].gain_lower
            )
    if used > max_decisions:
        raise ValueError("Shared patch budget exceeded")
    return CertifiedSkipResult(
        FrontierPatch(frontier, tuple(comparisons), gain, used), tuple(skipped)
    )
