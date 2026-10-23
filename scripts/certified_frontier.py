"""Skip candidate construction where public bounds fix every continuation's value."""

from dataclasses import dataclass
from fractions import Fraction

from scripts.continuation_frontier import compare_continuations
from scripts.frontier_policy_patch import FrontierPatch, parent_frontier
from scripts.public_score_bounds import outcome_certificates


@dataclass(frozen=True)
class CertifiedSkipResult:
    patch: FrontierPatch
    # Public key, certified winning mass, certified losing mass.
    skipped: tuple[tuple[bytes, int, int], ...]


def solve_certified_frontier(
    worlds,
    dynamics,
    actors,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    local_decisions=100000,
    max_nodes_per_phase=25000,
    max_rows=65536,
):
    if any(
        type(v) is not int or v < 1
        for v in (max_decisions, prefix_decisions, local_decisions)
    ):
        raise ValueError("Positive comparison budgets required")
    if dynamics.baseline is not actors.parent:
        raise ValueError("Dynamics reference is not the registered parent")
    frontier = parent_frontier(
        worlds,
        dynamics,
        actors.parent_identity,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used, comparisons, skipped, gain = frontier.work["decisions"], [], [], Fraction(0)
    if frontier.complete:
        for key, node in frontier.nodes:
            allowance = min(local_decisions, max_decisions - used)
            if allowance < 1:
                break
            win, loss = outcome_certificates(node.states, node.viewer)
            if (win & loss).any():
                raise ValueError("Conflicting public certificates")
            if (win | loss).all():
                # These are policy-independent bounds. The parent and every
                # legal candidate have the same value; no root action is needed.
                win_mass = sum(map(int, node.weights[win]))
                loss_mass = sum(map(int, node.weights[loss]))
                if win_mass + loss_mass != sum(map(int, node.weights)):
                    raise ValueError("Certificate mass does not cover the node")
                skipped.append((key, win_mass, loss_mass))
                continue
            result = compare_continuations(
                node,
                dynamics,
                actors,
                opponent_ids,
                max_decisions=allowance,
                max_nodes=max_nodes_per_phase,
            )
            used += result["decisions"]
            comparisons.append((key, result))
            gain += (
                Fraction(sum(map(int, node.weights)), frontier.total_mass)
                * result["selection"].gain_lower
            )
    if used > max_decisions:
        raise ValueError("Global transition allowance exceeded")
    return CertifiedSkipResult(
        FrontierPatch(frontier, tuple(comparisons), gain, used), tuple(skipped)
    )
