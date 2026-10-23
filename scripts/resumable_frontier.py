"""Cheap complete parent-default proposals with resumable interval allocation.

Each proposal changes one public root action and reuses the unchanged parent
as its entire executable continuation. No partial search map needs completion.
Only a strict same-belief interval comparison can activate a replacement.
"""

from fractions import Fraction

from scripts.frontier_policy_patch import FrontierPatch, parent_frontier
from scripts.interval_tail import IntervalEvaluator, make_program, select_interval


def compare_root_programs(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    max_decisions=100000,
    max_nodes=25000,
    quantum=4096,
):
    if any(type(x) is not int or x < 1 for x in (max_decisions, max_nodes, quantum)):
        raise ValueError("Positive integer comparison budgets required")
    original = make_program(worlds, dynamics, parent_id, opponent_ids)
    evaluators = [IntervalEvaluator(worlds, dynamics, original)]
    for action in original.binding.legal:
        if action != original.parent_root_action:
            program = make_program(
                worlds,
                dynamics,
                parent_id,
                opponent_ids,
                {original.binding.root_key: action},
            )
            evaluators.append(IntervalEvaluator(worlds, dynamics, program))
    used = 0
    blocked = set()
    while used < max_decisions:
        receipts = [e.receipt() for e in evaluators]
        parent = receipts[0]
        selection = select_interval(parent, receipts[1:])
        if selection.candidate is not None:
            break
        active = [
            i for i in range(1, len(receipts)) if receipts[i].upper > parent.lower
        ]
        if not active:
            break
        # Reduce the widest relevant uncertainty first, breaking ties in favor
        # of the shared parent. A parent interval serves every candidate.
        options = [
            i
            for i in [0, *active]
            if i not in blocked and receipts[i].upper > receipts[i].lower
        ]
        if not options:
            break
        index = max(options, key=lambda i: receipts[i].upper - receipts[i].lower)
        evaluator = evaluators[index]
        previous = evaluator.work["decisions"]
        allowance = min(quantum, max_decisions - used)
        receipt = evaluator.run(previous + allowance, max_nodes)
        spent = evaluator.work["decisions"] - previous
        used += spent
        if receipt.refusal not in (None, "Physical transition cap reached"):
            blocked.add(index)
        elif not spent:
            if allowance == max_decisions - used:
                blocked.add(index)
            else:
                quantum *= 2
    receipts = [e.receipt() for e in evaluators]
    if used != sum(r.work["decisions"] for r in receipts) or used > max_decisions:
        raise ValueError("Resumed evaluation work differs from the shared budget")
    selection = select_interval(receipts[0], receipts[1:])
    return {
        "selection": selection,
        "parent_receipt": receipts[0],
        "candidate_receipts": tuple(receipts[1:]),
        "decisions": used,
        "evaluation_work": [r.work for r in receipts],
    }


def solve_resumable_frontier(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    local_decisions=100000,
    max_nodes_per_phase=25000,
    max_rows=65536,
):
    frontier = parent_frontier(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used, comparisons, gain = frontier.work["decisions"], [], Fraction(0)
    if frontier.complete:
        for key, node in frontier.nodes:
            allowance = min(local_decisions, max_decisions - used)
            if allowance < 1:
                break
            result = compare_root_programs(
                node,
                dynamics,
                parent_id,
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
    return FrontierPatch(frontier, tuple(comparisons), gain, used)
