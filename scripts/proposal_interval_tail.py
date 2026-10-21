"""Optimize a complete contingent program, then bound the unchanged parent."""

from scripts.incumbent_tail_controller import (
    EvaluatedPhase,
    bind_worlds,
    validate_phase,
)
from scripts.interval_tail import (
    IntervalEvaluator,
    IntervalReceipt,
    make_program,
    select_interval,
)
from scripts.public_score_bounds import RULE
from scripts.score_bounded_tail import ScoreBoundPlan, ScoreBoundPlanner


def compile_proposal(phase, worlds, dynamics, parent_id, opponent_ids):
    """Transfer only a completed same-belief policy value, never a search bound.

    The exact value is backed by the score-bound planner's complete explicit map
    plus its certified parent-default regions. A retained incumbent may be used
    even when later alternatives exceeded the proposal cap. Independent full
    replay remains mandatory before accepting empirical evidence.
    """
    binding = bind_worlds(worlds, dynamics, parent_id, opponent_ids)
    if phase.binding != binding or phase.mode != "optimized":
        raise ValueError("Expected an optimized proposal for the same complete belief")
    validate_phase(phase)
    plan = phase.result.plan
    if plan is None:
        return None
    if not isinstance(plan, ScoreBoundPlan) or plan.leaf_rule != RULE:
        raise ValueError("Unknown proposal default-domain proof")
    program = make_program(worlds, dynamics, parent_id, opponent_ids, plan.policy)
    if plan.baseline_action != program.parent_root_action:
        raise ValueError("Proposal uses a different parent")
    mass = sum(map(int, worlds.weights))
    signed = (2 * plan.values[plan.action] - 1) * mass
    if signed.denominator != 1:
        raise ValueError("Completed proposal value lacks integer outcome mass")
    receipt = IntervalReceipt(
        program,
        mass,
        mass,
        int(signed),
        {
            "proposal_physical_decisions": phase.result.work["decisions"],
            "proposal_live_observation_nodes": phase.result.work[
                "live_observation_nodes"
            ],
        },
        None,
    )
    receipt.validate()
    return receipt


def propose_then_compare(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    proposal_decisions=1000000,
    reference_decisions=1000000,
    max_nodes_per_phase=250000,
    quantum=4096,
):
    """Two preassigned budgets with no borrowing or hidden per-action allowance."""
    if any(
        type(x) is not int or x < 1
        for x in (proposal_decisions, reference_decisions, max_nodes_per_phase, quantum)
    ):
        raise ValueError("Positive integer budgets required")
    binding = bind_worlds(worlds, dynamics, parent_id, opponent_ids)
    result = ScoreBoundPlanner(
        dynamics, max_nodes=max_nodes_per_phase, max_decisions=proposal_decisions
    ).solve(worlds, True)
    phase = EvaluatedPhase(binding, "optimized", result)
    candidate = compile_proposal(phase, worlds, dynamics, parent_id, opponent_ids)
    parent_program = make_program(worlds, dynamics, parent_id, opponent_ids)
    evaluator = IntervalEvaluator(worlds, dynamics, parent_program)
    parent = evaluator.receipt()
    if candidate is not None:
        while parent.lower < candidate.lower <= parent.upper:
            if parent.work["decisions"] >= reference_decisions:
                break
            previous = parent
            parent = evaluator.run(
                min(reference_decisions, parent.work["decisions"] + quantum),
                max_nodes_per_phase,
            )
            if parent.work == previous.work or parent.refusal not in (
                None,
                "Physical transition cap reached",
            ):
                break
    selection = select_interval(parent, [] if candidate is None else [candidate])
    return {
        "proposal_phase": phase,
        "candidate_certificate": candidate,
        "parent_receipt": parent,
        "selection": selection,
        "decisions": result.work["decisions"] + parent.work["decisions"],
    }
