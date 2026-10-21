"""Dispatch the selected five-draw extension within the original <=5 draw gate."""

from scripts.frontier_hybrid_controller import FrontierHybrid, validate_hybrid
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.selected_frontier_controller import (
    configuration,
    evaluate_selected_frontier,
)
from scripts.symbolic_final_turn_tail import SymbolicFinalTurnPlanner


def evaluate_scoped_frontier(worlds, dynamics, parent_id, opponent_ids):
    if worlds.draws == 5:
        return evaluate_selected_frontier(worlds, dynamics, parent_id, opponent_ids)
    if not 0 <= worlds.draws < 5:
        raise ValueError("Expected the original at-most-five-draw gate")
    config = configuration()
    fixed, optimized, selection = evaluate_controller(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        max_nodes=config["phase_nodes"],
        max_depth=32,
        max_decisions_per_phase=config["phase_decisions"],
        planner_type=SymbolicFinalTurnPlanner,
    )
    used = fixed.result.work["decisions"]
    if optimized is not None:
        used += optimized.result.work["decisions"]
    result = FrontierHybrid(
        fixed, optimized, selection, None, used, config["phase_decisions"]
    )
    validate_hybrid(result)
    return result
