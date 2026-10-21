"""Selected five-draw hybrid with the bounded next-discard public frontier."""

import numpy as np

from koikoi.layout import DISCARD, PHASE, TURN
from scripts.frontier_hybrid_controller import evaluate_frontier_hybrid
from scripts.next_discard_frontier import next_discard_frontier


def configuration():
    return {
        "phase_decisions": 2000000,
        "phase_nodes": 500000,
        "prefix_decisions": 250000,
        "local_phase_decisions": 50000,
        "local_nodes": 25000,
        "max_rows": 65536,
    }


def evaluate_selected_frontier(worlds, dynamics, parent_id, opponent_ids):
    if worlds.draws != 5:
        raise ValueError("Selected controller requires the original five-draw gate")
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    result = evaluate_frontier_hybrid(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        patch_draws=bound.draws,
        **configuration(),
    )
    if result.patch is not None:
        frontier = result.patch.frontier
        if (
            not frontier.complete
            or frontier.draws != bound.draws
            or frontier.work["peak_rows"] > bound.maximum_rows
            or frontier.work["decisions"] > bound.maximum_transitions
        ):
            raise ValueError("Selected bounded frontier construction failed")
        for _, node in frontier.nodes:
            if not np.all(node.states[:, TURN] == bound.target_turn) or not np.all(
                node.states[:, PHASE] == DISCARD
            ):
                raise ValueError("Selected frontier is not the next learner discard")
    return result
