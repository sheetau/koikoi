from fractions import Fraction

import numpy as np
import pytest

from koikoi.layout import DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND, PLAYER, STOCK, TURN
from scripts.next_discard_frontier import next_discard_frontier


@pytest.mark.parametrize("viewer", [0, 1])
@pytest.mark.parametrize(
    "turn,phase,rows,decisions",
    [
        (10, PICK_DRAW, 7280, 43680),
        (10, KOI, 7280, 36400),
        (11, DISCARD, 16380, 131040),
        (11, PICK_HAND, 16380, 114660),
    ],
)
def test_all_entry_phases_fit_existing_prefix_caps_without_private_inputs(
    viewer, turn, phase, rows, decisions
):
    state = np.full(208, -1, dtype=np.int32)
    state[PLAYER], state[TURN], state[PHASE], state[STOCK] = viewer, turn, phase, 36
    count = 3 if turn == 10 else 2
    state[:count] = 1 - viewer
    result = next_discard_frontier(state, viewer)
    assert result.maximum_rows == rows < 65536
    assert result.maximum_transitions == decisions < 250000
    assert result.draws == 14 - turn and result.target_turn == turn + 2
    assert result.future_draws == 5 - result.draws
    # Reassign hidden identities and stock order, retaining their public counts.
    changed = state.copy()
    changed[:48] = np.roll(changed[:48], 17)
    changed[72:96] = np.arange(24)[::-1]
    assert next_discard_frontier(changed, viewer) == result


@pytest.mark.parametrize("change", ["model", "turn", "stock", "hand_count", "viewer"])
def test_scope_mismatch_does_not_receive_a_false_bound(change):
    state = np.full(208, -1, dtype=np.int32)
    state[PLAYER], state[TURN], state[PHASE], state[STOCK] = 0, 11, DISCARD, 36
    state[:2] = 1
    model_count = 1
    if change == "model":
        model_count = 2
    elif change == "turn":
        state[TURN] = 12
    elif change == "stock":
        state[STOCK] += 1
    elif change == "hand_count":
        state[3] = 1
    else:
        state[PLAYER] = 1
    with pytest.raises(ValueError, match="five-draw learner gate"):
        next_discard_frontier(state, 0, model_count)


@pytest.mark.parametrize("steps", [11, 50, 51, 110])
@pytest.mark.parametrize("swap", [False, True])
def test_full_native_prefix_obeys_bound_and_independent_python_posterior(steps, swap):
    from koikoi.layout import ROUND
    from koikoi.reference import ReferenceEnv
    from scripts.deferred_belief_tail import deferred_worlds
    from scripts.frontier_policy_patch import FrontierPatch, parent_frontier
    from scripts.history_conditioned_worlds import enumerate_support
    from scripts.verify_frontier_patch import verify_frontier_patch
    from tests.test_belief_tail import swap_seats
    from tests.test_history_conditioned_worlds import Extremal
    from tests.test_symbolic_final_turn_tail import dynamics

    root, actor = ReferenceEnv(31010001, "paper8"), Extremal()
    for _ in range(steps):
        root.step(actor.act(root))
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    support = enumerate_support(root, {"fixture": actor}, max_hands=1000)
    worlds = deferred_worlds(root, support)
    bound = next_discard_frontier(root.state, worlds.viewer)
    frontier = parent_frontier(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=bound.draws,
    )
    assert frontier.complete and frontier.nodes
    assert frontier.work["peak_rows"] <= bound.maximum_rows
    assert frontier.work["decisions"] <= bound.maximum_transitions
    assert len(worlds.states) <= bound.maximum_hands
    for _, node in frontier.nodes:
        assert np.all(node.states[:, TURN] == bound.target_turn)
        assert np.all(node.states[:, PHASE] == DISCARD)
    patch = FrontierPatch(frontier, (), Fraction(0), frontier.work["decisions"])
    report = verify_frontier_patch(worlds, patch, actor, {"fixture": actor}, "parent")
    assert report["prefix_physical"]["decisions"] == frontier.work["decisions"]
