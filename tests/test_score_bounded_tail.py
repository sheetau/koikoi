from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.incumbent_tail_controller import (
    EvaluatedPhase,
    bind_worlds,
    select_improvement,
)
from scripts.public_score_bounds import RULE
from scripts.score_bounded_tail import ScoreBoundExecutor, ScoreBoundPlanner
from scripts.symbolic_final_turn_tail import SymbolicFinalTurnPlanner
from scripts.verify_score_bounded_tail import verify_program
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("margin", [-40, -6, 2, 6, 40])
def test_values_match_old_planner_and_full_python_replay(swap, margin):
    actor, _, worlds = setup(swap, margin)
    worlds = replace(
        worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
    )
    for optimize in (False, True):
        old = SymbolicFinalTurnPlanner(dynamics(actor, worlds)).solve(worlds, optimize)
        new = ScoreBoundPlanner(dynamics(actor, worlds)).solve(worlds, optimize)
        assert old.refusal is new.refusal is None
        assert old.plan.values == new.plan.values
        assert old.plan.action == new.plan.action
        assert new.plan.leaf_rule == RULE
        verified = verify_program(worlds, new.plan, actor, {"fixture": actor})
        assert verified["values"] == {
            str(a): str(q) for a, q in old.plan.values.items()
        }
        assert new.work["decisions"] <= old.work["decisions"]
        if abs(margin) == 40:
            assert new.work["decisions"] < old.work["decisions"]


def test_executor_rejects_unknown_rules_and_outside_default_region():
    actor, root, worlds = setup(margin=6)
    dyn = dynamics(actor, worlds)
    result = ScoreBoundPlanner(dyn).solve(worlds, False)
    binding = bind_worlds(worlds, dyn, "parent", {"fixture": "fixture"})
    selection = select_improvement(EvaluatedPhase(binding, "fixed", result))
    assert selection.plan is not None
    executor = ScoreBoundExecutor(selection, actor, "parent")
    assert executor.act(root) == selection.plan.action
    broken = deepcopy(selection)
    broken.plan.leaf_rule = "unknown"
    with pytest.raises(ValueError):
        ScoreBoundExecutor(broken, actor, "parent")
    # A map hole at a non-certified root must remain an error.
    for mapping in result.plan.root_policies.values():
        mapping.clear()
    with pytest.raises(KeyError):
        verify_program(worlds, result.plan, actor, {"fixture": actor})
