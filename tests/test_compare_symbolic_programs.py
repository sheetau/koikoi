from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL, TURN
from scripts.belief_tail import visible_keys
from scripts.compare_symbolic_programs import certified_own_win_key, compare_common
from scripts.incumbent_tail_controller import evaluate_controller
from scripts.symbolic_final_turn_tail import (
    SymbolicFinalTurnPlanner,
    final_own_turn_win,
)
from tests.test_symbolic_final_turn_tail import dynamics, setup


@pytest.mark.parametrize("swap", [False, True])
def test_key_predicate_matches_public_state_predicate_and_boundaries(swap):
    _, root, worlds = setup(swap)
    viewer = worlds.viewer
    states = np.repeat(root.state[None], 6, axis=0)
    states[:, TURN] = 15
    states[1, POINTS + viewer] = states[1, POINTS + 1 - viewer] + 2
    states[2, TURN] = 14
    states[3, ROUND] = 6
    states[4, PLAYER] = 1 - viewer
    states[5, PHASE] = TERMINAL
    expected = final_own_turn_win(states, viewer).tolist()
    keys = visible_keys(states, viewer)
    assert [certified_own_win_key(k, viewer) for k in keys] == expected
    states[:, :96] = states[:, :96][:, ::-1]
    assert [
        certified_own_win_key(k, viewer) for k in visible_keys(states, viewer)
    ] == expected
    with pytest.raises(ValueError, match="another viewer"):
        certified_own_win_key(keys[0], 1 - viewer)
    with pytest.raises(ValueError, match="complete own"):
        certified_own_win_key(keys[0][:-4], viewer)


def phases(symbolic):
    actor, _, worlds = setup()
    fixed, optimized, _ = evaluate_controller(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        **({"planner_type": SymbolicFinalTurnPlanner} if symbolic else {}),
    )
    return {"fixed": fixed, "optimized": optimized}


def test_completed_comparison_checks_omitted_keys_and_exact_retained_programs():
    old, new = phases(False), phases(True)
    counted = compare_common(old, new)
    assert counted["common_completed_root_values"] == 4
    assert counted["omitted_certified_entries"] > counted["retained_explicit_entries"]
    assert all(p.result.refusal is None for p in new.values())

    bad = deepcopy(new)
    plan = bad["fixed"].result.plan
    action = plan.action
    plan.values[action] += Fraction(1, 100)
    with pytest.raises(ValueError, match="root value"):
        compare_common(old, bad)
    bad = deepcopy(new)
    plan = bad["fixed"].result.plan
    root = bad["fixed"].binding.root_key
    del plan.root_policies[plan.action][root]
    with pytest.raises(ValueError, match="explicit root"):
        compare_common(old, bad)
    bad = deepcopy(new)
    plan = bad["fixed"].result.plan
    other = next(k for k in plan.root_policies[plan.action] if k != root)
    del plan.root_policies[plan.action][other]
    with pytest.raises(ValueError, match="outside the certified"):
        compare_common(old, bad)
    bad = deepcopy(new)
    plan = bad["fixed"].result.plan
    plan.root_policies[plan.action][other] = -1
    with pytest.raises(ValueError, match="explicit action"):
        compare_common(old, bad)
    bad = deepcopy(new)
    plan = bad["fixed"].result.plan
    plan.root_policies[plan.action][b"not-in-original"] = 0
    with pytest.raises(ValueError, match="introduced"):
        compare_common(old, bad)
    bad = deepcopy(new)
    bad["fixed"].result.plan.leaf_rule = "unknown"
    with pytest.raises(ValueError, match="recognized"):
        compare_common(old, bad)
    bad = deepcopy(new)
    bad["fixed"].binding = replace(bad["fixed"].binding, digest="different")
    with pytest.raises(ValueError, match="same belief"):
        compare_common(old, bad)


def test_uncompleted_roots_are_not_counted_as_equivalence_evidence():
    old, new = phases(False), phases(True)
    old = deepcopy(old)
    for phase in old.values():
        phase.result.plan = None
    assert compare_common(old, new) == {
        "common_completed_root_values": 0,
        "retained_explicit_entries": 0,
        "omitted_certified_entries": 0,
    }
