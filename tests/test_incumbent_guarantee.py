from fractions import Fraction

import numpy as np

from scripts.deferred_belief_tail import DeferredWorldSet
from scripts.incumbent_tail_controller import evaluate_controller
from tests.test_bounded_belief_tail import DelayedLossGame
from tests.test_incumbent_belief_tail import IncumbentToy


class TradeoffGame(DelayedLossGame):
    # Columns are complete root policies, rows are hidden worlds.
    wins = np.array([[1, 0], [1, 1], [0, 1], [0, 1]])

    def advance(self, states, actions):
        after = states.copy()
        after[:, 0] = 3
        after[:, 2] = 2 * self.wins[states[:, 1], actions] - 1
        return after, dict.fromkeys(
            ("decisions", "round_endings", "completions"), len(states)
        )


def expectation(outcomes, masses):
    return sum(
        Fraction(int(w) * int(y), sum(masses))
        for y, w in zip(outcomes, masses, strict=True)
    )


def test_exact_switching_identity_and_limits_of_the_model_guarantee():
    dynamics = TradeoffGame(False)
    worlds = DeferredWorldSet(
        np.array([[0, w, 0] for w in range(4)], dtype=np.int32),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.int64),
        ("unused",),
        0,
        0,
    )
    _, _, selection = evaluate_controller(
        worlds,
        dynamics,
        "parent",
        {"unused": "opponent"},
        planner_type=IncumbentToy,
    )
    parent_outcomes = dynamics.wins[:, 0]
    selected_outcomes = dynamics.wins[:, selection.plan.action]
    parent = expectation(parent_outcomes, (1, 1, 1, 1))
    candidate = expectation(selected_outcomes, (1, 1, 1, 1))
    assert selection.parent_q == parent == Fraction(1, 2)
    assert selection.selected_q == candidate == Fraction(3, 4)
    gate_probability, outside_value = Fraction(3, 10), Fraction(2, 5)
    before = gate_probability * parent + (1 - gate_probability) * outside_value
    after = gate_probability * candidate + (1 - gate_probability) * outside_value
    assert after - before == gate_probability * selection.gain == Fraction(3, 40)
    # Positive expected improvement can lose on a particular hidden world.
    assert selected_outcomes[0] < parent_outcomes[0]
    # Changing the true law without changing the declared evaluator invalidates
    # the guarantee; this is not a robust or per-opponent worst-case certificate.
    misspecified = (8, 1, 1, 1)
    assert expectation(selected_outcomes, misspecified) - expectation(
        parent_outcomes, misspecified
    ) == -Fraction(6, 11)
