import numpy as np
import pytest

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.final_round_actual_match import play_final_round_match


class StopFirst:
    def act(self, env):
        legal = np.flatnonzero(env.legal())
        return 48 if 48 in legal else int(legal[0])


class ContinueFirst:
    def act(self, env):
        legal = np.flatnonzero(env.legal())
        return 49 if 49 in legal else int(legal[0])


def independent_match(seed, seat, override=None, actor=None):
    env = ReferenceEnv(seed, "paper8")
    actor = actor or StopFirst()
    states, actions = {}, []
    rounds = 0
    while env.state[PHASE] != TERMINAL:
        legal = np.flatnonzero(env.legal())
        if (
            env.state[ROUND] == 7
            and env.state[PLAYER] == seat
            and env.state[PHASE] < 3
            and len(legal) > 1
        ):
            states[len(actions)] = env.state.copy()
        action = (
            override[1]
            if override is not None and len(actions) == override[0]
            else actor.act(env)
        )
        result = env.step(action)
        rounds += int(result[1])
        actions.append(action)
    return states, actions, env.state[POINTS : POINTS + 2].tolist(), rounds


@pytest.mark.parametrize("seat", [0, 1])
def test_natural_roots_single_intervention_and_complete_accounting(seat):
    native = BatchEnv(1, "cpu", "paper8")
    actor = StopFirst()
    case = {"seed": 31010001, "seat": seat}
    expected, actions, points, rounds = independent_match(case["seed"], seat)
    roots, row = play_final_round_match(case, actor, actor, native)
    assert roots
    assert [i for i, _ in roots] == list(expected)
    for i, state in roots:
        np.testing.assert_array_equal(state, expected[i])
    assert row["actions"] == actions
    assert row["terminal_points"] == points
    assert row["work"] == {
        "decisions": len(actions),
        "round_endings": rounds,
        "completed_matches": 1,
    }
    # A later root verifies that collection does not implicitly select the first.
    index, state = roots[-1]
    env = ReferenceEnv(case["seed"], "paper8")
    env.state[:] = state
    alternative = next(
        int(a) for a in np.flatnonzero(env.legal()) if a != actions[index]
    )
    _, changed = play_final_round_match(
        case, actor, actor, native, intervention=(index, state, alternative)
    )
    _, expected_actions, expected_points, expected_rounds = independent_match(
        case["seed"], seat, (index, alternative)
    )
    assert changed["actions"] == expected_actions
    assert changed["actions"][:index] == actions[:index]
    assert changed["terminal_points"] == expected_points
    assert changed["work"]["round_endings"] == expected_rounds
    assert changed["work"]["completed_matches"] == 1
    assert changed["intervention_indices"] == [index]
    with pytest.raises(ValueError, match="illegal"):
        play_final_round_match(
            case, actor, actor, native, intervention=(index, state, 49)
        )
    with pytest.raises(ValueError, match="never reached"):
        play_final_round_match(
            case, actor, actor, native, intervention=(1024, state, alternative)
        )
    bad_state = state.copy()
    bad_state[POINTS] += 1
    with pytest.raises(AssertionError):
        play_final_round_match(
            case, actor, actor, native, intervention=(index, bad_state, alternative)
        )


@pytest.mark.parametrize("seat", [0, 1])
def test_early_bankruptcy_remains_a_complete_match_without_opportunities(seat):
    case = {"seed": 19, "seat": seat}
    actor = ContinueFirst()
    native = BatchEnv(1, "cpu", "paper8")
    expected, actions, points, rounds = independent_match(19, seat, actor=actor)
    roots, row = play_final_round_match(case, actor, actor, native)
    assert not expected and not roots
    assert row["eligible_decision_indices"] == row["intervention_indices"] == []
    assert row["selected_action"] is None and row["terminal_round"] < 7
    assert row["actions"] == actions and row["terminal_points"] == points
    assert row["work"] == {
        "decisions": len(actions),
        "round_endings": rounds,
        "completed_matches": 1,
    }
    initial = ReferenceEnv(19, "paper8")
    with pytest.raises(ValueError, match="unforced final-round"):
        play_final_round_match(
            case,
            actor,
            actor,
            native,
            intervention=(0, initial.state.copy(), actor.act(initial)),
        )
