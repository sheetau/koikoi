from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
import torch

from koikoi.layout import (
    DEALER,
    EPISODE,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    SEED,
    STOCK,
    TERMINAL,
)
from scripts.belief_tail import (
    FiniteBeliefPlanner,
    KoiTailDynamics,
    TailBudgetExceeded,
    WorldSet,
    enumerate_tail_worlds,
    visible_keys,
)
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_worlds import enumerate_support
from tests.test_history_conditioned_worlds import Extremal, late_root


class GuessingGame:
    """Safe draw or a later guess; only the reveal variant discloses the bit."""

    def __init__(self, reveal):
        self.reveal = reveal

    def keys(self, states, viewer):
        return [
            (int(row[0]), int(row[1]) if self.reveal and row[0] == 1 else -1)
            for row in states
        ]

    @staticmethod
    def players(states):
        return np.zeros(len(states), dtype=np.int32)

    @staticmethod
    def terminal(states):
        return states[:, 0] == 2

    @staticmethod
    def outcomes(states, viewer):
        return states[:, 2]

    @staticmethod
    def legal(state):
        return (0, 1)

    @staticmethod
    def baseline_action(state):
        return 0

    @staticmethod
    def advance(states, actions):
        after = states.copy()
        for i, (state, action) in enumerate(zip(states, actions, strict=True)):
            if state[0] == 0:
                after[i, 0] = 2 if action == 0 else 1
                after[i, 2] = 0
            else:
                after[i, 0] = 2
                after[i, 2] = 1 if action == state[1] else -1
        return after, {
            "decisions": len(states),
            "round_endings": int((after[:, 0] == 2).sum()),
            "completions": int((after[:, 0] == 2).sum()),
        }


@pytest.mark.parametrize("reveal", [False, True])
@pytest.mark.parametrize("masses", [(1, 1), (3, 1)])
def test_exact_information_policy_matches_independent_strategy_enumeration(
    reveal, masses
):
    dynamics = GuessingGame(reveal)
    worlds = WorldSet(
        np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32),
        np.zeros(2, dtype=np.int64),
        np.array(masses, dtype=np.int64),
        ("unused",),
        0,
        0,
    )
    plan = FiniteBeliefPlanner(dynamics).solve(worlds)
    # Independently enumerate every legal information-contingent pure strategy.
    payoffs = []
    for root, guess0, guess1 in product((0, 1), repeat=3):
        if not reveal and guess0 != guess1:
            continue
        value = sum(
            weight
            * (0 if root == 0 else (1 if (guess0, guess1)[hidden] == hidden else -1))
            for hidden, weight in enumerate(masses)
        )
        payoffs.append(F(value + sum(masses), 2 * sum(masses)))
    assert plan.values[plan.action] == max(payoffs)
    if reveal:
        assert plan.action == 1 and plan.values[1] == 1
        assert plan.policy[(1, 0)] == 0 and plan.policy[(1, 1)] == 1
    elif masses == (1, 1):
        assert plan.action == 0 and plan.values[1] == F(1, 2)
        # A hidden-world maximum would incorrectly assign value 1 to action 1.
        assert plan.values[1] < 1
    fixed = FiniteBeliefPlanner(dynamics).solve(worlds, optimize_continuation=False)
    assert plan.values[plan.action] >= fixed.values[fixed.action]
    reverse = WorldSet(
        worlds.states[::-1],
        worlds.models[::-1],
        worlds.weights[::-1],
        worlds.model_names,
        0,
        0,
    )
    again = FiniteBeliefPlanner(dynamics).solve(reverse)
    assert (again.action, again.values, again.policy) == (
        plan.action,
        plan.values,
        plan.policy,
    )
    with pytest.raises(TailBudgetExceeded):
        FiniteBeliefPlanner(dynamics, max_nodes=1).solve(worlds)


def swap_seats(env):
    env = reference(env.state)
    s = env.state
    locations = s[:48].copy()
    for old, new in ((0, 1), (1, 0), (3, 4), (4, 3)):
        s[:48][locations == old] = new
    s[48:64] = np.r_[s[56:64], s[48:56]]
    s[PLAYER], s[DEALER] = 1 - s[PLAYER], 1 - s[DEALER]
    s[POINTS : POINTS + 2] = s[POINTS : POINTS + 2][::-1]
    s[108:110] = s[108:110][::-1]
    return env


@pytest.mark.parametrize("swap", [False, True])
def test_complete_prefixes_native_reference_replay_and_executable_plan(swap):
    torch.set_num_threads(2)
    root = late_root()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    original = root.state.copy()
    agent = Extremal()
    support = enumerate_support(root, {"first": agent}, max_hands=100)
    worlds = enumerate_tail_worlds(root, support)
    assert worlds.draws == 3 and len(worlds.states) == 4950
    assert all(
        key == visible_keys(root.state[None], worlds.viewer)[0]
        for key in visible_keys(worlds.states, worlds.viewer)
    )
    # Every compatible hand has all 11*10*9 equiprobable three-draw prefixes.
    seen = {}
    for state in worlds.states:
        hand = tuple(np.flatnonzero(state[:48] == 1 - worlds.viewer))
        top = int(state[STOCK]) + 48
        seen.setdefault(hand, set()).add(tuple(state[top - 2 : top + 1][::-1]))
    assert all(len(prefixes) == 990 for prefixes in seen.values())
    assert len(seen) == 5 and np.all(worlds.weights == 1)

    dynamics = KoiTailDynamics(
        len(worlds.states), worlds.model_names, {"first": agent}, agent
    )
    native_advance = dynamics.advance
    verified_steps = 0

    def checked(states, actions):
        nonlocal verified_steps
        after, work = native_advance(states, actions)
        independent = {"decisions": len(states), "round_endings": 0, "completions": 0}
        for state, action, actual in zip(states, actions, after, strict=True):
            env = reference(state)
            result = env.step(int(action))
            np.testing.assert_array_equal(env.state, actual)
            independent["round_endings"] += int(result[1])
            independent["completions"] += int(result[2])
        assert work == independent
        verified_steps += len(states)
        return after, work

    dynamics.advance = checked
    optimized = FiniteBeliefPlanner(dynamics).solve(worlds)
    fixed = FiniteBeliefPlanner(dynamics).solve(worlds, optimize_continuation=False)
    assert optimized.values[optimized.action] >= fixed.values[fixed.action]
    assert verified_steps == optimized.work["decisions"] + fixed.work["decisions"]

    # Execute every root action and its whole policy in the independent engine.
    for plan in (optimized, fixed):
        for first_action, policy in plan.root_policies.items():
            signed = 0
            for state, weight in zip(worlds.states, worlds.weights, strict=True):
                env = reference(state)
                for _ in range(32):
                    if env.state[PHASE] == TERMINAL:
                        break
                    action = (
                        policy[visible_keys(env.state[None], worlds.viewer)[0]]
                        if env.state[PLAYER] == worlds.viewer
                        else agent.act(env)
                    )
                    env.step(action)
                assert env.state[PHASE] == TERMINAL
                signed += int(weight) * int(
                    np.sign(
                        env.state[POINTS + worlds.viewer]
                        - env.state[POINTS + 1 - worlds.viewer]
                    )
                )
            mass = int(worlds.weights.sum())
            assert F(signed + mass, 2 * mass) == plan.values[first_action]
    np.testing.assert_array_equal(root.state, original)

    # Changing the actual hidden hand/stock and RNG leaves enumeration unchanged.
    alternate = reference(worlds.states[-1])
    alternate.state[SEED] = 987654
    alternate.state[EPISODE] = 17
    stock_end = 49 + int(alternate.state[STOCK])
    alternate.state[72:stock_end] = alternate.state[72:stock_end][::-1]
    again = enumerate_tail_worlds(alternate, support)
    np.testing.assert_array_equal(again.states, worlds.states)
    np.testing.assert_array_equal(again.weights, worlds.weights)
    with pytest.raises(TailBudgetExceeded, match="draw"):
        enumerate_tail_worlds(root, support, max_draws=2)
    with pytest.raises(TailBudgetExceeded, match="world"):
        enumerate_tail_worlds(root, support, max_worlds=len(worlds.states) - 1)
    with pytest.raises(ValueError, match="positive-prior"):
        enumerate_tail_worlds(root, support, prior={"first": 0})
