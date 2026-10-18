from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import HISTORY, KOI, PHASE, PLAYER, SEED, STOCK, TURN
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import canonical_world
from scripts.history_conditioned_worlds import (
    HistorySupport,
    enumerate_support,
    joint_probabilities,
    sample_joint_worlds,
)
from scripts.run_arbitration_study import PARENTS


class Extremal:
    def __init__(self, largest=False):
        self.largest = largest
        self.calls = 0

    def act(self, env):
        self.calls += 1
        legal = np.flatnonzero(env.legal())
        if 49 in legal:
            return 49
        return int(legal[-1 if self.largest else 0])


def late_root():
    env = ReferenceEnv(31010001, "paper8")
    agent = Extremal()
    for _ in range(2000):
        if env.state[TURN] == 13 and env.state[PHASE] == 0:
            return env
        env.step(agent.act(env))
    raise AssertionError("No late root found")


def test_joint_normalization_preserves_model_evidence_and_priors():
    support = HistorySupport(
        "x",
        ("a", "b"),
        "complete",
        3,
        3,
        0,
        ((1,), (2,), (3,)),
        ((True, True), (True, False), (True, False)),
        {},
    )
    p = joint_probabilities(support)
    np.testing.assert_array_equal(p, [[0.25, 0.25], [0.25, 0], [0.25, 0]])
    np.testing.assert_array_equal(p.sum(0), [0.75, 0.25])
    np.testing.assert_allclose(
        joint_probabilities(support, {"a": 1, "b": 3}),
        [[1 / 6, 1 / 2], [1 / 6, 0], [1 / 6, 0]],
    )
    with pytest.raises(ValueError, match="compatible"):
        joint_probabilities(replace(support, compatible=((False, False),) * 3))
    for prior in ({"a": 1}, {"a": 0, "b": 1}, {"a": np.nan, "b": 1}):
        with pytest.raises(ValueError):
            joint_probabilities(support, prior)


def test_public_budget_refuses_before_policy_calls():
    env = ReferenceEnv(31010001, "paper8")
    opponent = Extremal()
    support = enumerate_support(env, {"a": opponent}, max_hands=100)
    assert support.status == "too_many_hands" and opponent.calls == 0
    assert not support.hands and not support.counts
    with pytest.raises(ValueError, match="did not complete"):
        sample_joint_worlds(env, support, np.random.default_rng(1), 2)


def test_support_and_sample_do_not_depend_on_actual_hidden_world():
    env = late_root()
    before = env.state.copy()
    opponents = {"min": Extremal(), "max": Extremal(True)}
    support = enumerate_support(env, opponents)
    assert support.status == "complete" and support.candidate_hands <= 100
    actual = tuple(np.flatnonzero(env.state[:48] == 1 - env.state[PLAYER]))
    assert support.compatible[support.hands.index(actual)][
        support.model_names.index("min")
    ]
    assert support.valid_hands + support.invalid_hands == support.candidate_hands
    alternate = reference(sample_hidden(env, np.random.default_rng(61)))
    alternate_support = enumerate_support(alternate, opponents)
    assert alternate_support == support
    worlds, models = sample_joint_worlds(env, support, np.random.default_rng(62), 256)
    changed, changed_models = sample_joint_worlds(
        alternate, alternate_support, np.random.default_rng(62), 256
    )
    np.testing.assert_array_equal(worlds, changed)
    np.testing.assert_array_equal(models, changed_models)
    np.testing.assert_array_equal(env.state, before)
    assert len({int(s[SEED]) for s in worlds}) > 1
    assert len({s[72 : 49 + int(s[STOCK])].tobytes() for s in worlds}) > 1
    for world, model in zip(worlds, models):
        np.testing.assert_array_equal(np.sort(world[48:96]), np.arange(48))
        np.testing.assert_array_equal(reference(world).observe(), env.observe())
        hand = tuple(np.flatnonzero(world[:48] == 1 - world[PLAYER]))
        assert support.compatible[support.hands.index(hand)][model]
    other = ReferenceEnv(31010002, "paper8")
    with pytest.raises(ValueError, match="different public observation"):
        sample_joint_worlds(other, support, np.random.default_rng(1), 2)


def test_real_parent_and_point_support_matches_native_history_replay():
    torch.set_num_threads(2)
    states = np.load(
        "runs/observer_seat_information_v2/parent1_states.npy", allow_pickle=False
    )
    root = reference(states[13])  # fixed first saved root with twelve candidate hands
    opponents = {
        "parent": NeuralAgent(PARENTS[0][0], deterministic=True),
        "point": EndgameAgent(PublishedAgent("point")),
    }
    support = enumerate_support(root, opponents, max_hands=12)
    assert support.status == "complete" and support.candidate_hands == 12
    actual = tuple(np.flatnonzero(root.state[:48] == 1 - root.state[PLAYER]))
    assert support.compatible[support.hands.index(actual)][0]
    env = BatchEnv(1, "cpu", "paper8", seed=0)
    for hand, expected in zip(support.hands, support.compatible):
        target = canonical_world(root.state, hand)
        initial = reference(target)
        initial.deal(deck=target[48:96].copy())
        env.state.copy_(torch.from_numpy(initial.state[None]))
        compatible = [True, True]
        for _ in range(64):
            state = env.state[0].numpy().copy()
            if (state[TURN], state[PHASE]) == (target[TURN], target[PHASE]):
                break
            phase, turn = int(state[PHASE]), int(state[TURN])
            encoded = int(target[HISTORY + turn * 5 + (0, 1, 3, 4)[phase]])
            action = 48 + encoded - 1 if phase == KOI else encoded - 1
            ref = reference(state)
            if state[PLAYER] != target[PLAYER]:
                for i, name in enumerate(support.model_names):
                    compatible[i] &= opponents[name].act(ref) == action
            env.step(torch.tensor([action], dtype=torch.int64))
        else:
            raise AssertionError("Replay did not reach root")
        np.testing.assert_array_equal(env.state[0].numpy(), target)
        assert tuple(compatible) == expected
