import sys

import numpy as np
import pytest

from koikoi.agents import EndgameAgent, GreedyAgent, KoiAgent, RandomAgent
from koikoi.evaluate import duel, main, seed_agents
from koikoi.layout import KOI, PHASE, TERMINAL
from koikoi.reference import ReferenceEnv


def test_substitution_matches_independent_phase_selection_in_complete_match():
    class ContinueAgent:
        def act(self, env):
            return 49

    cards, koi = GreedyAgent(), ContinueAgent()
    hybrid = KoiAgent(cards, koi)
    actual, expected = ReferenceEnv(81, "paper8"), ReferenceEnv(81, "paper8")
    koi_count = 0
    while expected.state[PHASE] != TERMINAL:
        if expected.state[PHASE] == KOI:
            action = 49
            koi_count += 1
        else:
            action = cards.act(expected)
        assert hybrid.act(actual) == action
        assert expected.legal()[action]
        actual.step(action)
        expected.step(action)
    assert koi_count > 0
    np.testing.assert_array_equal(actual.state, expected.state)


def test_nested_substitution_replays_each_random_stream():
    cards, koi = RandomAgent(), RandomAgent()
    hybrid = EndgameAgent(KoiAgent(cards, EndgameAgent(koi)))
    opponent = GreedyAgent()
    seed_agents(hybrid, opponent, 963)
    first = duel(hybrid, opponent, 963, "paper8")
    cards.rng.random(97)
    koi.rng.random(43)
    seed_agents(hybrid, opponent, 963)
    second = duel(hybrid, opponent, 963, "paper8")
    assert first[:2] == second[:2]
    assert first[3:] == second[3:]


@pytest.mark.parametrize(
    "extra",
    [
        ["--profile", "research12"],
        ["--search", "uniform"],
    ],
)
def test_published_koi_substitution_rejects_invalid_protocol(monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--agent", "greedy", "--koi-agent", "rl-wp", *extra],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
