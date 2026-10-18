"""Online activation agrees with natural boundaries for either dealer seat."""

from dataclasses import replace

import pytest

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, ROUND, TURN
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.public_memory_continuation import ContinuationConfig, ContinuationTrainer
from scripts.public_memory_continuation_match import (
    ContinuationAgent,
    play_continuation_match,
)
from scripts.revealed_opponent_models import StrictPublishedPrediction
from tests.test_final_round_training import PARENT
from tests.test_public_memory_continuation import bank


@pytest.fixture
def initial_agents(tmp_path):
    config = ContinuationConfig(device="cpu", envs=8, critic_width=256)
    paths = []
    for public in (True, False):
        trainer = ContinuationTrainer.from_parent(
            replace(config, public_memory=public), PARENT, *bank()
        )
        path = tmp_path / f"initial_{public}.pt"
        trainer.save(path)
        paths.append(path)
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    predictors = {n: StrictPublishedPrediction(a) for n, a in models.items()}
    return paths, models, predictors


def test_initial_complete_matches_and_exact_activation(initial_agents):
    paths, models, predictors = initial_agents
    agents = [ContinuationAgent(path, predictors) for path in paths]
    parent = NeuralAgent(PARENT, deterministic=True)
    native = BatchEnv(1, "cpu", "paper8")
    dealer_owns = set()
    activated_before_final = 0
    for seat in (0, 1):
        for name, model in models.items():
            case = {"seat": seat, "seed": 88110001, "opponent": name}
            opponent = EndgameAgent(model)
            baseline = play_continuation_match(case, parent, opponent, native)
            records = [
                play_continuation_match(case, actor, opponent, native)
                for actor in agents
            ]
            assert records[0] == records[1]
            for actual in records:
                for key in ("actions", "terminal_points", "terminal_round", "work"):
                    assert actual[key] == baseline[key]
                first = actual["memory"]["first_identification"]
                expected = None
                ref = ReferenceEnv(case["seed"], "paper8")
                for decision, action in enumerate(actual["actions"]):
                    if (
                        expected is None
                        and first is not None
                        and decision >= first["available_after_decision"]
                        and ref.state[ROUND] > first["round"]
                        and ref.state[TURN] == 0
                        and ref.state[PHASE] == 0
                    ):
                        expected = {
                            "decision": decision,
                            "round": int(ref.state[ROUND]),
                            "candidate": first["model"],
                            "code": {"wp": 0, "point": 1}[first["model"]],
                        }
                        dealer_owns.add(int(ref.state[PLAYER]) == seat)
                        activated_before_final += int(ref.state[ROUND] < 7)
                    ref.step(action)
                assert actual["memory"]["activation"] == expected
                if expected is not None:
                    assert expected["candidate"] == name
                    assert all(
                        event["available_after_decision"] <= expected["decision"]
                        for event in actual["memory"]["events"]
                    )
    assert activated_before_final > 0
    assert dealer_owns == {False, True}


def test_ambiguous_fallback_reset_and_public_lifecycle(initial_agents):
    paths, models, predictors = initial_agents
    # Two identical candidate functions deliberately remain ambiguous against WP.
    actor = ContinuationAgent(
        paths[0], {"wp": predictors["wp"], "point": predictors["wp"]}
    )
    parent = NeuralAgent(PARENT, deterministic=True)
    opponent = EndgameAgent(models["wp"])
    native = BatchEnv(1, "cpu", "paper8")
    case = {"seat": 0, "seed": 88110001, "opponent": "wp"}
    original = play_continuation_match(case, parent, opponent, native)
    actual = play_continuation_match(case, actor, opponent, native)
    assert actual["actions"] == original["actions"]
    assert actual["memory"]["activation"] is None
    assert actual["memory"]["candidates"] == ["point", "wp"]
    actor.start_match("new", 0)
    assert (
        actor.activation is None and actor.context is None and not actor.tracker.events
    )
    with pytest.raises(ValueError, match="correct match"):
        actor.begin_decision(
            match_key="old", decision=0, round_index=0, turn=0, phase=0
        )
    with pytest.raises(ValueError, match="nonsequential"):
        actor.begin_decision(
            match_key="new", decision=1, round_index=0, turn=0, phase=0
        )
    actor.begin_decision(match_key="new", decision=0, round_index=0, turn=0, phase=0)
    with pytest.raises(ValueError, match="nonsequential"):
        actor.begin_decision(
            match_key="new", decision=0, round_index=0, turn=0, phase=0
        )
