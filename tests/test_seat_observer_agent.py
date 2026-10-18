"""Actual retained-parent and capability-model trajectories through inference."""

from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import GreedyAgent, NeuralAgent, solved_koi_action
from koikoi.evaluate import duel
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from scripts.seat_observer_agent import SeatObserverAgent


@pytest.mark.parametrize("parent", [1, 2])
@pytest.mark.parametrize("context", ["public", "masked"])
def test_initial_retained_parent_actions_and_hidden_invariance(parent, context):
    torch.set_num_threads(2)
    path = Path(
        f"runs/seat_observer_capability_cpu/parent{parent}_{context}_initial.pt"
    )
    if not path.exists():
        pytest.skip("Actual-parent CPU capability unavailable")
    agent = SeatObserverAgent(path)
    baseline = NeuralAgent(
        Path(
            f"runs/cold_ppo_screen/seed{6700001 + parent * 10000}_warm/update_00000520.pt"
        ),
        deterministic=True,
    )
    ref = ReferenceEnv(26093001 + parent, "paper8")
    phases = set()
    rounds = matches = 0
    for _ in range(256):
        original = ref.state.copy()
        phases.add(int(original[PHASE]))
        action = agent.act(ref)
        assert action == baseline.act(ref)
        assert ref.legal()[action]
        np.testing.assert_array_equal(original, ref.state)
        alternate = ReferenceEnv.__new__(ReferenceEnv)
        alternate.state = original.copy()
        seat = int(original[PLAYER])
        hidden = np.flatnonzero((original[:48] == 1 - seat) | (original[:48] == -1))
        alternate.state[hidden] = alternate.state[hidden][::-1]
        alternate.state[72:96] = alternate.state[72:96][::-1]
        alternate.state[SEED] += 117
        assert agent.act(alternate) == action
        result = ref.step(action)
        rounds += int(result[1])
        matches += int(result[2])
        if ref.state[PHASE] == 4:
            ref = ReferenceEnv(
                int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
            )
    assert phases == {0, 1, 2, 3} and matches > 0
    print(
        {
            "initial": [parent, context],
            "decisions": 256,
            "rounds": rounds,
            "matches": matches,
        }
    )


@pytest.mark.parametrize("parent", [1, 2])
@pytest.mark.parametrize("context", ["public", "masked"])
def test_trained_checkpoint_actor_route_and_complete_duplicate_matches(parent, context):
    torch.set_num_threads(2)
    path = Path(f"runs/seat_observer_capability_cpu/parent{parent}_{context}_final.pt")
    if not path.exists():
        pytest.skip("Actual-parent CPU capability unavailable")
    agent = SeatObserverAgent(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (
        agent.model.actor.checkpoint()["current"]["parameters_sha256"]
        == payload["actor"]["current"]["parameters_sha256"]
    )
    phases = set()
    native = agent.act

    def checked(ref):
        phases.add(int(ref.state[PHASE]))
        action = native(ref)
        assert ref.legal()[action]
        solved = solved_koi_action(ref) if agent.requires_endgame else None
        if solved is not None:
            assert action == solved
        else:
            agent.published_env.state.copy_(torch.from_numpy(ref.state).unsqueeze(0))
            obs, mask = packed_observation(agent.published_env)
            with torch.no_grad():
                scores = agent.model.actor(
                    obs[:, 561:].reshape(-1, 300, 50), obs[:, 560].int(), mask
                )
                logits = (scores / agent.model.temperature).masked_fill(~mask, -1e9)
            assert action == int(logits.argmax(-1))
        return action

    agent.act = checked
    decisions = 0
    for flip in (False, True):
        a, b = (GreedyAgent(), agent) if flip else (agent, GreedyAgent())
        result = duel(a, b, 26094001 + parent, "paper8")
        decisions += result[3]
        assert result[0] in (0, 0.5, 1)
        assert result[0] == (1 if result[1] > 0 else 0 if result[1] < 0 else 0.5)
    assert phases == {0, 1, 2, 3}
    print({"final": [parent, context], "matches": 2, "decisions": decisions})
