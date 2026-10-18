"""Parent-equivalent public inference and loading both experimental arms."""

from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer
from scripts.observer_selfplay_agent import ObserverAgent


@pytest.mark.upstream
def test_initial_parent_actions_and_hidden_world_invariance(tmp_path):
    path = Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt")
    if not path.exists():
        pytest.skip("Retained parent unavailable")
    torch.set_num_threads(2)
    initial = tmp_path / "initial.pt"
    trainer = ObserverTrainer.from_parent(
        ObserverConfig(device="cpu", envs=4, horizon=16), path
    )
    trainer.save(initial)
    agent, parent = ObserverAgent(initial), NeuralAgent(path, deterministic=True)
    env = ReferenceEnv(26050001, "paper8")
    phases, matches = set(), 0
    for _ in range(256):
        before = env.state.copy()
        phases.add(int(before[PHASE]))
        action = agent.act(env)
        assert action == parent.act(env)
        np.testing.assert_array_equal(before, env.state)
        alternate = ReferenceEnv.__new__(ReferenceEnv)
        alternate.state = before.copy()
        seat = int(before[PLAYER])
        hidden = np.flatnonzero((before[:48] == 1 - seat) | (before[:48] == -1))
        alternate.state[hidden] = alternate.state[hidden][::-1]
        alternate.state[72:96] = alternate.state[72:96][::-1]
        alternate.state[SEED] += 117
        assert agent.act(alternate) == action
        result = env.step(action)
        matches += result[2]
        if env.state[PHASE] == 4:
            env = ReferenceEnv(
                int(env.state[SEED]), "paper8", int(env.state[EPISODE]) + 1
            )
    assert phases == {0, 1, 2, 3} and matches > 0


@pytest.mark.parametrize("mode", ["observer", "signed"])
def test_capability_endpoint_logits_are_preserved(mode):
    path = Path(f"runs/observer_selfplay_capability_cuda/{mode}_final.pt")
    if not path.exists():
        pytest.skip("Capability endpoint unavailable")
    from koikoi.published_train import packed_observation

    torch.set_num_threads(2)
    agent = ObserverAgent(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (
        agent.model.actor.checkpoint()["current"]["parameters_sha256"]
        == payload["actor"]["current"]["parameters_sha256"]
    )
    env = ReferenceEnv(26050002, "paper8")
    phases = set()
    for _ in range(128):
        phases.add(int(env.state[PHASE]))
        agent.published_env.state.copy_(torch.from_numpy(env.state).unsqueeze(0))
        obs, mask = packed_observation(agent.published_env)
        with torch.no_grad():
            scores = agent.model.actor(
                obs[:, 561:].reshape(-1, 300, 50), obs[:, 560].int(), mask
            )
            logits = agent.model(obs, mask)[0]
        torch.testing.assert_close(
            logits,
            (scores / agent.model.temperature).masked_fill(~mask, -1e9),
            rtol=0,
            atol=0,
        )
        env.step(agent.act(env))
        if env.state[PHASE] == 4:
            env = ReferenceEnv(
                int(env.state[SEED]), "paper8", int(env.state[EPISODE]) + 1
            )
    assert phases == {0, 1, 2, 3}
