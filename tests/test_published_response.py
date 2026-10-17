"""Response PPO: actual opponent actions, signed returns and restart state."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH
from scripts.published_response import ResponseAgent, ResponseConfig, ResponseTrainer

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(device="cpu", **overrides):
    return ResponseConfig(
        **(
            {
                "device": device,
                "envs": 4,
                "horizon": 160,
                "minibatch": 160,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 1,
                "seed": 19870001,
            }
            | overrides
        )
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_reference_roles_actions_returns_and_accounting(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = ResponseTrainer(config(device))
    refs = []
    for row in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = row.copy()
        refs.append(ref)
    with torch.no_grad():
        trainer.model.value[-1].bias.fill_(0.12)
        trainer.opponent_value.network[-1].bias.fill_(-0.19)
    step = trainer.env.step
    trace = []
    work = {k: {"decisions": 0, "rounds": 0, "matches": 0} for k in ("wp", "point")}
    roles_seen = set()

    def checked(actions):
        rows = []
        for lane, ref in enumerate(refs):
            episode = int(ref.state[EPISODE])
            seat = (episode + lane) % 2
            name = ("wp", "point")[(lane // 2 + episode // 2) % 2]
            learner = int(ref.state[PLAYER]) == seat
            roles_seen.add((name, seat))
            rows.append((learner, int(ref.state[PLAYER]), name))
        # Compare against each full-batch frozen expert rather than the routing helper.
        for name, opponent in trainer.frozen.agents.items():
            expected = opponent.agent.act(trainer.env)
            for lane, (learner, _, assigned) in enumerate(rows):
                if not learner and assigned == name:
                    assert int(actions[lane]) == int(expected[lane])
        expected_results = [r.step(a) for r, a in zip(refs, actions.cpu().tolist())]
        result = step(actions)
        np.testing.assert_array_equal(result.cpu(), expected_results)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        trace.append((rows, expected_results))
        for lane, ((learner, _, name), events) in enumerate(
            zip(rows, expected_results)
        ):
            work[name]["decisions"] += int(not learner)
            work[name]["rounds"] += int(events[1])
            work[name]["matches"] += int(events[2])
            if refs[lane].state[PHASE] == 4:
                refs[lane] = ReferenceEnv(
                    int(refs[lane].state[SEED]),
                    "paper8",
                    int(refs[lane].state[EPISODE]) + 1,
                )
        return result

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    h, n = trainer.config.horizon, trainer.config.envs
    roles = np.array([[r[0] for r in rows] for rows, _ in trace])
    players = np.array([[r[1] for r in rows] for rows, _ in trace])
    results = np.array([events for _, events in trace])
    np.testing.assert_array_equal(batch["learner"].cpu(), roles.flatten())
    values = np.where(roles, 0.12, -0.19) * (1 - 2 * players)
    with torch.no_grad():
        obs, mask = trainer.observe()
        tail = torch.where(
            trainer.learner_rows(), trainer.actor(obs, mask)[1], trainer.role_value(obs)
        )
    future = tail.cpu().numpy() * (1 - 2 * trainer.env.state[:, PLAYER].cpu().numpy())
    carry = np.zeros(n)
    expected = np.empty((h, n))
    for t in reversed(range(h)):
        live = 1 - results[t, :, 2]
        carry = (
            results[t, :, 0]
            + live * future
            - values[t]
            + live * trainer.config.gae_lambda * carry
        )
        expected[t] = (values[t] + carry) * (1 - 2 * players[t])
        future = values[t]
    np.testing.assert_allclose(
        batch["return"].cpu(), expected.flatten(), atol=3e-6, rtol=0
    )
    assert roles_seen == {(name, seat) for name in work for seat in (0, 1)}
    assert work == trainer.opponent_work
    assert trainer.counters["learner_decisions"] == int(roles.sum())
    assert trainer.counters["published_decisions"] == h * n - int(roles.sum())
    assert trainer.counters["published_matches"] == trainer.counters["matches"] > 0
    assert trainer.counters["published_rounds"] == trainer.counters["rounds"]
    assert trainer.env.step == checked


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_exact_restart_and_frozen_opponents(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = ResponseTrainer(config(device, horizon=32, minibatch=64))
    original = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    opponents = copy.deepcopy(trainer.frozen.checkpoint())
    assert trainer.update()["critic_only"]
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] == original
    path = tmp_path / "response.pt"
    trainer.save(path)
    metrics = trainer.update()
    assert not metrics["critic_only"]
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] != original
    model = copy.deepcopy(trainer.trainable.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    state = trainer.env.state.clone()
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    restored = ResponseTrainer.restore(path)
    assert restored.update() == metrics
    assert restored.counters == trainer.counters
    assert restored.opponent_work == trainer.opponent_work
    assert torch.equal(restored.env.state, state)
    assert torch.equal(torch.get_rng_state(), rng)
    for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()):
        assert torch.equal(before, after)
    for key, value in model.items():
        assert torch.equal(value, restored.trainable.state_dict()[key])
    actual = restored.optimizer.state_dict()
    assert actual["param_groups"] == optimizer["param_groups"]
    for key, items in optimizer["state"].items():
        for name, value in items.items():
            assert torch.equal(value, actual["state"][key][name])
    for name, saved in opponents.items():
        now = restored.frozen.checkpoint()[name]
        assert saved["agent"]["parameters_sha256"] == now["agent"]["parameters_sha256"]
        assert all(
            not p.requires_grad
            for m in restored.frozen.agents[name].agent.models.values()
            for p in m.parameters()
        )


def test_opponent_rows_do_not_update_actor():
    trainer = ResponseTrainer(config(horizon=4, critic_warmup_updates=0))
    batch = trainer.collect()
    batch["learner"].zero_()
    batch["advantage"].fill_(100)
    batch["return"].fill_(1)
    before = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    trainer.minibatch_step(batch)
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] == before
    assert trainer.opponent_value.network[-1].bias.abs().sum() > 0


def test_config_rejections_and_single_opponent():
    for changes in (
        {"opponent_fraction": 0},
        {"opponent_fraction": 2},
        {"published_opponent": "bad"},
        {"cuda_graph": True},
    ):
        with pytest.raises(ValueError):
            replace(config(), **changes)
    trainer = ResponseTrainer(
        config(horizon=4, published_opponent="point", opponent_fraction=0.5)
    )
    batch = trainer.collect()
    assert set(trainer.opponent_work) == {"point"}
    assert batch["learner"].reshape(4, 4)[:, 2:].all()
    assert (
        trainer.counters["published_decisions"]
        == trainer.opponent_work["point"]["decisions"]
    )


def test_actual_parent_initial_actions_and_deployment(tmp_path):
    path = "runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt"
    from pathlib import Path

    if not Path(path).exists():
        pytest.skip("Retained research parent not available")
    trainer = ResponseTrainer.from_parent(config(horizon=8, critic_width=256), path)
    checkpoint = tmp_path / "initial.pt"
    trainer.save(checkpoint)
    agent = ResponseAgent(checkpoint)
    parent = NeuralAgent(path, deterministic=True)
    phases = set()
    env = ReferenceEnv(19871001, "paper8")
    for _ in range(256):
        phases.add(int(env.state[PHASE]))
        action = agent.act(env)
        assert action == parent.act(env)
        env.step(action)
        if env.state[PHASE] == 4:
            env = ReferenceEnv(19871001, "paper8", int(env.state[EPISODE]) + 1)
    assert phases == {0, 1, 2, 3}
    restored = ResponseTrainer.restore(checkpoint)
    assert (
        restored.provenance["weight_initialization"]
        == trainer.provenance["weight_initialization"]
    )
    assert all(value == 0 for value in restored.counters.values())
