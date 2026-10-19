import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from koikoi.agents import NeuralAgent
from koikoi.layout import PHASE, TERMINAL
from koikoi.published_train import (
    PublishedConfig,
    PublishedPolicyValue,
    PublishedTrainer,
)
from koikoi.reference import ReferenceEnv
from scripts.koi_temperature_ppo import (
    KoiTemperatureAgent,
    KoiTemperatureConfig,
    KoiTemperatureTrainer,
    KoiTemperatureValue,
)


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.arange(50, dtype=torch.float32) / 10)

    def forward(self, features, phases, mask):
        return self.score.expand(len(features), -1)


def test_only_koi_scaling_changes_and_gradients_have_expected_factor():
    old = PublishedPolicyValue.__new__(PublishedPolicyValue)
    nn.Module.__init__(old)
    old.actor, old.value, old.temperature = ToyActor(), nn.Linear(560, 1), 0.05
    new = KoiTemperatureValue(old, 0.2)
    obs = torch.zeros(4, 15561)
    obs[:, 560] = torch.arange(4)
    legal = torch.ones(4, 50, dtype=torch.bool)
    legal[:, 10] = False
    before = old(obs, legal)
    after = new(obs, legal)
    assert torch.equal(before[0][:3], after[0][:3])
    assert torch.equal(before[1], after[1])
    assert after[0][3, 11] == before[0][3, 11] / 4
    assert torch.equal(after[0][:, 10], torch.full((4,), -1e9))
    after[0][:, 11].sum().backward()
    assert new.actor.score.grad[11] == 3 / 0.05 + 1 / 0.2
    assert all(p.grad is None for p in new.value.parameters())
    with pytest.raises(ValueError):
        KoiTemperatureConfig(koi_temperature=0)


def tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            tensors_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            tensors_equal(a, b)
    else:
        assert left == right


def config(temperature):
    return KoiTemperatureConfig(
        device="cuda",
        seed=10770001,
        envs=8,
        horizon=64,
        minibatch=64,
        epochs=1,
        critic_warmup_updates=1,
        koi_temperature=temperature,
    )


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def test_cuda_all_equal_temperature_reproduces_original_updates():
    require_cuda()
    cfg = config(0.05)
    values = asdict(cfg)
    values.pop("koi_temperature")
    original = PublishedTrainer(PublishedConfig(**values))
    original_metrics = [original.update() for _ in range(3)]
    expected = copy.deepcopy(original.model.state_dict())
    optimizer = copy.deepcopy(original.optimizer.state_dict())
    state = original.env.state.clone()
    counters = original.counters.copy()
    candidate = KoiTemperatureTrainer(cfg)
    assert [candidate.update() for _ in range(3)] == original_metrics
    tensors_equal(candidate.model.state_dict(), expected)
    tensors_equal(candidate.optimizer.state_dict(), optimizer)
    assert torch.equal(candidate.env.state, state) and candidate.counters == counters


def test_cuda_calibrated_warmup_and_exact_restart(tmp_path):
    require_cuda()
    trainer = KoiTemperatureTrainer(config(0.21989584747027907))
    initial = copy.deepcopy(trainer.model.actor.state_dict())
    trainer.update()
    tensors_equal(initial, trainer.model.actor.state_dict())
    trainer.update()
    assert any(
        not torch.equal(v, trainer.model.actor.state_dict()[k])
        for k, v in initial.items()
    )
    checkpoint = tmp_path / "candidate.pt"
    trainer.save(checkpoint)
    expected_metrics = trainer.update()
    expected = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    state, counters = trainer.env.state.clone(), trainer.counters.copy()
    restored = KoiTemperatureTrainer.restore(checkpoint)
    assert restored.update() == expected_metrics
    tensors_equal(restored.model.state_dict(), expected)
    tensors_equal(restored.optimizer.state_dict(), optimizer)
    assert torch.equal(restored.env.state, state) and restored.counters == counters
    restored.model.koi_temperature = 0.05
    with pytest.raises(ValueError, match="changed"):
        restored.collect()


def test_saved_training_temperature_does_not_change_greedy_agent(tmp_path):
    trainer = KoiTemperatureTrainer(
        KoiTemperatureConfig(device="cpu", koi_temperature=0.21989584747027907)
    )
    candidate_path, original_path = tmp_path / "candidate.pt", tmp_path / "original.pt"
    trainer.save(candidate_path)
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    payload["format"] = "published_ppo_v1"
    payload["config"].pop("koi_temperature")
    torch.save(payload, original_path)
    candidate, original = (
        KoiTemperatureAgent(candidate_path),
        NeuralAgent(original_path, device="cpu", deterministic=True),
    )
    for seed in (10771001, 10771002):
        env = ReferenceEnv(seed, "paper8")
        while env.state[PHASE] != TERMINAL:
            a, b = candidate.act(env), original.act(env)
            assert a == b
            env.step(a)
    assert type(candidate.model) is PublishedPolicyValue
    assert candidate.model.temperature == 0.05
    np.testing.assert_equal(
        candidate.training_config["koi_temperature"], 0.21989584747027907
    )
