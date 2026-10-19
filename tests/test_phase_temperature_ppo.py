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
from scripts.phase_temperature_ppo import (
    PhaseTemperatureAgent,
    PhaseTemperatureConfig,
    PhaseTemperatureTrainer,
    PhaseTemperatureValue,
)

MATCHED = (
    0.16808073347585306,
    0.19483521603032863,
    0.21949649006919622,
    0.23576122751596976,
)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.arange(50, dtype=torch.float32) / 10)

    def forward(self, features, phases, legal):
        return self.score.expand(len(features), -1)


def test_phase_scaling_preserves_mask_value_and_parameter_objects():
    old = PublishedPolicyValue.__new__(PublishedPolicyValue)
    nn.Module.__init__(old)
    old.actor, old.value, old.temperature = ToyActor(), nn.Linear(560, 1), 0.05
    temperatures = (0.05, 0.1, 0.2, 0.4)
    new = PhaseTemperatureValue(old, temperatures)
    obs = torch.zeros(4, 15561)
    obs[:, 560] = torch.arange(4)
    legal = torch.ones(4, 50, dtype=torch.bool)
    legal[:, 10] = False
    before, after = old(obs, legal), new(obs, legal)
    assert old.actor is new.actor and old.value is new.value
    assert torch.equal(before[1], after[1])
    assert torch.equal(after[0][:, 10], torch.full((4,), -1e9))
    for phase, temperature in enumerate(temperatures):
        assert after[0][phase, 11] == old.actor.score[11] / temperature
    after[0][:, 11].sum().backward()
    assert new.actor.score.grad[11] == sum(1 / t for t in temperatures)
    assert all(p.grad is None for p in new.value.parameters())
    equal(PhaseTemperatureValue(old, (0.05,) * 4)(obs, legal), before)


@pytest.mark.parametrize(
    "temperatures",
    [(0.05,) * 3, (0, 1, 1, 1), (1, float("nan"), 1, 1), (True, 1, 1, 1)],
)
def test_invalid_temperatures(temperatures):
    with pytest.raises(ValueError):
        PhaseTemperatureConfig(phase_temperatures=temperatures)


def config(variant, temperatures):
    return PhaseTemperatureConfig(
        device="cuda",
        seed=102000101,
        envs=8,
        horizon=64,
        minibatch=64,
        epochs=1,
        critic_warmup_updates=1,
        initial_variant=variant,
        phase_temperatures=temperatures,
    )


@pytest.mark.parametrize("variant", ["wp", "point"])
def test_cuda_uniform_control_exact_updates(variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = config(variant, (0.05,) * 4)
    values = asdict(cfg)
    values.pop("phase_temperatures")
    original = PublishedTrainer(PublishedConfig(**values))
    expected_metrics = [original.update() for _ in range(3)]
    expected = copy.deepcopy(original.model.state_dict())
    optimizer = copy.deepcopy(original.optimizer.state_dict())
    state, counters = original.env.state.clone(), original.counters.copy()
    candidate = PhaseTemperatureTrainer(cfg)
    assert [candidate.update() for _ in range(3)] == expected_metrics
    equal(candidate.model.state_dict(), expected)
    equal(candidate.optimizer.state_dict(), optimizer)
    equal(candidate.env.state, state)
    assert candidate.counters == counters


def test_cuda_point_warmup_and_exact_next_update(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = PhaseTemperatureTrainer(config("point", MATCHED))
    initial = copy.deepcopy(trainer.model.actor.state_dict())
    assert trainer.model.actor.initialization["variant"] == "point"
    trainer.update()
    equal(initial, trainer.model.actor.state_dict())
    trainer.update()
    assert any(
        not torch.equal(v, trainer.model.actor.state_dict()[k])
        for k, v in initial.items()
    )
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save(checkpoint)
    metrics = trainer.update()
    trainer.save(tmp_path / "expected.pt")
    restored = PhaseTemperatureTrainer.restore(checkpoint)
    assert restored.update() == metrics
    restored.save(tmp_path / "actual.pt")
    expected = torch.load(tmp_path / "expected.pt", weights_only=False)
    actual = torch.load(tmp_path / "actual.pt", weights_only=False)
    for key in (
        "actor",
        "value",
        "optimizer",
        "env",
        "counters",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "numpy_rng",
    ):
        equal(actual[key], expected[key])
    restored.model.phase_temperatures = (0.05,) * 4
    with pytest.raises(ValueError, match="changed"):
        restored.collect()


def test_phase_temperatures_do_not_change_saved_greedy_inference(tmp_path):
    trainer = PhaseTemperatureTrainer(
        PhaseTemperatureConfig(
            device="cpu", initial_variant="point", phase_temperatures=list(MATCHED)
        )
    )
    path = tmp_path / "candidate.pt"
    trainer.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format"] = "published_ppo_v1"
    payload["config"].pop("phase_temperatures")
    torch.save(payload, tmp_path / "original.pt")
    candidate = PhaseTemperatureAgent(path)
    original = NeuralAgent(tmp_path / "original.pt", device="cpu", deterministic=True)
    for seed in (102000301, 102000302):
        env = ReferenceEnv(seed, "paper8")
        while env.state[PHASE] != TERMINAL:
            action = candidate.act(env)
            assert action == original.act(env)
            env.step(action)
    assert type(candidate.model) is PublishedPolicyValue
    assert candidate.model.temperature == 0.05
    assert tuple(candidate.training_config["phase_temperatures"]) == MATCHED
