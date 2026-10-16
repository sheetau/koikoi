from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.capture import CapturedMinibatch
from koikoi.estimators import ppo_value_input
from koikoi.layout import PLAYER
from koikoi.train import Config, Trainer


def config(**kwargs):
    fields = {
        "device": "cpu",
        "profile": "paper8",
        "envs": 8,
        "horizon": 32,
        "width": 32,
        "critic_width": 32,
        "epochs": 1,
        "minibatch": 64,
        "ppo_value": "privileged",
    }
    fields.update(kwargs)
    return Config(**fields)


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def assert_trainers(a, b):
    assert_same(a.trainable.state_dict(), b.trainable.state_dict())
    assert_same(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert_same(a.env.state, b.env.state)
    assert a.counters == b.counters


def test_control_initialization_rng_and_hidden_input_isolation(tmp_path):
    shared = Trainer(config(ppo_value="shared"))
    rng = torch.get_rng_state()
    observation = Trainer(config(ppo_value="observation"))
    assert_same(rng, torch.get_rng_state())
    privileged = Trainer(config())
    assert_same(rng, torch.get_rng_state())
    assert_same(shared.model.state_dict(), privileged.model.state_dict())
    assert_same(observation.trainable.state_dict(), privileged.trainable.state_dict())

    obs, mask = privileged.observe()
    original = privileged.env.state.clone()
    # Permuting only the future stock order leaves legal actor observations intact.
    privileged.env.state[:, 72:96] = original[:, 72:96].flip(-1)
    changed_obs, changed_mask = privileged.observe()
    assert_same(obs, changed_obs)
    assert_same(mask, changed_mask)
    assert_same(
        privileged.actor(obs, mask), privileged.actor(changed_obs, changed_mask)
    )
    control = ppo_value_input(obs, original, "observation")
    assert_same(
        control, ppo_value_input(changed_obs, privileged.env.state, "observation")
    )
    assert torch.count_nonzero(control[:, 560:]) == 0
    private = ppo_value_input(obs, original, "privileged")
    changed_private = ppo_value_input(changed_obs, privileged.env.state, "privileged")
    assert not torch.equal(private[:, 608:], changed_private[:, 608:])
    expected_hand = (original[:, :48] == 1 - original[:, PLAYER, None]).float()
    assert_same(private[:, 560:608], expected_hand)

    # Auxiliary critic gradients have no path into the deployed actor parameters.
    privileged.ppo_critic_value(private).sum().backward()
    assert all(p.grad is None for p in privileged.model.parameters())
    assert any(p.grad is not None for p in privileged.ppo_critic.parameters())
    path = tmp_path / "export.pt"
    privileged.save(path)
    agent = NeuralAgent(path, deterministic=True)
    assert_same(agent.model.state_dict(), privileged.model.state_dict())
    assert_same(agent.model(obs, mask), privileged.model(obs, mask))


@pytest.mark.parametrize("mode", ["observation", "privileged"])
def test_complete_match_and_tail_use_separate_critic(mode):
    trainer = Trainer(config(ppo_value=mode, horizon=256, gae_lambda=1))
    with torch.no_grad():
        trainer.ppo_critic.network[-1].bias.fill_(0.375)
        trainer.model.value.weight.zero_()
        trainer.model.value.bias.fill_(-0.875)
    batch = trainer.collect()
    assert torch.all(batch["value"] == 0.375)
    assert not any(v.requires_grad for v in batch.values())
    h, n = trainer.config.horizon, trainer.config.envs
    players = batch["player"].reshape(h, n)
    done = batch["done"].reshape(h, n)
    rewards = batch["reward"].reshape(h, n)
    returns = batch["return"].reshape(h, n)
    completed = 0
    for column in range(n):
        start = 0
        for end in done[:, column].nonzero().flatten().tolist():
            p0_result = rewards[end, column] * (1 - 2 * players[end, column])
            expected = p0_result * (1 - 2 * players[start : end + 1, column])
            torch.testing.assert_close(returns[start : end + 1, column], expected)
            start = end + 1
            completed += 1
        next_player = trainer.env.state[column, PLAYER]
        expected = 0.375 * (1 - 2 * next_player) * (1 - 2 * players[start:, column])
        torch.testing.assert_close(returns[start:, column], expected)
    assert completed > 5


@pytest.mark.parametrize("mode", ["observation", "privileged"])
@pytest.mark.parametrize(
    "device,precision",
    [
        ("cpu", "float32"),
        pytest.param("cuda", "float32", marks=pytest.mark.cuda),
        pytest.param("cuda", "bf16", marks=pytest.mark.cuda),
    ],
)
def test_exact_resume_and_both_value_heads_learn(tmp_path, mode, device, precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(
        config(
            ppo_value=mode,
            device=device,
            precision=precision,
            kind="relational",
            cuda_graph=device == "cuda",
            cuda_rollout=device == "cuda",
            horizon=256,
        )
    )
    original_value = trainer.model.value.weight.detach().clone()
    trainer.update()
    assert not torch.equal(original_value, trainer.model.value.weight)
    assert torch.count_nonzero(trainer.ppo_critic.network[-1].weight) > 0
    path = tmp_path / "state.pt"
    trainer.save(path)
    expected = trainer.update()
    after = tmp_path / "after.pt"
    trainer.save(after)
    restored = Trainer.restore(path)
    assert restored.update() == expected
    assert_trainers(trainer, restored)
    resumed = tmp_path / "resumed.pt"
    restored.save(resumed)
    a = torch.load(after, weights_only=False, map_location="cpu")
    b = torch.load(resumed, weights_only=False, map_location="cpu")
    for key in ("torch_rng", "cuda_rng", "python_rng", "numpy_rng"):
        assert_same(a[key], b[key])


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["observation", "privileged"])
def test_minibatch_and_rollout_capture_match_eager(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    c = config(
        device="cuda",
        ppo_value=mode,
        precision="bf16",
        kind="relational",
        cuda_graph=True,
    )
    source = Trainer(c)
    data = source.collect()
    # Ensure critic updates occur even without a completed match in this short batch.
    data["return"].add_(0.5)
    eager, captured = Trainer(c), Trainer(c)
    graph = CapturedMinibatch(
        captured.minibatch_step, captured.trainable, captured.optimizer, data
    )
    assert_same(eager.trainable.state_dict(), captured.trainable.state_dict())
    assert all(int(s["step"]) == 0 for s in captured.optimizer.state.values())
    for _ in range(2):
        assert_same(eager.minibatch_step(data), graph(data))
        assert_trainers(eager, captured)

    eager = Trainer(c)
    captured = Trainer(replace(c, cuda_rollout=True))
    # Nonzero changing critic weights catch stale BF16 inference caches.
    for _ in range(2):
        for trainer in (eager, captured):
            with torch.no_grad():
                trainer.ppo_critic.network[-1].bias.add_(0.125)
        rng = torch.cuda.get_rng_state()
        expected = eager.update()
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        assert captured.update() == expected
        assert_same(after, torch.cuda.get_rng_state())
        assert_trainers(eager, captured)


def test_invalid_configurations():
    for fields in (
        {"ppo_value": "unknown"},
        {"algorithm": "vrpo"},
        {"opponent_fraction": 0.5},
        {"critic_width": 0},
    ):
        with pytest.raises(ValueError):
            config(**fields)
