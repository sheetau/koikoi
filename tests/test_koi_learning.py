from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.capture import CapturedMinibatch
from koikoi.estimators import ppo_value_input
from koikoi.model import PolicyValue
from koikoi.opponents import FrozenOpponent
from koikoi.train import Config, Trainer


def config(**overrides):
    fields = {
        "device": "cpu",
        "profile": "paper8",
        "envs": 8,
        "horizon": 128,
        "width": 32,
        "critic_width": 32,
        "epochs": 1,
        "minibatch": 256,
        "policy_scope": "koi",
        "koi_width": 32,
        "ppo_value": "observation",
        "seed": 60001,
    }
    fields.update(overrides)
    return Config(**fields)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("kind", ["mlp", "relational"])
def test_initialization_freezes_cards_but_learns_koi_and_values(tmp_path, kind):
    parent = Trainer(config(kind=kind, policy_scope="all", koi_width=0))
    parent.update()
    checkpoint = tmp_path / "parent.pt"
    parent.save(checkpoint)
    trainer = Trainer(config(kind=kind))
    rng = torch.get_rng_state().clone()
    simulator = trainer.env.state.clone()
    trainer.initialize_weights(checkpoint)
    same(rng, torch.get_rng_state())
    same(simulator, trainer.env.state)
    assert not any(trainer.counters.values())
    assert not trainer.optimizer.state
    assert trainer.provenance["initialized_from"]["counters"] == parent.counters
    probe = parent.collect()
    obs, mask = probe["obs"], probe["mask"]
    same(parent.actor(obs, mask), trainer.actor(obs, mask))
    weights = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    before = trainer.actor(obs, torch.ones_like(mask))[0].detach().clone()
    for _ in range(3):
        trainer.update()
    for key, value in trainer.model.state_dict().items():
        if not key.startswith(("koi.", "koi_residual.", "value.")):
            same(value, weights[key])
    after = trainer.actor(obs, torch.ones_like(mask))[0]
    same(before[:, :48], after[:, :48])
    assert not torch.equal(before[:, 48:], after[:, 48:])
    assert not torch.equal(weights["value.weight"], trainer.model.value.weight)
    assert torch.count_nonzero(trainer.model.koi_residual[-1].weight) > 0
    assert torch.count_nonzero(trainer.ppo_critic.network[-1].weight) > 0
    assert 0 < trainer.counters["policy_decisions"] < trainer.counters["decisions"]

    exported = tmp_path / "koi.pt"
    trainer.save(exported)
    agent = NeuralAgent(exported, deterministic=True)
    same(trainer.actor(obs, mask), agent.model(obs, mask))
    frozen = FrozenOpponent(agent.training_config, agent.model.state_dict(), "cpu")
    same(frozen.logits(obs, mask), agent.model(obs, mask)[0])
    compatible = Trainer(config(kind=kind))
    compatible.initialize_weights(exported)
    same(compatible.model.state_dict(), trainer.model.state_dict())
    incompatible = Trainer(config(kind=kind, koi_width=0))
    with pytest.raises(ValueError, match="koi_width"):
        incompatible.initialize_weights(exported)


def test_residual_initialization_preserves_rng_and_hidden_world_invariance():
    torch.manual_seed(11)
    base = PolicyValue(width=32)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(11)
    expanded = PolicyValue(width=32, koi_width=32)
    same(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        same(value, expanded.state_dict()[key])
    trainer = Trainer(config())
    with torch.no_grad():
        trainer.model.koi_residual[-1].weight.fill_(0.2)
    obs, mask = trainer.observe()
    output = trainer.actor(obs, mask)
    trainer.env.state[:, 72:96] = trainer.env.state[:, 72:96].flip(-1)
    changed_obs, changed_mask = trainer.observe()
    same(obs, changed_obs)
    same(mask, changed_mask)
    same(output, trainer.actor(changed_obs, changed_mask))


def synthetic_batch(trainer):
    obs, _ = trainer.observe()
    obs = obs[:3]
    mask = torch.zeros(3, 50, dtype=torch.bool)
    mask[0, :2] = True
    mask[1:, 48:] = True
    with torch.no_grad():
        logits, value, _ = trainer.actor(obs, mask)
    action = torch.tensor([0, 48, 49])
    return dict(
        obs=obs,
        mask=mask,
        action=action,
        old_logp=logits.log_softmax(-1).gather(1, action[:, None]).squeeze(-1),
        advantage=torch.tensor([1000000.0, 1.0, 3.0]),
        **{"return": value + 0.5},
        target=torch.zeros(3, 48),
        ppo_value_obs=ppo_value_input(obs, trainer.env.state[:3], "observation"),
    )


def test_policy_loss_entropy_and_normalization_use_only_koi_rows(monkeypatch):
    first, second = Trainer(config()), Trainer(config())
    data = synthetic_batch(first)
    with torch.no_grad():
        logits = first.actor(data["obs"], data["mask"])[0]
        logp = logits.log_softmax(-1)[1:]
        entropy = -(logp.exp() * logp).sum(-1).mean()
    altered = {k: v.clone() for k, v in data.items()}
    altered["advantage"][0] = -1000000
    altered["old_logp"][0] += 100
    result = first.minibatch_step(data)
    same(result, second.minibatch_step(altered))
    same(first.trainable.state_dict(), second.trainable.state_dict())
    assert result[0] == -2
    same(result[2], entropy)
    assert result[4] == 0

    trainer = Trainer(config(minibatch=3))
    data = synthetic_batch(trainer)
    seen = []
    monkeypatch.setattr(trainer, "collect", lambda: data.copy())

    def inspect(batch):
        seen.append(dict(zip(batch["action"].tolist(), batch["advantage"].tolist())))
        return torch.zeros(7)

    monkeypatch.setattr(trainer, "minibatch_step", inspect)
    trainer.update()
    assert seen == [{0: 0.0, 48: -1.0, 49: 1.0}]
    data["mask"][:] = False
    data["mask"][:, :2] = True
    trainer.update()
    assert all(value == 0 for value in seen[-1].values())


@pytest.mark.parametrize("koi_width", [0, 32])
@pytest.mark.parametrize(
    "device,precision",
    [
        ("cpu", "float32"),
        pytest.param("cuda", "float32", marks=pytest.mark.cuda),
        pytest.param("cuda", "bf16", marks=pytest.mark.cuda),
    ],
)
def test_exact_resume_counters_and_freeze_state(tmp_path, koi_width, device, precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(
        config(
            koi_width=koi_width,
            device=device,
            precision=precision,
            cuda_graph=device == "cuda",
            cuda_rollout=device == "cuda",
        )
    )
    batch = trainer.collect()
    assert trainer.counters["policy_decisions"] == int(
        batch["mask"][:, 48:].any(-1).sum()
    )
    trainer.update()
    path = tmp_path / "before.pt"
    trainer.save(path)
    expected = trainer.update()
    after = tmp_path / "after.pt"
    trainer.save(after)
    restored = Trainer.restore(path)
    assert restored.update() == expected
    same(trainer.trainable.state_dict(), restored.trainable.state_dict())
    same(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    same(trainer.env.state, restored.env.state)
    assert trainer.counters == restored.counters
    assert all(not p.requires_grad for p in restored.model.body.parameters())
    resumed = tmp_path / "resumed.pt"
    restored.save(resumed)
    a = torch.load(after, map_location="cpu", weights_only=False)
    b = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in ("torch_rng", "cuda_rng", "python_rng", "numpy_rng"):
        same(a[key], b[key])


@pytest.mark.cuda
@pytest.mark.parametrize("koi_width", [0, 32])
def test_capture_matches_eager_with_frozen_card_parameters(koi_width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    c = config(
        device="cuda",
        precision="bf16",
        koi_width=koi_width,
        horizon=32,
        cuda_graph=True,
    )
    source = Trainer(c)
    data = source.collect()
    data["return"].add_(0.5)
    eager, captured = Trainer(c), Trainer(c)
    graph = CapturedMinibatch(
        captured.minibatch_step, captured.trainable, captured.optimizer, data
    )
    for _ in range(2):
        same(eager.minibatch_step(data), graph(data))
        same(eager.trainable.state_dict(), captured.trainable.state_dict())
        same(eager.optimizer.state_dict(), captured.optimizer.state_dict())
    paths = []
    for cfg in (c, replace(c, cuda_rollout=True)):
        trainer = Trainer(cfg)
        metrics = [trainer.update() for _ in range(2)]
        paths.append(
            (
                metrics,
                trainer.trainable.state_dict(),
                trainer.optimizer.state_dict(),
                trainer.env.state,
                trainer.counters,
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
            )
        )
    same(*paths)


@pytest.mark.parametrize(
    "fields",
    [
        {"policy_scope": "unknown"},
        {"koi_width": -1},
        {"ppo_value": "shared"},
        {"ppo_value": "privileged"},
        {"algorithm": "vrpo"},
        {"reward": "round"},
        {"opponent_fraction": 0.5},
        {"belief_weight": 0.1},
    ],
)
def test_reject_incompatible_koi_learning(fields):
    with pytest.raises(ValueError):
        config(**fields)
