"""Pretrained PPO reference returns, critic-only warmup and exact restart."""

import copy

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.published_actor import PublishedActor
from koikoi.published_train import PublishedConfig, PublishedTrainer, packed_observation
from koikoi.reference import ReferenceEnv
from koikoi.search import SearchAgent, sample_hidden_batch
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(device="cpu", **extra):
    return PublishedConfig(
        **(
            {
                "device": device,
                "envs": 2,
                "horizon": 128,
                "minibatch": 128,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 1,
                "seed": 6640001,
            }
            | extra
        )
    )


def verify_adam_counts(trainer):
    groups = [(trainer.model.value, "value")] + [
        (trainer.model.actor.models[str(phase)], name)
        for phase, name in ((0, "discard"), (1, "pick"), (3, "koi"))
    ]
    for model, name in groups:
        expected = trainer.counters["optimizer_steps_" + name]
        for parameter in model.parameters():
            state = trainer.optimizer.state.get(parameter)
            assert (0 if not state else int(state["step"])) == expected


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_reference_collection_rewards_signed_targets_and_public_inputs(
    device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = PublishedTrainer(config(device, horizon=160))
    with torch.no_grad():
        trainer.model.value[-1].weight.fill_(0.07)
        trainer.model.value[-1].bias.fill_(0.03)
    refs = []
    for state in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        refs.append(ref)
    step = trainer.env.step
    trace = []

    def checked(actions):
        before = [
            (ref.observe(), int(ref.state[PLAYER]), int(ref.state[PHASE]))
            for ref in refs
        ]
        results = [
            ref.step(action) for ref, action in zip(refs, actions.cpu().tolist())
        ]
        actual = step(actions)
        np.testing.assert_array_equal(actual.cpu(), results)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([ref.state for ref in refs])
        )
        trace.append((before, results))
        for index, ref in enumerate(refs):
            if ref.state[PHASE] == 4:
                refs[index] = ReferenceEnv(
                    int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                )
        return actual

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    n, h = trainer.config.envs, trainer.config.horizon
    observations = np.array(
        [[row[0] for row in before] for before, _ in trace]
    ).reshape(-1, 560)
    np.testing.assert_array_equal(batch["obs"][:, :560].cpu(), observations)
    result = np.array([row[1] for row in trace])
    players = batch["player"].cpu().numpy().reshape(h, n)
    values = batch["value"].cpu().numpy().reshape(h, n) * (1 - 2 * players)
    # Independent seat-zero backward recursion, including a nonzero horizon tail.
    next_obs, mask = trainer.observe()
    with torch.no_grad():
        tail = trainer.actor(next_obs, mask)[1].cpu().numpy() * (
            1 - 2 * trainer.env.state[:, PLAYER].cpu().numpy()
        )
    future, carry = tail.astype(float), np.zeros(n)
    expected = np.empty((h, n))
    for time in reversed(range(h)):
        live = 1 - result[time, :, 2]
        delta = result[time, :, 0] + live * future - values[time]
        carry = delta + live * trainer.config.gae_lambda * carry
        expected[time] = (carry + values[time]) * (1 - 2 * players[time])
        future = values[time]
    np.testing.assert_allclose(
        batch["return"].cpu(), expected.flatten(), rtol=0, atol=2e-6
    )
    assert trainer.counters["matches"] == result[:, :, 2].sum() > 0
    assert trainer.counters["rounds"] == result[:, :, 1].sum()
    assert trainer.counters["decisions"] == h * n
    np.testing.assert_array_equal(
        trainer.env.state.cpu(), np.stack([ref.state for ref in refs])
    )
    # Replace every hidden hand/stock/RNG while keeping observable information.
    rng = np.random.default_rng(66403)
    original, _ = packed_observation(trainer.env)
    hidden = np.stack([sample_hidden_batch(ref, rng, 1)[0] for ref in refs])
    trainer.env.state.copy_(torch.tensor(hidden, device=device))
    altered, _ = packed_observation(trainer.env)
    torch.testing.assert_close(original, altered, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("saved_updates", [1, 2])
def test_warmup_and_post_warmup_resume_exact(
    tmp_path, device, saved_updates, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = PublishedTrainer(config(device, horizon=160))
    initial_actor = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    initial_value = copy.deepcopy(trainer.model.value.state_dict())
    assert trainer.update()["critic_only"]
    assert (
        trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        == initial_actor
    )
    assert any(
        not torch.equal(v, trainer.model.value.state_dict()[k])
        for k, v in initial_value.items()
    )
    assert (
        trainer.counters["policy_updates"] == 0
        and trainer.counters["warmup_updates"] == 1
    )
    verify_adam_counts(trainer)
    if saved_updates == 2:
        assert not trainer.update()["critic_only"]
        assert (
            trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
            != initial_actor
        )
    path = tmp_path / "published.pt"
    trainer.save(path)
    expected = trainer.update()
    model, opt = (
        copy.deepcopy(trainer.model.state_dict()),
        copy.deepcopy(trainer.optimizer.state_dict()),
    )
    state, rng = trainer.env.state.clone(), torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None

    def reject_original_weights(*args, **kwargs):
        raise AssertionError("Resume must use embedded actor weights")

    monkeypatch.setattr(PublishedActor, "from_published", reject_original_weights)
    restored = PublishedTrainer.restore(path)
    assert restored.update() == expected
    assert restored.counters == trainer.counters
    assert torch.equal(state, restored.env.state) and torch.equal(
        rng, torch.get_rng_state()
    )
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for k, v in model.items():
        assert torch.equal(v, restored.model.state_dict()[k])
    actual = restored.optimizer.state_dict()
    assert opt["param_groups"] == actual["param_groups"]
    for k, items in opt["state"].items():
        for name, value in items.items():
            assert torch.equal(value, actual["state"][k][name])
    verify_adam_counts(restored)
    assert (
        restored.provenance["published_initialization"]
        == trainer.provenance["published_initialization"]
    )


def test_value_loss_has_no_actor_gradient():
    trainer = PublishedTrainer(config(horizon=8))
    obs, mask = trainer.observe()
    logits, values, _ = trainer.actor(obs, mask)
    (values - 1).square().mean().backward()
    assert all(p.grad is None for p in trainer.model.actor.parameters())
    assert trainer.model.value[-1].bias.grad.abs().sum() > 0
    trainer.optimizer.zero_grad(set_to_none=True)
    logits.log_softmax(-1).max(-1).values.mean().backward()
    assert all(p.grad is None for p in trainer.model.value.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainer.model.actor.parameters()
    )


def test_initial_checkpoint_greedy_policy_and_search_rejection(tmp_path):
    trainer = PublishedTrainer(config(horizon=8))
    path = tmp_path / "initial.pt"
    trainer.save(path)
    agent = NeuralAgent(path, deterministic=True)
    original = EndgameAgent(PublishedAgent("wp"))
    ref = ReferenceEnv(6641001)
    decisions = 0
    while ref.state[PHASE] != 4:
        action = agent.act(ref)
        assert action == original.act(ref)
        ref.step(action)
        decisions += 1
    assert decisions > 20 and agent.training_config["kind"] == "published"
    with pytest.raises(ValueError, match="does not yet support"):
        SearchAgent(agent)


@pytest.mark.parametrize("fault", ["temperature", "runtime", "ancestry", "weights"])
def test_snapshot_or_live_drift_rejected(tmp_path, fault):
    trainer = PublishedTrainer(config(horizon=8))
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    if fault == "temperature":
        trainer.model.temperature = 1
        with pytest.raises(ValueError, match="temperature changed"):
            trainer.collect()
        return
    payload = torch.load(path, weights_only=False)
    if fault == "runtime":
        payload["runtime"]["matmul_tf32"] = True
    elif fault == "ancestry":
        payload["provenance"]["published_initialization"]["parameters_sha256"] = (
            "0" * 64
        )
    else:
        weights = payload["actor"]["current"]["weights"][0]
        weights[next(iter(weights))].add_(0.1)
    torch.save(payload, path)
    with pytest.raises(ValueError):
        PublishedTrainer.restore(path)


@pytest.mark.parametrize(
    "extra",
    [
        {"score_temperature": 0},
        {"score_temperature": float("nan")},
        {"profile": "research12"},
        {"cuda_graph": True},
        {"complete_matches": True},
        {"opponent_fraction": 0.5},
        {"critic_warmup_updates": -1},
        {"belief_weight": 0.1},
    ],
)
def test_rejects_unsupported_or_invalid_config(extra):
    with pytest.raises(ValueError):
        config(**extra)
