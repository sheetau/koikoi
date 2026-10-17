"""Independent transition/return checks and information boundaries for arbitration."""

import copy

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.arbitration import ArbitrationAgent, ArbitrationConfig, ArbitrationTrainer
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.fixture(scope="module")
def experts():
    torch.set_num_threads(2)
    return [PublishedActor.from_published(v).checkpoint() for v in ("wp", "point")]


def config(device="cpu", **extra):
    return ArbitrationConfig(
        **(
            {
                "device": device,
                "envs": 2,
                "horizon": 160,
                "minibatch": 128,
                "epochs": 1,
                "width": 32,
                "critic_width": 32,
                "critic_warmup_updates": 1,
                "seed": 19000001,
            }
            | extra
        )
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reference_transitions_signed_returns_and_hidden_invariance(
    experts, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = ArbitrationTrainer(config(device), experts)
    with torch.no_grad():
        trainer.model.value[-1].weight.fill_(0.1)
        trainer.model.value[-1].bias.fill_(0.2)
    refs = []
    for s in trainer.env.state.cpu().numpy():
        r = ReferenceEnv.__new__(ReferenceEnv)
        r.state = s.copy()
        refs.append(r)
    original_step, results = trainer.env.step, []

    def checked(actions):
        expected = np.stack([r.step(a) for r, a in zip(refs, actions.cpu().tolist())])
        actual = original_step(actions)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        results.append(expected)
        for i, r in enumerate(refs):
            if r.state[PHASE] == 4:
                refs[i] = ReferenceEnv(
                    int(r.state[SEED]), "paper8", int(r.state[EPISODE]) + 1
                )
        return actual

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    h, n = trainer.config.horizon, trainer.config.envs
    result = np.stack(results)
    players = batch["player"].cpu().numpy().reshape(h, n)
    values = batch["value"].cpu().numpy().reshape(h, n) * (1 - 2 * players)
    obs, mask = trainer.observe()
    with torch.no_grad():
        tail = trainer.model(obs, mask)[1].cpu().numpy() * (
            1 - 2 * trainer.env.state[:, PLAYER].cpu().numpy()
        )
    future, carry = tail.astype(float), np.zeros(n)
    expected = np.empty((h, n))
    for t in reversed(range(h)):
        live = 1 - result[t, :, 2]
        carry = result[t, :, 0] + live * future - values[t] + live * 0.95 * carry
        expected[t] = (carry + values[t]) * (1 - 2 * players[t])
        future = values[t]
    np.testing.assert_allclose(
        batch["return"].cpu(), expected.flatten(), rtol=0, atol=2e-6
    )
    assert trainer.counters["matches"] == result[:, :, 2].sum() > 0
    assert trainer.counters["rounds"] == result[:, :, 1].sum()
    rng = np.random.default_rng(500)
    hidden = np.stack([sample_hidden_batch(r, rng, 1)[0] for r in refs])
    trainer.env.state.copy_(torch.tensor(hidden, device=device))
    altered, altered_mask = trainer.observe()
    torch.testing.assert_close(obs, altered, rtol=0, atol=0)
    assert torch.equal(mask, altered_mask)
    with torch.no_grad():
        torch.testing.assert_close(
            trainer.model(obs, mask)[0],
            trainer.model(altered, altered_mask)[0],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_frozen_experts_warmup_and_exact_resume(experts, device, tmp_path, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = ArbitrationTrainer(config(device), experts)
    initial = copy.deepcopy(trainer.model.actor.state_dict())
    assert trainer.update()["critic_only"]
    assert all(
        torch.equal(v, trainer.model.actor.state_dict()[k]) for k, v in initial.items()
    )
    assert not trainer.update()["critic_only"]
    assert any(
        not torch.equal(v, trainer.model.actor.state_dict()[k])
        for k, v in initial.items()
    )
    assert [
        s["current"]["parameters_sha256"] for s in trainer.bank.snapshots()
    ] == trainer.expert_digests
    path = tmp_path / "arb.pt"
    trainer.save(path)
    expected = trainer.update()
    state, model, optimizer = (
        trainer.env.state.clone(),
        copy.deepcopy(trainer.model.state_dict()),
        copy.deepcopy(trainer.optimizer.state_dict()),
    )
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None

    def no_original_weights(*args, **kwargs):
        raise AssertionError("Restore must use embedded experts")

    monkeypatch.setattr(PublishedActor, "from_published", no_original_weights)
    restored = ArbitrationTrainer.restore(path)
    assert restored.update() == expected
    assert restored.counters == trainer.counters
    assert torch.equal(restored.env.state, state) and torch.equal(
        torch.get_rng_state(), rng
    )
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    for key, value in model.items():
        assert torch.equal(value, restored.model.state_dict()[key])
    actual = restored.optimizer.state_dict()
    assert optimizer["param_groups"] == actual["param_groups"]
    for key, record in optimizer["state"].items():
        for name, value in record.items():
            assert torch.equal(value, actual["state"][key][name])


@pytest.mark.parametrize("context", ["full", "phase", "no_match"])
def test_initial_greedy_matches_base_and_agreement_has_no_policy_gradient(
    experts, context, tmp_path
):
    trainer = ArbitrationTrainer(config(context=context), experts)
    path = tmp_path / "initial.pt"
    trainer.save(path)
    agent, base = ArbitrationAgent(path), EndgameAgent(PublishedAgent("wp"))
    ref = ReferenceEnv(19000401)
    while ref.state[PHASE] != 4:
        action = agent.act(ref)
        assert action == base.act(ref)
        ref.step(action)
    obs, mask = trainer.observe()
    obs[:, 561] = obs[:, 560]
    mask.zero_().scatter_(1, obs[:, 560:561].long(), True)
    logprobs = trainer.model(obs, mask)[0].log_softmax(-1)
    loss = -logprobs.gather(1, obs[:, 560:561].long()).mean()
    loss.backward()
    assert float(loss.detach()) == 0
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in trainer.model.actor.parameters()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"base_probability": 0.5},
        {"base_probability": float("nan")},
        {"context": "unknown"},
        {"profile": "research12"},
        {"lr": 0},
        {"endgame": False},
        {"critic_warmup_updates": -1},
    ],
)
def test_bad_config_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)
