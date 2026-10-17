"""Pretrained actor compatibility, gradients, dropout and embedded ancestry."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import UPSTREAM_PATH
from koikoi.upstream_batch import BatchPublishedAgent

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(66301)
    groups = {i: [] for i in range(4)}
    for seed in range(6630001, 6630011):
        ref = ReferenceEnv(seed)
        while ref.state[PHASE] != 4:
            phase = int(ref.state[PHASE])
            if len(groups[phase]) < 8:
                groups[phase].append(ref.state.copy())
            ref.step(49 if phase == 3 else int(rng.choice(np.flatnonzero(ref.legal()))))
    assert all(len(group) == 8 for group in groups.values())
    return np.stack([state for group in groups.values() for state in group])


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("variant", ["wp", "point"])
def test_original_scores_gradients_dropout_and_rng(states, device, variant):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    env = BatchEnv(len(states), device)
    env.state.copy_(torch.tensor(states, device=device))
    original = BatchPublishedAgent(variant, device)
    before = original.checkpoint()["parameters_sha256"]
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    actor = PublishedActor.from_published(variant, device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    actor.train()
    assert not any(module.training for module in actor.modules())
    assert all(parameter.requires_grad for parameter in actor.parameters())
    features, mask = env.published_observe()
    expected = original.logits(env)
    actual = actor(features, env.state[:, PHASE], mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    assert actor.checkpoint()["current"]["parameters_sha256"] == before
    repeated = actor(features, env.state[:, PHASE], mask)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    F.cross_entropy(actual, expected.argmax(-1)).backward()
    for model in actor.models.values():
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0
    assert actor.checkpoint()["current"]["parameters_sha256"] == before
    assert original.checkpoint()["parameters_sha256"] == before
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())


def test_checkpoint_after_test_update_keeps_original_ancestry(states, tmp_path):
    actor = PublishedActor.from_published("wp")
    initial = copy.deepcopy(actor.initialization)
    env = BatchEnv(len(states))
    env.state.copy_(torch.from_numpy(states))
    features, mask = env.published_observe()
    scores = actor(features, env.state[:, PHASE], mask)
    F.cross_entropy(scores, scores.detach().argmax(-1)).backward()
    torch.optim.SGD(actor.parameters(), lr=1e-4).step()
    payload = actor.checkpoint()
    assert payload["initialization"] == initial
    assert payload["current"]["parameters_sha256"] != initial["parameters_sha256"]
    path = tmp_path / "actor.pt"
    torch.save(payload, path)
    restored = PublishedActor(torch.load(path, weights_only=False))
    assert restored.initialization == initial
    for name, value in actor.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    torch.testing.assert_close(
        restored(features, env.state[:, PHASE], mask),
        actor(features, env.state[:, PHASE], mask),
        rtol=0,
        atol=0,
    )
    bad = copy.deepcopy(payload)
    bad["current"]["weights"][0][next(iter(bad["current"]["weights"][0]))].add_(0.1)
    with pytest.raises(ValueError, match="digest mismatch"):
        PublishedActor(bad)
    bad = copy.deepcopy(payload)
    bad["initialization"]["variant"] = "point"
    with pytest.raises(ValueError, match="identity mismatch"):
        PublishedActor(bad)


def test_hidden_world_invariance_and_terminal_rows(states):
    rng = np.random.default_rng(66311)
    pairs = []
    for state in states:
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        pairs.extend([state, sample_hidden_batch(ref, rng, 1)[0]])
    terminal = states[0].copy()
    terminal[PHASE] = 4
    pairs.append(terminal)
    env = BatchEnv(len(pairs))
    env.state.copy_(torch.tensor(np.stack(pairs)))
    actor = PublishedActor.from_published("wp")
    features, mask = env.published_observe()
    scores = actor(features, env.state[:, PHASE], mask)
    torch.testing.assert_close(scores[:-1:2], scores[1:-1:2], rtol=0, atol=0)
    assert torch.isneginf(scores[-1]).all()
    with pytest.raises(ValueError, match="input shapes"):
        actor(features[:, :, :48], env.state[:, PHASE], mask)
