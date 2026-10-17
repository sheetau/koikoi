"""Batched published policy contracts; CPU singleton stays the exact oracle."""

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent, reconstruct
from koikoi.upstream_batch import BatchPublishedAgent

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(62101)
    by_phase = {phase: [] for phase in range(4)}
    for seed in range(6210000, 6210010):
        ref = ReferenceEnv(seed)
        while ref.state[PHASE] != TERMINAL:
            phase = int(ref.state[PHASE])
            if len(by_phase[phase]) < 8:
                by_phase[phase].append(ref.state.copy())
            action = (
                49 if phase == KOI else int(rng.choice(np.flatnonzero(ref.legal())))
            )
            ref.step(action)
    assert all(len(v) == 8 for v in by_phase.values())
    return [s for phase in range(4) for s in by_phase[phase]]


@pytest.mark.parametrize("variant", ["wp", "point"])
def test_native_singleton_cpu_logits_close_and_actions_match_original(samples, variant):
    torch.set_num_threads(2)
    original = PublishedAgent(variant)
    native = BatchPublishedAgent(variant, "cpu", endgame=True)
    env = BatchEnv(1)
    for state in samples:
        env.state.copy_(torch.from_numpy(state[None]))
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        g = reconstruct(ref)
        with torch.no_grad():
            expected = original.models[int(state[PHASE])](g.feature_tensor[None])[0]
        legal = torch.from_numpy(g.round_state.action_mask).bool()
        expected = expected.masked_fill(~legal, -torch.inf)
        actual = native.logits(env)[0]
        actual = actual[48:] if state[PHASE] == KOI else actual[:48]
        # Exact feature parity is tested separately. Separate execution paths
        # may round differently even with the same float32 inputs and weights.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert native.act(env).item() == EndgameAgent(original).act(ref)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_mixed_batch_hidden_invariance_rng_and_terminal(samples, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    rng = np.random.default_rng(62109)
    states = []
    for state in samples:
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        states.extend([state, sample_hidden_batch(ref, rng, 1)[0]])
    terminal = samples[0].copy()
    terminal[PHASE] = TERMINAL
    states.append(terminal)
    env = BatchEnv(len(states), device)
    env.state.copy_(torch.tensor(np.stack(states), device=device))
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    agent = BatchPublishedAgent("wp", device, endgame=True)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if gpu_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    before = env.state.clone()
    logits, actions = agent.logits(env), agent.act(env)
    with torch.autocast(
        device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16
    ):
        torch.testing.assert_close(agent.logits(env), logits, rtol=0, atol=0)
        assert torch.equal(agent.act(env), actions)
    torch.testing.assert_close(logits[:-1:2], logits[1:-1:2], rtol=0, atol=0)
    assert torch.equal(actions[:-1:2], actions[1:-1:2])
    assert actions[-1] == -1 and torch.isneginf(logits[-1]).all()
    masks = env.observe()[1]
    assert masks[torch.arange(env.count - 1), actions[:-1]].all()
    assert torch.equal(env.state, before)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if gpu_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    assert all(
        not p.requires_grad
        for model in agent.models.values()
        for p in model.parameters()
    )


def test_profile_and_variant_rejected():
    with pytest.raises(ValueError, match="wp or point"):
        BatchPublishedAgent("invalid", "cpu")
    agent = BatchPublishedAgent("wp", "cpu")
    with pytest.raises(ValueError, match="paper8"):
        agent.act(BatchEnv(1, profile="research12"))


def test_teacher_float32_contract_with_float64_default():
    env = BatchEnv(1)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        agent = BatchPublishedAgent("wp", "cpu")
        logits = agent.logits(env)
    finally:
        torch.set_default_dtype(previous)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits[env.observe()[1]]).all()
    assert all(
        p.dtype == torch.float32
        for model in agent.models.values()
        for p in model.parameters()
    )
