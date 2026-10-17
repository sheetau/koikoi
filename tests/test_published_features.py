"""Pinned upstream input parity, including hidden-world and device invariance."""

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.upstream import UPSTREAM_PATH, reconstruct


@pytest.fixture(scope="module")
def states():
    rng = np.random.default_rng(62003)
    snapshots = []
    for seed in range(6200000, 6200012):
        env = ReferenceEnv(seed)
        while env.state[PHASE] != TERMINAL:
            snapshots.append(env.state.copy())
            # Exercise long histories and repeated koi declarations.
            action = (
                49
                if env.state[PHASE] == KOI
                else int(rng.choice(np.flatnonzero(env.legal())))
            )
            env.step(action)
        snapshots.append(env.state.copy())
    result = np.stack(snapshots)
    assert set(result[:, PHASE]) == {0, 1, 2, 3, 4}
    return result


@pytest.fixture(scope="module")
def native(states):
    env = BatchEnv(len(states), "cpu")
    env.state.copy_(torch.from_numpy(states))
    before = env.state.clone()
    features, masks = env.published_observe()
    assert torch.equal(env.state, before)
    assert torch.isfinite(features).all()
    return features, masks


@pytest.mark.upstream
@pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched")
def test_every_decision_matches_pinned_upstream(states, native):
    features, masks = native
    for i, state in enumerate(states):
        if state[PHASE] == TERMINAL:
            assert not features[i].any() and not masks[i].any()
            continue
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        upstream = reconstruct(ref)
        width = 50 if state[PHASE] == KOI else 48
        torch.testing.assert_close(
            features[i, :, :width],
            upstream.feature_tensor,
            rtol=0,
            atol=0,
            msg=lambda msg, i=i, phase=int(state[PHASE]): (
                f"state {i}, phase {phase}: {msg}"
            ),
        )
        np.testing.assert_array_equal(masks[i], ref.legal())
        original_mask = masks[i, 48:] if state[PHASE] == KOI else masks[i, :48]
        np.testing.assert_array_equal(original_mask, upstream.round_state.action_mask)
        if width == 48:
            assert not features[i, :, 48:].any()


@pytest.mark.cuda
def test_cuda_features_and_masks_exactly_match_cpu(states, native):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = BatchEnv(len(states), "cuda")
    env.state.copy_(torch.from_numpy(states).cuda())
    before = env.state.clone()
    features, masks = env.published_observe()
    torch.testing.assert_close(features.cpu(), native[0], rtol=0, atol=0)
    assert torch.equal(masks.cpu(), native[1])
    assert torch.equal(env.state, before)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_features_ignore_actual_hidden_assignments(states, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(62009)
    cases = []
    for phase in range(4):
        rows = states[states[:, PHASE] == phase]
        for state in rows[:: max(1, len(rows) // 16)][:16]:
            ref = ReferenceEnv.__new__(ReferenceEnv)
            ref.state = state.copy()
            worlds = sample_hidden_batch(ref, rng, 2)
            cases.extend([state, *worlds])
    env = BatchEnv(len(cases), device)
    env.state.copy_(torch.tensor(np.stack(cases), device=device))
    features, masks = env.published_observe()
    assert torch.equal(features[::3], features[1::3])
    assert torch.equal(features[::3], features[2::3])
    assert torch.equal(masks[::3], masks[1::3])
    assert torch.equal(masks[::3], masks[2::3])


def test_research_profile_rejected():
    with pytest.raises(ValueError, match="paper8"):
        BatchEnv(1, profile="research12").published_observe()


def test_published_buffer_is_float32_with_changed_global_default():
    env = BatchEnv(1)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        features, masks = env.published_observe()
    finally:
        torch.set_default_dtype(previous)
    assert features.dtype == torch.float32 and masks.dtype == torch.bool
    assert torch.isfinite(features).all()
