import numpy as np
import pytest
import torch

from koikoi.layout import PHASE
from koikoi.public_stock_native import PublicStockBackend, PublicStockBatch
from koikoi.public_stock_residual import PublicStockResidual
from koikoi.public_stock_residual_search import rollout_actions
from koikoi.public_stock_search import sample_batch, view
from scripts.run_arbitration_study import PARENTS


@pytest.fixture(scope="module")
def setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    parent = torch.load(PARENTS[0][0], map_location="cpu", weights_only=False)
    return PublicStockResidual(parent), PublicStockBackend("cuda")


def test_initial_residual_preserves_parent_on_control(setup):
    model, backend = setup
    batch = PublicStockBatch(
        [71, 72, 73, 74], "preview0_choice0_stakes0", "cuda", backend
    )
    for _ in range(24):
        obs, legal = model.encode(batch)
        with torch.no_grad():
            scores, value = model(obs, legal)
        expected = obs[:, 660:].masked_fill(~legal, -1e9)
        torch.testing.assert_close(scores, expected, rtol=0, atol=0)
        # The original critic sees the same 560 inputs at initialization.
        old = model.parent["value"]
        x = obs[:, :560].cpu()
        x = torch.tanh(torch.nn.functional.linear(x, old["0.weight"], old["0.bias"]))
        x = torch.tanh(torch.nn.functional.linear(x, old["2.weight"], old["2.bias"]))
        old_value = torch.nn.functional.linear(
            x, old["4.weight"], old["4.bias"]
        ).squeeze(1)
        torch.testing.assert_close(value.cpu(), old_value, rtol=1e-4, atol=1e-5)
        batch.step(scores.argmax(1))


@pytest.mark.parametrize(
    "profile", ["preview0_choice1_stakes1", "preview1_choice1_stakes1"]
)
def test_residual_encoding_and_search_do_not_see_hidden_state(setup, profile):
    model, backend = setup
    batch = PublicStockBatch([25, 26], profile, "cuda", backend)
    generator = torch.Generator(device="cuda").manual_seed(78)
    for _ in range(12):
        obs, mask = model.encode(batch)
        altered = view(sample_batch(batch, 1, generator), batch)
        altered_obs, altered_mask = model.encode(altered)
        torch.testing.assert_close(obs, altered_obs, rtol=0, atol=0)
        assert torch.equal(mask, altered_mask)
        if (batch.state[:, PHASE] == 5).all():
            break
        batch.step(mask.float().argmax(1))
    a, work = rollout_actions(
        batch, model, 2, 3, torch.Generator(device="cuda").manual_seed(88)
    )
    b, other_work = rollout_actions(
        altered, model, 2, 3, torch.Generator(device="cuda").manual_seed(88)
    )
    np.testing.assert_array_equal(a.cpu(), b.cpu())
    assert work == other_work


def test_correction_can_train_without_changing_teacher(setup):
    model, backend = setup
    batch = PublicStockBatch([91, 92], "preview1_choice1_stakes1", "cuda", backend)
    obs, mask = model.encode(batch)
    logits, value = model(obs, mask)
    loss = logits.log_softmax(1)[mask].mean() + value.square().mean()
    loss.backward()
    assert model.actor.weight.grad is not None
    assert all(
        p.grad is None and not p.requires_grad for p in model.teacher.parameters()
    )
    model.zero_grad(set_to_none=True)


def test_encoder_chunks_large_rollout_batches(setup):
    model, backend = setup
    batch = PublicStockBatch(range(530), "preview1_choice1_stakes1", "cuda", backend)
    obs, legal = model.encode(batch)
    assert obs.shape == (530, 710)
    assert torch.isfinite(obs).all()
    assert legal.any(1).all()
    original, original_mask = batch.observe()
    torch.testing.assert_close(obs[:, :660], original, rtol=0, atol=0)
    assert torch.equal(legal, original_mask)
