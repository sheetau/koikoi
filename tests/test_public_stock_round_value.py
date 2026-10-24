import numpy as np
import pytest
import torch

from koikoi.layout import DEALER, KOIS, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.public_stock_native import PublicStockBatch
from koikoi.public_stock_residual import load_residual
from koikoi.public_stock_round_value import boundary_value, round_rollout_actions
from koikoi.public_stock_search import view


def test_boundary_orientation_and_terminal_override():
    states = torch.zeros((4, 208), dtype=torch.int32)
    states[:, ROUND] = 3
    states[:, POINTS : POINTS + 2] = torch.tensor(
        [[40, 20], [20, 40], [65, -5], [-5, 65]]
    )
    states[:, DEALER] = torch.tensor([0, 1, 0, 1])
    states[2:, PHASE] = TERMINAL
    table = torch.zeros((8, 2, 61))
    table[3, 0, 40] = 0.6
    table[3, 1, 20] = -0.4
    np.testing.assert_allclose(
        boundary_value(states, torch.zeros(4), table), [0.6, -0.4, 1, -1]
    )
    np.testing.assert_allclose(
        boundary_value(states, torch.ones(4), table), [-0.6, 0.4, -1, 1]
    )


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    model, _ = load_residual(
        "runs/public_stock_residual_v2/preview1_choice1_stakes1_parent0/policy_00128.pt"
    )
    return model


def test_published_features_require_absolute_dealer_in_cache_key(model):
    batch = PublicStockBatch([61, 62, 63], "preview1_choice1_stakes1", "cuda")
    saw_difference = False
    for _ in range(36):
        swapped = batch.state.clone()
        locations = swapped[:, :48].clone()
        for a, b in [(0, 1), (1, 0), (3, 4), (4, 3)]:
            swapped[:, :48][locations == a] = b
        swapped[:, 48:56], swapped[:, 56:64] = (
            batch.state[:, 56:64],
            batch.state[:, 48:56],
        )
        swapped[:, PLAYER], swapped[:, DEALER] = (
            1 - batch.state[:, PLAYER],
            1 - batch.state[:, DEALER],
        )
        swapped[:, POINTS : POINTS + 2] = batch.state[:, POINTS : POINTS + 2].flip(1)
        swapped[:, KOIS : KOIS + 2] = batch.state[:, KOIS : KOIS + 2].flip(1)
        obs, legal = model.encode(batch)
        other_obs, other_legal = model.encode(view(swapped, batch))
        torch.testing.assert_close(obs[:, :660], other_obs[:, :660], rtol=0, atol=0)
        saw_difference |= not torch.equal(obs[:, 660:], other_obs[:, 660:])
        assert torch.equal(legal, other_legal)
        with torch.no_grad():
            actions = model(obs, legal)[0].argmax(1)
        batch.step(actions)
    assert saw_difference


def test_round_search_finishes_without_neural_value_substitution(model):
    batch = PublicStockBatch([31, 32], "preview1_choice1_stakes1", "cuda")
    model.round_table = torch.zeros((8, 2, 61), device="cuda")
    before = batch.state.clone()
    actions, work = round_rollout_actions(
        batch, model, 2, 80, torch.Generator(device="cuda").manual_seed(7)
    )
    assert torch.equal(before, batch.state)
    mask = model.encode(batch)[1]
    assert mask[torch.arange(2, device="cuda"), actions].all()
    assert work["transitions"] >= work["rollouts"]
    with pytest.raises(RuntimeError, match="Incomplete round"):
        round_rollout_actions(
            batch, model, 2, 1, torch.Generator(device="cuda").manual_seed(7)
        )
