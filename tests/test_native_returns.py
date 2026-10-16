from dataclasses import replace

import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.estimators import native_gae
from koikoi.train import Config, Trainer, signed_gae


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_fused_returns_are_bit_identical_to_torch(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(73)
    env = BatchEnv(37, device=device)
    shape = (129, 37)
    rewards, values = (torch.randn(shape, device=device) for _ in range(2))
    dones = torch.rand(shape, device=device) < 0.07
    players = torch.randint(2, shape, device=device, dtype=torch.int32)
    bootstrap = torch.randn(37, device=device)
    next_players = torch.randint(2, (37,), device=device, dtype=torch.int32)
    for lam in (0.0, 0.95, 1.0):
        args = rewards, values, dones, players, bootstrap, next_players, lam
        expected = signed_gae(*args)
        actual = native_gae(env.backend, *args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_native_and_torch_ppo_updates_are_identical(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        device=device,
        envs=32,
        horizon=16,
        width=32,
        epochs=1,
        minibatch=256,
        returns_backend="torch",
    )
    reference = Trainer(config)
    expected = reference.update()
    native = Trainer(replace(config, returns_backend="native"))
    assert native.update() == expected
    assert torch.equal(native.env.state, reference.env.state)
    for key, value in reference.model.state_dict().items():
        assert torch.equal(value, native.model.state_dict()[key])
