"""Player-relative Expected-SARSA Q-boosting (Fan & Farina, 2026, eqs. 3.2–3.3)."""

import torch
from torch import nn

from .layout import PLAYER, STOCK


def native_gae(
    backend, rewards, values, dones, players, next_value, next_player, lam=0.95
):
    """One independent reverse recurrence per environment, on CPU or CUDA."""
    h, n = rewards.shape
    if min(h, n) < 1 or rewards.device != getattr(
        backend, "device", torch.device("cpu")
    ):
        raise ValueError("GAE needs nonempty inputs on the backend device")
    specs = (
        (rewards, torch.float32),
        (values, torch.float32),
        (dones, torch.bool),
        (players, torch.int32),
        (next_value, torch.float32),
        (next_player, torch.int32),
    )
    inputs = [tensor.to(dtype=dtype).contiguous() for tensor, dtype in specs]
    if any(tensor.device != rewards.device for tensor in inputs):
        raise ValueError("GAE inputs must share a device")
    if any(tensor.shape != (h, n) for tensor in inputs[:4]) or any(
        tensor.shape != (n,) for tensor in inputs[4:]
    ):
        raise ValueError("GAE input shapes must agree")
    advantages, returns = torch.empty_like(inputs[0]), torch.empty_like(inputs[0])
    backend.call("kk_gae", n, *inputs, advantages, returns, h, n, float(lam))
    return advantages, returns


def critic_input(obs, state):
    """Training-only private extension. Never passed to PolicyValue.forward."""
    opponent = (state[:, :48] == (1 - state[:, PLAYER, None])).float()
    indices = torch.arange(24, device=state.device)[None]
    ranks = ((indices + 24) <= state[:, STOCK, None]).float() * (indices + 1) / 24
    stock_order = torch.zeros_like(opponent).scatter(1, state[:, 72:96].long(), ranks)
    return torch.cat([obs, opponent, stock_order], dim=-1)


class CentralQ(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(656, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 50),
        )

    def forward(self, features):
        return self.network(features)


def q_boost(
    rewards,
    q_selected,
    expected_values,
    dones,
    players,
    next_value,
    next_player,
    lam=0.95,
):
    trace = torch.zeros_like(next_value)
    targets = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        sign = torch.where(players[t] == next_player, 1.0, -1.0)
        live = (~dones[t]).float()
        delta = rewards[t] + live * sign * next_value - q_selected[t]
        trace = delta + live * sign * lam * trace
        targets[t] = q_selected[t] + trace
        next_value, next_player = expected_values[t], players[t]
    return targets - expected_values, targets
