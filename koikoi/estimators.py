"""Player-relative Expected-SARSA Q-boosting (Fan & Farina, 2026, eqs. 3.2–3.3)."""

import torch
from torch import nn

from .layout import PLAYER, STOCK


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
