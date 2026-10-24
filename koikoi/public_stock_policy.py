"""Observation-only actor and critic for the public-stock development study."""

import torch
from torch import nn


class PublicStockPolicy(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.body = nn.Sequential(
            nn.Linear(660, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.actor = nn.Linear(width, 50)
        self.critic = nn.Linear(width, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observation, legal):
        features = self.body(observation)
        return (
            self.actor(features).masked_fill(~legal, -1e9),
            self.critic(features).squeeze(-1),
        )


def load_policy(path, device="cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PublicStockPolicy(payload["config"]["width"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload
