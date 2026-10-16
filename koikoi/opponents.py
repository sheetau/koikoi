"""Frozen observation-only opponents and a training-only opponent-role value."""

import torch
from torch import nn

from .layout import EPISODE, OBS_SIZE, PLAYER
from .model import PolicyValue


def learner_turn(state, rows, frozen_envs):
    """Alternate learner seats at match resets, never at round boundaries."""
    seat = (state[:, EPISODE].long() + rows) % 2
    return (rows >= frozen_envs) | (state[:, PLAYER] == seat)


class OpponentValue(nn.Module):
    """Value from the frozen actor's perspective under the joint policy.

    This trainable critic is distinct from the frozen checkpoint's old value
    function, which estimates a different matchup. It sees acting-player
    observations only and is never exported as the learner's acting policy.
    """

    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBS_SIZE, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, obs):
        return self.network(obs).squeeze(-1)


class FrozenOpponent:
    def __init__(self, config, weights, device):
        self.config = config.copy()
        device = torch.device(device)
        devices = [device.index or 0] if device.type == "cuda" else []
        # Loading an opponent must not consume the learner's random stream.
        with torch.random.fork_rng(devices=devices):
            self.model = PolicyValue(
                config["kind"], config["width"], config["layers"], config.get("koi_width", 0)
            ).to(device)
            self.model.load_state_dict(weights)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def logits(self, obs, mask):
        with torch.autocast(
            device_type=obs.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.get("precision", "float32") == "bf16",
            cache_enabled=False,
        ):
            logits, _, _ = self.model(obs, mask)
        return logits.float()
