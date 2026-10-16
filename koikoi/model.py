"""Observation-only actor, value estimator and optional hidden-card predictor."""

import torch
from torch import nn

from .layout import CARD_FEATURES, OBS_SIZE


class PolicyValue(nn.Module):
    def __init__(self, kind="mlp", width=128, layers=2):
        super().__init__()
        self.config = dict(kind=kind, width=width, layers=layers)
        self.kind = kind
        if kind == "mlp":
            self.body = nn.Sequential(nn.Linear(OBS_SIZE, width), nn.Tanh(),
                                      nn.Linear(width, width), nn.Tanh())
            self.discard = nn.Linear(width, 48)
            self.pick = nn.Linear(width, 48)
            self.belief = nn.Linear(width, 48)
        elif kind == "transformer":
            if width % 4:
                raise ValueError("Transformer width must be divisible by four")
            self.card = nn.Linear(CARD_FEATURES, width)
            self.identity = nn.Embedding(48, width)
            self.context = nn.Linear(32, width)
            layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.discard, self.pick, self.belief = (nn.Linear(width, 1) for _ in range(3))
        else:
            raise ValueError(f"Unknown network {kind}")
        self.koi = nn.Linear(width, 2)
        self.value = nn.Linear(width, 1)
        for head in (self.discard, self.pick, self.koi):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs, mask):
        if self.kind == "mlp":
            pooled = self.body(obs)
            discard, pick, belief = self.discard(pooled), self.pick(pooled), self.belief(pooled)
        else:
            cards = obs[:, :528].reshape(-1, 48, CARD_FEATURES)
            features = self.card(cards) + self.identity.weight.unsqueeze(0)
            context = self.context(obs[:, 528:]).unsqueeze(1)
            features = self.norm(self.body(torch.cat([features, context], dim=1)))
            pooled = features[:, -1]
            discard, pick = self.discard(features[:, :48]).squeeze(-1), self.pick(features[:, :48]).squeeze(-1)
            belief = self.belief(features[:, :48]).squeeze(-1)
        card_logits = torch.where(obs[:, 528:529] > 0, discard, pick)
        logits = torch.cat([card_logits, self.koi(pooled)], dim=-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, self.value(pooled).squeeze(-1), belief


def sample_action(logits):
    logprobs = logits.log_softmax(-1)
    action = torch.multinomial(logprobs.exp(), 1).squeeze(-1)
    return action, logprobs.gather(1, action[:, None]).squeeze(-1)
