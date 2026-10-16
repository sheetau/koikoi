"""Observation-only actor, value estimator and optional hidden-card predictor."""

import torch
from torch import nn

from .layout import CARD_FEATURES, OBS_SIZE
from .reference import BLUE, KASU, LIGHTS, RED, RIBBONS, TANE


class PolicyValue(nn.Module):
    def __init__(self, kind="mlp", width=128, layers=2):
        super().__init__()
        self.config = {"kind": kind, "width": width, "layers": layers}
        self.kind = kind
        if kind in ("mlp", "features", "relational"):
            extra = 0
            if kind in ("features", "relational"):
                groups = (
                    {0},
                    {8},
                    {28},
                    {40},
                    {44},
                    {32},
                    {20, 24, 36},
                    TANE,
                    RED,
                    BLUE,
                    RED | BLUE,
                    RIBBONS,
                    KASU,
                )
                roles = torch.zeros(48, len(groups))
                for column, members in enumerate(groups):
                    roles[list(members), column] = 1 / len(members)
                self.register_buffer("roles", roles)
                extra = 5 * len(groups) + 24
            self.body = nn.Sequential(
                nn.Linear(OBS_SIZE + extra, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
            )
            self.discard = nn.Linear(width, 48)
            self.pick = nn.Linear(width, 48)
            self.belief = nn.Linear(width, 48)
            if kind == "relational":
                attributes = torch.zeros(48, 20)
                attributes[torch.arange(48), torch.arange(48) // 4] = 1
                for column, members in enumerate(
                    (LIGHTS, TANE, RIBBONS, KASU, {8}, {28}, {40}, {32}), 12
                ):
                    attributes[list(members), column] = 1
                self.register_buffer("attributes", attributes)
                self.card = nn.Sequential(
                    nn.Linear(CARD_FEATURES + 20, width),
                    nn.SiLU(),
                    nn.Linear(width, width),
                    nn.SiLU(),
                )
                self.relation = nn.Sequential(nn.Linear(width * 4, width), nn.SiLU())
                self.discard, self.pick, self.belief = (
                    nn.Linear(width, 1) for _ in range(3)
                )
        elif kind == "transformer":
            if width % 4:
                raise ValueError("Transformer width must be divisible by four")
            attributes = torch.zeros(48, 20)
            attributes[torch.arange(48), torch.arange(48) // 4] = 1
            for column, members in enumerate(
                (LIGHTS, TANE, RIBBONS, KASU, {8}, {28}, {40}, {32}), 12
            ):
                attributes[list(members), column] = 1
            self.register_buffer("attributes", attributes)
            self.card = nn.Linear(CARD_FEATURES + 20, width)
            self.identity = nn.Embedding(48, width)
            self.context = nn.Linear(32, width)
            layer = nn.TransformerEncoderLayer(
                width, 4, 2 * width, dropout=0, batch_first=True, norm_first=True
            )
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.discard, self.pick, self.belief = (
                nn.Linear(width, 1) for _ in range(3)
            )
        else:
            raise ValueError(f"Unknown network {kind}")
        self.koi = nn.Linear(width, 2)
        self.value = nn.Linear(width, 1)
        for head in (self.discard, self.pick, self.koi):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, obs, mask):
        if self.kind in ("mlp", "features", "relational"):
            inputs = obs
            if self.kind in ("features", "relational"):
                cards = obs[:, :528].reshape(-1, 48, CARD_FEATURES)
                counts = (cards[:, :, :5].transpose(1, 2) @ self.roles).flatten(1)
                months = cards[:, :, :2].reshape(-1, 12, 4, 2).sum(2).flatten(1) / 4
                inputs = torch.cat([obs, counts, months], dim=-1)
            pooled = self.body(inputs)
            if self.kind == "relational":
                attributes = self.attributes.unsqueeze(0).expand(cards.shape[0], -1, -1)
                encoded = self.card(torch.cat([cards, attributes], dim=-1))
                width = encoded.shape[-1]
                # Shared scoring per candidate, with exact same-month aggregates.
                own = (
                    (encoded * cards[:, :, 0:1])
                    .reshape(-1, 12, 4, width)
                    .sum(2)
                    .repeat_interleave(4, dim=1)
                )
                field = (
                    (encoded * cards[:, :, 1:2])
                    .reshape(-1, 12, 4, width)
                    .sum(2)
                    .repeat_interleave(4, dim=1)
                )
                relation = self.relation(
                    torch.cat(
                        [encoded, own, field, pooled[:, None].expand(-1, 48, -1)],
                        dim=-1,
                    )
                )
                discard, pick, belief = (
                    head(relation).squeeze(-1)
                    for head in (self.discard, self.pick, self.belief)
                )
            else:
                discard, pick, belief = (
                    self.discard(pooled),
                    self.pick(pooled),
                    self.belief(pooled),
                )
        else:
            cards = obs[:, :528].reshape(-1, 48, CARD_FEATURES)
            attributes = self.attributes.unsqueeze(0).expand(cards.shape[0], -1, -1)
            features = self.card(
                torch.cat([cards, attributes], dim=-1)
            ) + self.identity.weight.unsqueeze(0)
            context = self.context(obs[:, 528:]).unsqueeze(1)
            features = self.norm(self.body(torch.cat([features, context], dim=1)))
            pooled = features[:, -1]
            discard, pick = (
                self.discard(features[:, :48]).squeeze(-1),
                self.pick(features[:, :48]).squeeze(-1),
            )
            belief = self.belief(features[:, :48]).squeeze(-1)
        card_logits = torch.where(obs[:, 528:529] > 0, discard, pick)
        logits = torch.cat([card_logits, self.koi(pooled)], dim=-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, self.value(pooled).squeeze(-1), belief


def sample_action(logits):
    logprobs = logits.log_softmax(-1)
    action = torch.multinomial(logprobs.exp(), 1).squeeze(-1)
    return action, logprobs.gather(1, action[:, None]).squeeze(-1)
