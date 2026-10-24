"""Preserve a strong frozen actor while learning public-stock corrections."""

import torch

from .env import BatchEnv
from .layout import PHASE
from .public_stock_policy import PublicStockPolicy
from .published_actor import PublishedActor
from .reference import LIGHTS, RIBBONS, TANE


class PublicStockResidual(PublicStockPolicy):
    def __init__(self, parent, device="cuda"):
        super().__init__(parent["config"]["critic_width"])
        self.to(device)
        self.parent = parent
        self.temperature = parent["config"]["score_temperature"]
        self.teacher = (
            PublishedActor(parent["actor"], device).requires_grad_(False).eval()
        )
        self.encoder = BatchEnv(1, device, "paper8")
        with torch.no_grad():
            old = parent["value"]
            self.body[0].weight.zero_()
            self.body[0].weight[:, :560].copy_(old["0.weight"])
            self.body[0].bias.copy_(old["0.bias"])
            self.body[2].weight.copy_(old["2.weight"])
            self.body[2].bias.copy_(old["2.bias"])
            self.critic.weight.copy_(old["4.weight"])
            self.critic.bias.copy_(old["4.bias"])
            self.actor.weight.zero_()
            self.actor.bias.zero_()

    @torch.no_grad()
    def encode(self, batch):
        """The encoder receives state; the actor receives only public features.

        Published features intentionally ignore newly public stock cards. The
        correction network receives those cards through the 660-feature input.
        """
        if batch.count > 512:
            from .public_stock_search import view

            pieces = [
                self.encode(view(batch.state[i : i + 512], batch))
                for i in range(0, batch.count, 512)
            ]
            return torch.cat([x[0] for x in pieces]), torch.cat([x[1] for x in pieces])
        obs, legal = batch.observe()
        solved = batch.solved_actions()
        forced = solved >= 0
        legal[forced] = False
        legal[forced, solved[forced].long()] = True
        self.encoder.state, self.encoder.count = batch.state, batch.count
        features, _ = self.encoder.published_observe()
        phases = batch.state[:, PHASE]
        base = self.teacher(features, phases, legal) / self.temperature
        stock = phases == 5
        if stock.any():
            cards = obs[stock, :528].reshape(-1, 48, 11)
            weights = torch.ones(48, device=obs.device)
            weights[list(RIBBONS)], weights[list(TANE)], weights[list(LIGHTS)] = 2, 3, 5
            field = cards[:, :, 1].reshape(-1, 12, 4)
            gain = (field * weights.reshape(12, 4)).sum(2).repeat_interleave(4, 1)
            match = field.sum(2).repeat_interleave(4, 1) > 0
            base[stock, :48] = 0.1 * torch.where(match, gain + weights, -weights)
        # Invalid entries are finite placeholders, never selected or optimized.
        base.masked_fill_(~legal, 0)
        return torch.cat((obs, base), 1), legal

    def forward(self, observation, legal):
        features = self.body(observation[:, :660])
        logits = observation[:, 660:] + self.actor(features)
        return logits.masked_fill(~legal, -1e9), self.critic(features).squeeze(-1)

    def correction_state(self):
        return {
            k: v for k, v in self.state_dict().items() if not k.startswith("teacher.")
        }


def load_residual(path, device="cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "public_stock_residual_v1":
        raise ValueError("Expected a public-stock residual checkpoint")
    model = PublicStockResidual(payload["parent"], device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected or any(not k.startswith("teacher.") for k in missing):
        raise ValueError("Incomplete correction checkpoint")
    return model.eval(), payload
