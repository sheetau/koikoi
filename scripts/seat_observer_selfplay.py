"""Isolated public-seat critic extension with a same-shape masked control.

Reuses the frozen observer trainer's collection, returns and PPO update. Neither
condition changes the published actor input or the original study's sources.
"""

import copy
import hashlib
import io
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from koikoi.published_train import PublishedPolicyValue
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer
from scripts.observer_selfplay_returns import ObserverFeatures


@dataclass
class SeatObserverConfig(ObserverConfig):
    seat_context: str = "public"

    def __post_init__(self):
        super().__post_init__()
        if self.seat_context not in ("public", "masked"):
            raise ValueError("Expected public or masked seat context")


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SeatObserverFeatures(ObserverFeatures):
    def __init__(self, env, context):
        super().__init__(env)
        if context not in ("public", "masked"):
            raise ValueError("Expected public or masked seat context")
        self.context = context

    @torch.no_grad()
    def __call__(self, env):
        features = super().__call__(env)
        seat = self.seats.reshape(env.count, 2, 1).to(features.dtype)
        if self.context == "masked":
            seat = torch.zeros_like(seat)
        return torch.cat((features, seat), dim=-1)


class SeatValue(nn.Sequential):
    def forward(self, features):
        if features.shape[-1] != 562:
            raise ValueError("Seat critic requires explicit observer role and seat")
        return super().forward(features)


class SeatPolicyValue(nn.Module):
    def __init__(self, original, context):
        super().__init__()
        self.actor = original.actor
        self.temperature = original.temperature
        self.context = context
        old = original.value
        with torch.random.fork_rng(devices=[]):
            first = nn.Linear(562, old[0].out_features).to(old[0].weight.device)
        with torch.no_grad():
            first.weight.zero_()
            first.weight[:, : old[0].in_features].copy_(old[0].weight)
            first.bias.copy_(old[0].bias)
        self.value = SeatValue(first, *list(old.children())[1:])

    def active_features(self, obs):
        if obs.ndim != 2 or obs.shape[1] != 15561:
            raise ValueError("Expected packed published observation")
        # Public actor row51 is absolute dealer=1; ordinary column533 means
        # viewer-is-dealer. Together they determine the active viewer's seat.
        dealer = obs[:, 561 + 51 * 50 : 561 + 51 * 50 + 1]
        relative = obs[:, 533:534].bool()
        seat = torch.where(relative, dealer, 1 - dealer)
        if self.context == "masked":
            seat = torch.zeros_like(seat)
        return torch.cat((obs[:, :560], torch.ones_like(seat), seat), dim=-1)

    def forward(self, obs, mask):
        features = self.active_features(obs)
        scores = self.actor(obs[:, 561:].reshape(-1, 300, 50), obs[:, 560].int(), mask)
        logits = (scores / self.temperature).masked_fill(~mask, -1e9)
        return logits, self.value(features).squeeze(-1), obs.new_zeros((len(obs), 48))


class SeatObserverTrainer(ObserverTrainer):
    def __init__(self, config, actor_snapshot=None):
        super().__init__(config, actor_snapshot)
        self.model = SeatPolicyValue(self.model, config.seat_context)
        self.trainable = self.model
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": config.lr},
                {"params": self.model.value.parameters(), "lr": config.value_lr},
            ],
            eps=1e-5,
        )
        self.observer_features = SeatObserverFeatures(self.env, config.seat_context)
        self.provenance["seat_observer_source"] = source_identity()

    def _check_runtime(self):
        super()._check_runtime()
        if self.provenance["seat_observer_source"] != source_identity():
            raise ValueError("Seat observer source identity changed")
        if (
            self.model.context != self.config.seat_context
            or self.observer_features.context != self.config.seat_context
        ):
            raise ValueError("Seat context changed")

    @classmethod
    def from_parent(cls, config, path):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_ppo_v1":
            raise ValueError("Expected published PPO parent")
        for key in ("profile", "endgame", "initial_variant", "critic_width"):
            if payload["config"][key] != getattr(config, key):
                raise ValueError("Parent rules/initialization/value shape mismatch")
        obj = cls(config, payload["actor"])
        if (
            obj.provenance["published_initialization"]
            != payload["provenance"]["published_initialization"]
        ):
            raise ValueError("Parent actor identity mismatch")
        value = dict(payload["value"])
        value["0.weight"] = torch.cat(
            (value["0.weight"], torch.zeros_like(value["0.weight"][:, :2])), dim=1
        )
        obj.model.value.load_state_dict(value)
        obj.provenance["initialized_from"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
        }
        return obj

    def save(self, path):
        self._check_runtime()
        payload = {
            "format": "seat_observer_selfplay_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "value": self.model.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state,
            "counters": self.counters,
            "provenance": self.provenance,
            "runtime": self.runtime,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "seat_observer_selfplay_v1":
            raise ValueError("Unsupported seat observer checkpoint")
        obj = cls(SeatObserverConfig(**payload["config"]), payload["actor"])
        if obj.runtime != payload["runtime"]:
            raise ValueError("Seat observer runtime mismatch")
        for key in (
            "source_sha256",
            "observer_sources",
            "seat_observer_source",
            "published_initialization",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError(
                    "Seat observer source/initialization identity mismatch"
                )
        obj.provenance = copy.deepcopy(payload["provenance"])
        obj.provenance["parent_checkpoint"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
        }
        obj.model.value.load_state_dict(payload["value"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.env.state.copy_(payload["env"])
        obj.counters = payload["counters"]
        obj.model.actor.requires_grad_(
            obj.counters["updates"] >= obj.config.critic_warmup_updates
        )
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        return obj


def load_policy(path):
    """Load the public actor/value model on CPU; does not resume training."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "seat_observer_selfplay_v1":
        raise ValueError("Unsupported seat observer checkpoint")
    config = SeatObserverConfig(**payload["config"])
    if (
        payload["provenance"]["published_initialization"]
        != payload["actor"]["initialization"]
    ):
        raise ValueError("Seat observer policy initialization mismatch")
    model = SeatPolicyValue(
        PublishedPolicyValue(config, payload["actor"], "cpu"), config.seat_context
    )
    model.value.load_state_dict(payload["value"])
    return model.eval()
