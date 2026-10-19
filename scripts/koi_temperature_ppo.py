"""Isolated PPO sampling-temperature intervention; greedy inference is unchanged."""

import copy
import hashlib
import io
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import KOI
from koikoi.published_train import (
    PublishedConfig,
    PublishedPolicyValue,
    PublishedTrainer,
)


@dataclass
class KoiTemperatureConfig(PublishedConfig):
    koi_temperature: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.koi_temperature) or self.koi_temperature <= 0:
            raise ValueError("Koi sampling temperature must be positive and finite")


class KoiTemperatureValue(PublishedPolicyValue):
    """Reuse parameter objects and preserve the original scalar division elsewhere."""

    def __init__(self, original, koi_temperature):
        nn.Module.__init__(self)
        self.actor, self.value = original.actor, original.value
        self.temperature = original.temperature
        self.koi_temperature = koi_temperature

    def forward(self, obs, mask):
        if obs.ndim != 2 or obs.shape[1] != 15561:
            raise ValueError("Expected the existing packed public observation")
        phases = obs[:, 560].int()
        scores = self.actor(obs[:, 561:].reshape(-1, 300, 50), phases, mask)
        logits = scores / self.temperature
        if self.koi_temperature != self.temperature:
            selected = phases == KOI
            logits[selected] = scores[selected] / self.koi_temperature
        logits = logits.masked_fill(~mask, -1e9)
        value = self.value(obs[:, :560]).squeeze(-1)
        return logits, value, obs.new_zeros((len(obs), 48))


class KoiTemperatureTrainer(PublishedTrainer):
    def __init__(self, config, actor_snapshot=None):
        if not isinstance(config, KoiTemperatureConfig):
            raise TypeError("Explicit koi-temperature configuration required")
        super().__init__(config, actor_snapshot)
        self.model = KoiTemperatureValue(self.model, config.koi_temperature)
        self.trainable = self.model
        # Optimizer references are deliberately retained; no extra model or RNG
        # initialization is allowed between the original and wrapped control.
        assert {id(p) for g in self.optimizer.param_groups for p in g["params"]} == {
            id(p) for p in self.model.parameters()
        }
        self.provenance["koi_temperature_source"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    def _check_runtime(self):
        super()._check_runtime()
        if self.model.koi_temperature != self.config.koi_temperature:
            raise ValueError("Koi sampling temperature changed during training")

    def save(self, path):
        self._check_runtime()
        payload = {
            "format": "koi_temperature_ppo_v1",
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
        if payload.get("format") != "koi_temperature_ppo_v1":
            raise ValueError("Unsupported koi-temperature checkpoint")
        obj = cls(KoiTemperatureConfig(**payload["config"]), payload["actor"])
        if obj.runtime != payload["runtime"]:
            raise ValueError("Koi-temperature runtime mismatch")
        for key in (
            "source_sha256",
            "published_initialization",
            "koi_temperature_source",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Koi-temperature source/initialization mismatch")
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


class KoiTemperatureAgent(NeuralAgent):
    """Evaluate through the old singleton CPU path at the old score temperature."""

    def __init__(self, checkpoint):
        content = Path(checkpoint).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "koi_temperature_ppo_v1":
            raise ValueError("Unsupported koi-temperature policy")
        config = KoiTemperatureConfig(**payload["config"])
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Koi-temperature policy initialization mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        self.training_counters, self.training_provenance = (
            payload["counters"],
            payload["provenance"],
        )
        self.training_config = payload["config"]
        # Deliberately use PublishedPolicyValue, not the training-temperature class.
        self.model = PublishedPolicyValue(config, payload["actor"], "cpu")
        self.model.value.load_state_dict(payload["value"])
        self.model.eval()
        self.published_env = BatchEnv(1, "cpu", "paper8")
        self.device, self.deterministic = "cpu", True
        self.profile, self.belief_trained = "paper8", False
        self.requires_endgame = config.endgame
        self.solved_decisions = 0
