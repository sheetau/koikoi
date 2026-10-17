"""Experimental pretrained PPO against a fixed public opponent mixture.

This module reuses the existing signed-return and learner-masked PPO operations.
It does not change the production trainer, comparison agents or playable model.
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

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import EPISODE
from koikoi.opponents import OpponentValue
from koikoi.published_opponent import PublishedOpponent
from koikoi.published_train import (
    PublishedConfig,
    PublishedPolicyValue,
    PublishedTrainer,
)


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class ResponseConfig(PublishedConfig):
    opponent_fraction: float = 1.0
    published_opponent: str = "wp_point"

    def __post_init__(self):
        # Validate all inherited published-PPO restrictions on a separate config;
        # only the two explicitly supported opponent settings differ here.
        ordinary = asdict(self)
        ordinary.update(opponent_fraction=0.0, published_opponent="")
        PublishedConfig(**ordinary)
        if self.published_opponent not in ("wp", "point", "wp_point"):
            raise ValueError("Response opponent must be wp, point or wp_point")
        if not 0 < self.opponent_fraction <= 1:
            raise ValueError("Response opponent fraction must be in (0, 1]")
        if int(self.envs * self.opponent_fraction) < 1:
            raise ValueError("Response training needs at least one opponent lane")


class PublicMixture:
    """Seat and opponent schedules are fixed for each whole match.

    Across four episodes each lane visits both learner seats against both experts.
    Opponent identity is routing metadata, never an additional actor input.
    """

    def __init__(self, config, snapshots=None):
        variants = (
            ("wp", "point")
            if config.published_opponent == "wp_point"
            else (config.published_opponent,)
        )
        if snapshots is not None and set(snapshots) != set(variants):
            raise ValueError("Response opponent checkpoint variants mismatch")
        self.agents = {
            variant: PublishedOpponent(
                variant,
                config.device,
                config.endgame,
                None if snapshots is None else snapshots[variant],
            )
            for variant in variants
        }
        self.rows = torch.arange(config.envs, device=config.device)

    def ids(self, env):
        return (self.rows // 2 + env.state[:, EPISODE].long() // 2) % len(self.agents)

    def check_runtime(self):
        for agent in self.agents.values():
            agent.check_runtime()

    @torch.no_grad()
    def actions(self, env, learner):
        result = torch.zeros(env.count, dtype=torch.int64, device=env.state.device)
        ids = self.ids(env)
        for index, agent in enumerate(self.agents.values()):
            selected = (~learner) & (ids == index)
            values = agent.actions(env, ~selected)
            result = torch.where(selected, values, result)
        return result

    def checkpoint(self):
        return {name: agent.checkpoint() for name, agent in self.agents.items()}


class ResponseTrainer(PublishedTrainer):
    def __init__(self, config, actor_snapshot=None, opponent_snapshots=None):
        if not isinstance(config, ResponseConfig):
            raise TypeError("ResponseTrainer requires ResponseConfig")
        super().__init__(config, actor_snapshot)
        self.rows = torch.arange(config.envs, device=config.device)
        self.frozen_envs = int(config.envs * config.opponent_fraction)
        self.frozen = PublicMixture(config, opponent_snapshots)
        self.opponent_value = OpponentValue(config.critic_width).to(
            device=config.device, dtype=torch.float32
        )
        nn.init.zeros_(self.opponent_value.network[-1].weight)
        nn.init.zeros_(self.opponent_value.network[-1].bias)
        self.trainable = nn.ModuleDict(
            {"policy_value": self.model, "opponent_value": self.opponent_value}
        )
        self.optimizer.add_param_group(
            {"params": self.opponent_value.parameters(), "lr": config.value_lr}
        )
        self.counters.update(
            learner_decisions=0,
            published_decisions=0,
            published_rounds=0,
            published_matches=0,
            optimizer_steps_opponent_value=0,
        )
        self.opponent_work = {
            name: {"decisions": 0, "rounds": 0, "matches": 0}
            for name in self.frozen.agents
        }
        self.provenance["response_source_sha256"] = source_identity()
        self.provenance["response_opponents"] = {
            name: copy.deepcopy(agent.metadata)
            for name, agent in self.frozen.agents.items()
        }

    @classmethod
    def from_parent(cls, config, path):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_ppo_v1":
            raise ValueError("Response initialization requires a published PPO parent")
        if (
            payload["config"]["profile"] != config.profile
            or payload["config"]["endgame"] != config.endgame
        ):
            raise ValueError("Response parent rules/assistance mismatch")
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Response parent initialization identity mismatch")
        obj = cls(config, payload["actor"])
        obj.model.value.load_state_dict(payload["value"])
        obj.provenance["weight_initialization"] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
            "source_sha256": payload["provenance"]["source_sha256"],
        }
        return obj

    def _check_runtime(self):
        super()._check_runtime()
        if source_identity() != self.provenance["response_source_sha256"]:
            raise ValueError("Response implementation changed during training")

    def role_value(self, obs):
        return self.opponent_value(obs[:, :560]).float()

    def collect_device(self):
        # Accounting wraps the transition, not policy inference. IDs are read
        # before reset; each terminal event is attributed to its actual opponent.
        step = self.env.step
        counts = torch.zeros(
            (len(self.frozen.agents), 3), dtype=torch.int64, device=self.env.device
        )

        def counted_step(actions):
            ids = self.frozen.ids(self.env)
            learner = self.learner_rows()
            result = step(actions)
            for index in range(len(self.frozen.agents)):
                lanes = (ids == index) & (self.rows < self.frozen_envs)
                counts[index, 0] += ((~learner) & lanes).sum()
                counts[index, 1:] += (result[:, 1:3] * lanes[:, None]).sum(0)
            return result

        self.env.step = counted_step
        try:
            result = super().collect_device()
        finally:
            self.env.step = step
        for name, values in zip(self.frozen.agents, counts.cpu().tolist()):
            for key, value in zip(("decisions", "rounds", "matches"), values):
                self.opponent_work[name][key] += value
        return result

    def minibatch_step(self, data):
        result = super().minibatch_step(data)
        self.counters["optimizer_steps_opponent_value"] += 1
        return result

    def save(self, path):
        self._check_runtime()
        if source_identity() != self.provenance["response_source_sha256"]:
            raise ValueError("Response implementation changed during training")
        payload = {
            "format": "published_response_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "value": self.model.value.state_dict(),
            "opponent_value": self.opponent_value.state_dict(),
            "opponents": self.frozen.checkpoint(),
            "opponent_work": self.opponent_work,
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
    def restore(cls, path, allow_code_change=False):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_response_v1":
            raise ValueError("Unsupported response checkpoint")
        obj = cls(
            ResponseConfig(**payload["config"]), payload["actor"], payload["opponents"]
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Response runtime mismatch")
        for key in ("published_initialization", "response_opponents"):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Response initialization/opponent identity mismatch")
        for key in ("source_sha256", "response_source_sha256"):
            if (
                not allow_code_change
                and obj.provenance[key] != payload["provenance"][key]
            ):
                raise ValueError("Response source changed; exact resume rejected")
        obj.provenance["parent_checkpoint"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
        }
        if "weight_initialization" in payload["provenance"]:
            obj.provenance["weight_initialization"] = copy.deepcopy(
                payload["provenance"]["weight_initialization"]
            )
        obj.model.value.load_state_dict(payload["value"])
        obj.opponent_value.load_state_dict(payload["opponent_value"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.env.state.copy_(payload["env"])
        obj.counters = payload["counters"]
        obj.opponent_work = payload["opponent_work"]
        obj.model.actor.requires_grad_(
            obj.counters["updates"] >= obj.config.critic_warmup_updates
        )
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        return obj


class ResponseAgent(NeuralAgent):
    """Greedy CPU deployment of the learner only, using the existing action path."""

    def __init__(self, checkpoint):
        content = Path(checkpoint).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_response_v1":
            raise ValueError("Unsupported response policy checkpoint")
        config = ResponseConfig(**payload["config"])
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Response policy initialization mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        self.training_counters = payload["counters"]
        self.training_provenance = payload["provenance"]
        self.training_config = payload["config"]
        self.model = PublishedPolicyValue(config, payload["actor"], "cpu").eval()
        self.model.value.load_state_dict(payload["value"])
        self.published_env = BatchEnv(1, "cpu", "paper8")
        self.device, self.deterministic = "cpu", True
        self.profile, self.belief_trained = "paper8", False
        self.requires_endgame = config.endgame
        self.solved_decisions = 0
