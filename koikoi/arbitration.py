"""Public-observation arbitration between frozen win- and point-oriented experts.

Only proposed legal actions enter the trainable policy. Agreement produces one
action, hence zero policy gradient. The simulator and expert actors are unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import BatchEnv
from .layout import PHASE
from .published_actor import PublishedActor
from .published_opponent import runtime_identity
from .train import Config, Trainer, provenance


def read_checkpoint(path):
    content = Path(path).read_bytes()
    return torch.load(
        io.BytesIO(content), map_location="cpu", weights_only=False
    ), hashlib.sha256(content).hexdigest()


@dataclass
class ArbitrationConfig(Config):
    kind: str = "arbitration"
    profile: str = "paper8"
    envs: int = 128
    horizon: int = 128
    minibatch: int = 512
    width: int = 128
    cpu_threads: int = 2
    endgame: bool = True
    lr: float = 0.0001
    value_lr: float = 0.0001
    critic_warmup_updates: int = 20
    context: str = "full"
    base_probability: float = 0.9

    def __post_init__(self):
        super().__post_init__()
        if (
            self.kind != "arbitration"
            or self.profile != "paper8"
            or self.algorithm != "ppo"
            or self.reward != "match"
            or self.precision != "float32"
            or self.opponent_fraction
            or self.published_opponent
            or self.ppo_value != "shared"
            or self.policy_scope != "all"
            or self.koi_width
            or self.belief_weight
            or self.cuda_graph
            or self.cuda_rollout
            or self.complete_matches
            or self.round_continuation is not None
            or self.deterministic_cards
            or not self.endgame
        ):
            raise ValueError(
                "Arbitration requires eager float32 paper8 assisted self-play PPO"
            )
        if self.context not in ("full", "phase", "no_match"):
            raise ValueError("Unknown arbitration context")
        if not 0.5 < self.base_probability < 1:
            raise ValueError(
                "Initial base probability must lie strictly between 0.5 and 1"
            )
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.value_lr)):
            raise ValueError("Learning rates must be finite and positive")
        if (
            type(self.critic_warmup_updates) is not int
            or self.critic_warmup_updates < 0
        ):
            raise ValueError("Invalid critic warmup")


class ExpertBank(nn.Module):
    """Frozen experts consume only native public features and legal masks."""

    def __init__(self, snapshots, device):
        super().__init__()
        if len(snapshots) != 2:
            raise ValueError("Expected base and point expert snapshots")
        self.experts = nn.ModuleList(
            [PublishedActor(s, device).requires_grad_(False) for s in snapshots]
        )
        self.eval()

    @torch.no_grad()
    def observe(self, env, labels=False):
        ordinary = env.observe(labels=labels)
        features, legal = env.published_observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=env.device)[None] == solved[:, None]
        )
        proposals = torch.stack(
            [
                actor(features, env.state[:, PHASE], legal).argmax(-1)
                for actor in self.experts
            ],
            dim=1,
        )
        proposed_mask = torch.zeros_like(legal).scatter_(1, proposals, True) & legal
        public = torch.cat([ordinary[0], proposals.float()], dim=1)
        return (
            (public, proposed_mask, ordinary[2]) if labels else (public, proposed_mask)
        )

    def snapshots(self):
        return [actor.checkpoint() for actor in self.experts]


class ArbitrationPolicyValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config.context
        self.actor = (
            nn.Linear(4, 2)
            if config.context == "phase"
            else nn.Sequential(
                nn.Linear(660, config.width),
                nn.Tanh(),
                nn.Linear(config.width, config.width),
                nn.Tanh(),
                nn.Linear(config.width, 2),
            )
        )
        last = self.actor if self.context == "phase" else self.actor[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[0] = math.log(
                config.base_probability / (1 - config.base_probability)
            )
        self.value = nn.Sequential(
            nn.Linear(560, config.critic_width),
            nn.Tanh(),
            nn.Linear(config.critic_width, config.critic_width),
            nn.Tanh(),
            nn.Linear(config.critic_width, 1),
        )
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, obs, mask):
        if obs.ndim != 2 or obs.shape[1] != 562:
            raise ValueError(
                "Arbitration requires 560 public features and two proposal IDs"
            )
        proposals = obs[:, 560:].long()
        public = obs[:, :560]
        if self.context == "phase":
            inputs = public[:, 528:532]
        else:
            if self.context == "no_match":
                public = public.clone()
                public[:, 534:538] = 0  # Round count and both scores, controller only.
            inputs = torch.cat(
                [public, F.one_hot(proposals, 50).flatten(1).float()], dim=1
            )
        choices = self.actor(inputs)
        duplicate = proposals[:, 0] == proposals[:, 1]
        second = torch.where(duplicate, choices[:, 0], choices[:, 1])
        logits = obs.new_full((len(obs), 50), -1e9)
        logits = logits.scatter(1, proposals[:, :1], choices[:, :1])
        logits = logits.scatter(1, proposals[:, 1:], second[:, None]).masked_fill(
            ~mask, -1e9
        )
        return (
            logits,
            self.value(obs[:, :560]).squeeze(-1),
            obs.new_zeros((len(obs), 48)),
        )


class ArbitrationTrainer(Trainer):
    def __init__(self, config, snapshots, parent=None, value_snapshot=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Arbitration requires float32 observations")
        self.config = config
        self._config_json = json.dumps(asdict(config), sort_keys=True)
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.env = BatchEnv(config.envs, config.device, config.profile, config.seed)
        self.bank = ExpertBank(snapshots, config.device)
        self.model = ArbitrationPolicyValue(config).to(config.device)
        if value_snapshot is not None:
            self.model.value.load_state_dict(value_snapshot)
        self.trainable = self.model
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": config.lr},
                {"params": self.model.value.parameters(), "lr": config.value_lr},
            ],
            eps=1e-5,
        )
        self.frozen = self.critic = self.ppo_critic = self.opponent_value = None
        self.round_continuation = None
        self.captured_minibatch = self.captured_rollout = None
        self.counters = dict.fromkeys(
            (
                "updates",
                "decisions",
                "rounds",
                "matches",
                "policy_updates",
                "warmup_updates",
                "disagreement_decisions",
                "point_choices",
                "optimizer_steps",
            ),
            0,
        )
        self.provenance = provenance()
        self.provenance["weight_initialization"] = copy.deepcopy(parent)
        self.runtime = runtime_identity(self.env.device)
        self.expert_digests = [s["current"]["parameters_sha256"] for s in snapshots]

    @classmethod
    def from_parent(cls, config, path):
        payload, digest = read_checkpoint(path)
        if (
            payload.get("format") != "published_ppo_v1"
            or payload["config"]["profile"] != "paper8"
        ):
            raise ValueError("Expected a paper8 published PPO parent")
        point = PublishedActor.from_published("point", "cpu").checkpoint()
        parent = {
            "path": str(path),
            "sha256": digest,
            "counters": payload["counters"],
            "config": payload["config"],
        }
        return cls(config, [payload["actor"], point], parent, payload["value"])

    def observe(self, labels=False):
        return self.bank.observe(self.env, labels)

    def collect(self):
        if json.dumps(asdict(self.config), sort_keys=True) != self._config_json:
            raise ValueError("Training configuration changed")
        if runtime_identity(self.env.device) != self.runtime:
            raise ValueError("Training runtime changed")
        batch = super().collect()
        different = batch["mask"].sum(-1) > 1
        self.counters["disagreement_decisions"] += int(different.sum())
        self.counters["point_choices"] += int(
            (different & (batch["action"] == batch["obs"][:, 561].long())).sum()
        )
        return batch

    def update(self):
        warmup = self.counters["updates"] < self.config.critic_warmup_updates
        self.model.actor.requires_grad_(not warmup)
        metrics = super().update()
        self.counters["warmup_updates" if warmup else "policy_updates"] += 1
        metrics["critic_only"] = warmup
        return metrics

    def minibatch_step(self, data):
        metrics = super().minibatch_step(data)
        self.counters["optimizer_steps"] += 1
        return metrics

    def save(self, path):
        snapshots = self.bank.snapshots()
        if [
            s["current"]["parameters_sha256"] for s in snapshots
        ] != self.expert_digests:
            raise ValueError("Frozen expert weights changed")
        payload = {
            "format": "arbitration_v1",
            "config": asdict(self.config),
            "experts": snapshots,
            "model": self.model.state_dict(),
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
        payload, digest = read_checkpoint(path)
        if payload.get("format") != "arbitration_v1":
            raise ValueError("Unsupported arbitration checkpoint")
        obj = cls(
            ArbitrationConfig(**payload["config"]),
            payload["experts"],
            payload["provenance"]["weight_initialization"],
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Runtime changed")
        if (
            not allow_code_change
            and obj.provenance["source_sha256"]
            != payload["provenance"]["source_sha256"]
        ):
            raise ValueError("Source changed")
        obj.provenance["resumed_checkpoint"] = {
            "sha256": digest,
            "counters": payload["counters"],
        }
        obj.model.load_state_dict(payload["model"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.env.state.copy_(payload["env"])
        obj.counters = payload["counters"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        return obj


class ArbitrationAgent:
    def __init__(self, checkpoint, device="cpu"):
        payload, self.checkpoint_sha256 = read_checkpoint(checkpoint)
        if payload.get("format") != "arbitration_v1":
            raise ValueError("Unsupported arbitration checkpoint")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.config = ArbitrationConfig(**payload["config"])
        self.bank = ExpertBank(payload["experts"], device)
        self.model = ArbitrationPolicyValue(self.config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.env = BatchEnv(1, device, "paper8")
        self.counters = payload["counters"]
        self.provenance = payload["provenance"]

    @torch.no_grad()
    def batch_act(self, env):
        obs, mask = self.bank.observe(env)
        return self.model(obs, mask)[0].argmax(-1)

    def act(self, env):
        self.env.state.copy_(torch.from_numpy(env.state).to(self.env.device)[None])
        return int(self.batch_act(self.env)[0])
