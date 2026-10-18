"""Isolated role-aware self-play PPO with matched policy-return controls."""

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

from koikoi.layout import PLAYER
from koikoi.model import sample_action
from koikoi.published_train import PublishedConfig, PublishedTrainer
from scripts.observer_selfplay_returns import (
    ObserverFeatures,
    observer_gae,
    policy_advantages,
)


@dataclass
class ObserverConfig(PublishedConfig):
    policy_returns: str = "observer"
    returns_backend: str = "torch"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.policy_returns not in ("observer", "signed")
            or self.returns_backend != "torch"
        ):
            raise ValueError(
                "Observer self-play requires observer/signed torch returns"
            )


def source_identities():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("observer_selfplay.py", "observer_selfplay_returns.py")
    }


class ObserverValue(nn.Sequential):
    """Row-wise value; a 560-column active observation has role bit one."""

    def forward(self, features):
        if features.shape[-1] == 560:
            features = torch.cat((features, torch.ones_like(features[..., :1])), dim=-1)
        if features.shape[-1] != 561:
            raise ValueError("Expected single-observer features")
        return super().forward(features)


class ObserverTrainer(PublishedTrainer):
    def __init__(self, config, actor_snapshot=None):
        super().__init__(config, actor_snapshot)
        old = self.model.value
        # Its entire state is copied below; do not perturb the original
        # trainer's sampling RNG merely to allocate the extra role column.
        with torch.random.fork_rng(devices=[]):
            first = nn.Linear(561, config.critic_width).to(self.env.device)
        with torch.no_grad():
            first.weight[:, :560].copy_(old[0].weight)
            first.weight[:, 560].zero_()
            first.bias.copy_(old[0].bias)
        self.model.value = ObserverValue(first, *list(old.children())[1:])
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": config.lr},
                {"params": self.model.value.parameters(), "lr": config.value_lr},
            ],
            eps=1e-5,
        )
        self.observer_features = ObserverFeatures(self.env)
        self.counters.update(value_rows=0, warmup_value_rows=0)
        self.provenance["observer_sources"] = source_identities()

    def _check_runtime(self):
        super()._check_runtime()
        if self.provenance["observer_sources"] != source_identities():
            raise ValueError("Observer self-play source identity changed")

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
            (value["0.weight"], torch.zeros_like(value["0.weight"][:, :1])), dim=1
        )
        obj.model.value.load_state_dict(value)
        obj.provenance["initialized_from"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
        }
        return obj

    @torch.no_grad()
    def collect(self):
        self._check_runtime()
        buffers = {
            key: []
            for key in (
                "obs",
                "mask",
                "action",
                "old_logp",
                "observer_obs",
                "value",
                "reward",
                "done",
                "player",
            )
        }
        totals = torch.zeros(2, dtype=torch.int64, device=self.env.device)
        for _ in range(self.config.horizon):
            obs, mask = self.observe()
            views = self.observer_features(self.env)
            value = self.model.value(views).squeeze(-1)
            logits = self.actor(obs, mask)[0]
            action, logp = sample_action(logits)
            player = self.env.state[:, PLAYER].clone()
            result = self.env.step(action)
            data = (
                obs,
                mask,
                action,
                logp,
                views,
                value,
                result[:, 0].float(),
                result[:, 2].bool(),
                player,
            )
            for name, tensor in zip(buffers, data):
                buffers[name].append(tensor)
            totals += result[:, 1:3].sum(0)
            self.env.reset_done()
        batch = {key: torch.stack(rows) for key, rows in buffers.items()}
        tail = self.model.value(self.observer_features(self.env)).squeeze(-1)
        _, targets = observer_gae(
            batch["reward"], batch["value"], batch["done"], tail, self.config.gae_lambda
        )
        advantage = policy_advantages(
            self.config.policy_returns,
            batch["reward"],
            batch["value"],
            batch["done"],
            batch["player"],
            tail,
            self.env.state[:, PLAYER],
            self.config.gae_lambda,
        )
        batch.update(advantage=advantage, **{"return": targets})
        rounds, matches = totals.cpu().tolist()
        self.counters["rounds"] += rounds
        self.counters["matches"] += matches
        decisions = self.config.envs * self.config.horizon
        self.counters["decisions"] += decisions
        self.counters["value_rows"] += 2 * decisions
        return {key: tensor.flatten(0, 1) for key, tensor in batch.items()}

    def minibatch_step(self, data):
        c = self.config
        logits = self.actor(data["obs"], data["mask"])[0]
        logprobs = logits.log_softmax(-1)
        delta = (
            logprobs.gather(1, data["action"][:, None]).squeeze(-1) - data["old_logp"]
        )
        ratio, adv = delta.exp(), data["advantage"]
        policy_loss = -torch.minimum(
            ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv
        ).mean()
        entropy = -(logprobs.exp() * logprobs).sum(-1).mean()
        value = self.model.value(data["observer_obs"]).squeeze(-1)
        # Mean across BOTH seat rows: doubling value examples does not double
        # the value-loss coefficient. Neither arm changes this convention.
        value_loss = (value - data["return"]).square().mean()
        loss = policy_loss + c.value_weight * value_loss - c.entropy * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.trainable.parameters(), 0.5)
        self.optimizer.step()
        self.counters["optimizer_steps_value"] += 1
        if self.counters["updates"] >= c.critic_warmup_updates:
            phases = data["obs"][:, 560]
            present = (
                torch.stack(
                    (
                        (phases == 0).any(),
                        ((phases == 1) | (phases == 2)).any(),
                        (phases == 3).any(),
                    )
                )
                .cpu()
                .tolist()
            )
            for name, active in zip(("discard", "pick", "koi"), present):
                self.counters["optimizer_steps_" + name] += int(active)
        with torch.no_grad():
            return torch.stack(
                (policy_loss, value_loss, entropy, ((ratio - 1) - delta).mean(), grad)
            )

    def update(self):
        warmup = self.counters["updates"] < self.config.critic_warmup_updates
        self.model.actor.requires_grad_(not warmup)
        before = self.counters.copy()
        batch = self.collect()
        adv = batch["advantage"]
        batch["advantage"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        metrics = torch.zeros(5, device=self.env.device)
        count = 0
        for _ in range(self.config.epochs):
            for indices in torch.randperm(len(adv), device=self.env.device).split(
                self.config.minibatch
            ):
                metrics += self.minibatch_step(
                    {key: tensor[indices] for key, tensor in batch.items()}
                )
                count += 1
        self.counters["updates"] += 1
        self.counters["warmup_updates" if warmup else "policy_updates"] += 1
        if warmup:
            for key in ("decisions", "rounds", "matches", "value_rows"):
                self.counters["warmup_" + key] += self.counters[key] - before[key]
        values = (metrics / count).cpu().tolist()
        if not all(np.isfinite(values)):
            raise FloatingPointError("Non-finite observer self-play metrics")
        return dict(
            zip(
                ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"),
                values,
            )
        ) | {"critic_only": warmup}

    def save(self, path):
        self._check_runtime()
        payload = {
            "format": "observer_selfplay_v1",
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
        if payload.get("format") != "observer_selfplay_v1":
            raise ValueError("Unsupported observer self-play checkpoint")
        obj = cls(ObserverConfig(**payload["config"]), payload["actor"])
        if obj.runtime != payload["runtime"]:
            raise ValueError("Observer self-play runtime mismatch")
        for key in ("source_sha256", "observer_sources", "published_initialization"):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError(
                    "Observer self-play source/initialization identity mismatch"
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
