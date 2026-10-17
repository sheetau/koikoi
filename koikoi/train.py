"""Restartable self-play PPO with optional frozen opponents and match returns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .capture import CapturedMinibatch, CapturedRollout
from .env import BatchEnv
from .estimators import (
    CentralQ,
    PPOValue,
    critic_input,
    native_gae,
    ppo_value_input,
    q_boost,
)
from .layout import PLAYER
from .model import PolicyValue, sample_action
from .opponents import FrozenOpponent, OpponentValue, learner_turn


@dataclass
class Config:
    algorithm: str = "ppo"
    device: str = "cuda"
    profile: str = "research12"
    seed: int = 1
    envs: int = 1024
    horizon: int = 128
    kind: str = "mlp"
    width: int = 128
    layers: int = 2
    lr: float = 0.0003
    epochs: int = 3
    minibatch: int = 8192
    clip: float = 0.2
    entropy: float = 0.02
    value_weight: float = 0.5
    belief_weight: float = 0.0
    gae_lambda: float = 0.95
    reward: str = "match"
    cpu_threads: int = 4
    critic_width: int = 256
    critic_epochs: int = 3
    precision: str = "float32"
    returns_backend: str = "native"
    endgame: bool = False
    cuda_graph: bool = False
    cuda_rollout: bool = False
    opponent_fraction: float = 0.0
    published_opponent: str = ""
    ppo_value: str = "shared"
    policy_scope: str = "all"
    koi_width: int = 0
    deterministic_cards: bool = False
    complete_matches: bool = False
    round_continuation: dict | None = None

    def __post_init__(self):
        if self.round_continuation is not None:
            if (
                not self.complete_matches
                or self.profile != "paper8"
                or self.gae_lambda != 1
            ):
                raise ValueError(
                    "Round continuation requires complete paper8 matches and lambda1"
                )
            from .round_continuation import validate_spec

            validate_spec(self.round_continuation)
        if self.complete_matches and (
            self.algorithm != "ppo"
            or self.reward != "match"
            or self.opponent_fraction
            or self.ppo_value != "shared"
            or self.policy_scope != "all"
            or self.cuda_graph
            or self.cuda_rollout
        ):
            raise ValueError(
                "Complete matches require eager pure-self-play match PPO "
                "with shared value and all-action policy"
            )
        if self.published_opponent not in ("", "wp", "point"):
            raise ValueError("published_opponent must be empty, wp or point")
        if self.published_opponent and (
            not self.opponent_fraction
            or self.profile != "paper8"
            or self.reward != "match"
            or self.cuda_rollout
        ):
            raise ValueError(
                "Published opponent requires paper8 match rewards, positive "
                "opponent_fraction and eager collection (cuda_rollout=false)"
            )
        if self.deterministic_cards and self.policy_scope != "koi":
            raise ValueError(
                "Deterministic training cards require a frozen koi-only policy"
            )
        if self.policy_scope not in ("all", "koi") or self.koi_width < 0:
            raise ValueError("Invalid policy scope or koi width")
        if self.policy_scope == "koi" and (
            self.algorithm != "ppo"
            or self.opponent_fraction
            or self.ppo_value != "observation"
            or self.reward != "match"
            or self.belief_weight
        ):
            raise ValueError(
                "Koi-only learning requires self-play match PPO, observation critic, "
                "and no belief loss"
            )
        if self.ppo_value not in ("shared", "observation", "privileged"):
            raise ValueError("Unknown PPO value mode")
        if self.ppo_value != "shared" and (
            self.algorithm != "ppo" or self.opponent_fraction
        ):
            raise ValueError("Separate PPO value currently requires pure self-play PPO")
        if not 0 <= self.opponent_fraction <= 1:
            raise ValueError("opponent_fraction must be between zero and one")
        if self.opponent_fraction and (
            self.algorithm != "ppo" or int(self.envs * self.opponent_fraction) < 1
        ):
            raise ValueError(
                "Frozen opponents require PPO and at least one environment"
            )
        if self.endgame and self.reward != "match":
            raise ValueError(
                "Endgame dominance applies to match outcome, not point rewards"
            )
        if self.returns_backend not in ("native", "torch"):
            raise ValueError("Expected native or torch returns_backend")
        if self.precision not in ("float32", "bf16"):
            raise ValueError("Expected float32 or bf16 precision")
        if self.algorithm not in ("ppo", "vrpo") or self.reward not in (
            "match",
            "round",
        ):
            raise ValueError("Unknown algorithm or reward")
        if (
            min(
                self.envs,
                self.horizon,
                self.width,
                self.epochs,
                self.minibatch,
                self.cpu_threads,
                self.critic_epochs,
                self.critic_width,
            )
            < 1
        ):
            raise ValueError("Training sizes must be positive")
        if not 0 <= self.gae_lambda <= 1 or self.belief_weight < 0:
            raise ValueError("Invalid loss/trace configuration")
        if self.cuda_graph:
            if not self.device.startswith("cuda") or self.algorithm != "ppo":
                raise ValueError("CUDA graphs currently support CUDA PPO only")
            size = self.envs * self.horizon
            if size % min(size, self.minibatch):
                raise ValueError("CUDA graph minibatches must all have the same size")
        if self.cuda_rollout and (
            not self.device.startswith("cuda") or self.algorithm != "ppo"
        ):
            raise ValueError("Captured rollouts currently support CUDA PPO only")


def signed_gae(rewards, values, dones, players, next_value, next_player, lam=0.95):
    """All values are relative to the acting player; chance/forced steps are folded."""
    adv = torch.zeros_like(values)
    carry = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        sign = torch.where(players[t] == next_player, 1.0, -1.0)
        live = (~dones[t]).float()
        delta = rewards[t] + live * sign * next_value - values[t]
        carry = delta + live * sign * lam * carry
        adv[t] = carry
        next_value, next_player = values[t], players[t]
    return adv, adv + values


def provenance():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"
    return {
        "revision": revision,
        "source_sha256": digest.hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class Trainer:
    def __init__(self, config: Config, published_snapshot=None):
        self.config = config
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.env = BatchEnv(config.envs, config.device, config.profile, config.seed)
        self.model = PolicyValue(
            config.kind, config.width, config.layers, config.koi_width
        ).to(config.device)
        if config.policy_scope == "koi":
            self.model.requires_grad_(False)
            for head in (self.model.koi, self.model.value, self.model.koi_residual):
                if head is not None:
                    head.requires_grad_(True)
        self.opponent_value = (
            OpponentValue(config.width).to(config.device)
            if config.opponent_fraction
            else None
        )
        self.trainable = (
            torch.nn.ModuleDict(
                {"actor": self.model, "opponent_value": self.opponent_value}
            )
            if self.opponent_value is not None
            else self.model
        )
        self.ppo_critic = None
        if config.ppo_value != "shared":
            # New critic initialization must not shift actor sampling RNG.
            with torch.random.fork_rng(devices=[]):
                self.ppo_critic = PPOValue(config.critic_width).to(config.device)
            self.trainable = torch.nn.ModuleDict(
                {"actor": self.model, "ppo_critic": self.ppo_critic}
            )
        self.frozen = None
        self.frozen_envs = int(config.envs * config.opponent_fraction)
        self.rows = torch.arange(config.envs, device=config.device)
        self.optimizer = torch.optim.Adam(
            self.trainable.parameters(),
            lr=config.lr,
            eps=1e-5,
            capturable=config.cuda_graph,
        )
        self.captured_minibatch = None
        self.captured_rollout = None
        self.critic = (
            CentralQ(config.critic_width).to(config.device)
            if config.algorithm == "vrpo"
            else None
        )
        self.critic_optimizer = (
            torch.optim.Adam(self.critic.parameters(), lr=config.lr, eps=1e-5)
            if self.critic
            else None
        )
        self.counters = {"updates": 0, "decisions": 0, "rounds": 0, "matches": 0}
        if config.complete_matches:
            self.counters["collection_rows"] = 0
        if config.opponent_fraction:
            self.counters["learner_decisions"] = 0
        if config.published_opponent:
            self.counters.update(
                published_decisions=0, published_rounds=0, published_matches=0
            )
        if config.policy_scope == "koi":
            self.counters["policy_decisions"] = 0
        self.provenance = provenance()
        self.round_continuation = None
        if config.round_continuation is not None:
            from .round_continuation import RoundContinuation

            self.round_continuation = RoundContinuation(
                config.round_continuation, config.device
            )
            self.provenance["round_continuation_sha256"] = (
                self.round_continuation.sha256
            )
        if config.published_opponent:
            from .published_opponent import PublishedOpponent

            self.frozen = PublishedOpponent(
                config.published_opponent,
                config.device,
                config.endgame,
                published_snapshot,
            )
            self.provenance["frozen_opponent"] = {
                **self.frozen.metadata,
                "frozen_envs": self.frozen_envs,
            }

    def set_frozen_opponent(self, path):
        """Load a fixed opponent for a new experiment, not an in-flight match."""
        if not self.config.opponent_fraction:
            raise ValueError("Set opponent_fraction before loading a frozen opponent")
        if (
            any(self.counters.values())
            or self.optimizer.state
            or self.frozen is not None
        ):
            raise ValueError("Frozen opponent loading requires a fresh trainer")
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        self._load_frozen(payload["config"], payload["model"])
        self.provenance["frozen_opponent"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "config": payload["config"],
            "counters": payload["counters"],
            "provenance": payload["provenance"],
            "frozen_envs": self.frozen_envs,
        }

    def _load_frozen(self, config, weights):
        if config["profile"] != self.config.profile:
            raise ValueError("Frozen opponent profile mismatch")
        if config.get("endgame", False) != self.config.endgame:
            raise ValueError("Frozen opponent endgame setting must match training")
        self.frozen = FrozenOpponent(config, weights, self.config.device)

    def learner_rows(self):
        return learner_turn(self.env.state, self.rows, self.frozen_envs)

    def role_value(self, obs):
        with torch.autocast(
            device_type=self.env.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
            cache_enabled=not (self.config.cuda_graph or self.config.cuda_rollout),
        ):
            value = self.opponent_value(obs)
        return value.float()

    def observe(self, labels=False):
        result = self.env.observe(labels=labels)
        if self.config.endgame:
            solved = self.env.solved_actions()
            result[1].logical_and_(
                (solved[:, None] < 0)
                | (torch.arange(50, device=self.env.device)[None] == solved[:, None])
            )
        return result

    def initialize_weights(self, path):
        """Start a new experiment from actor weights, with fresh state and counters."""
        if any(self.counters.values()) or self.optimizer.state:
            raise ValueError("Weight initialization requires a fresh trainer")
        content = Path(path).read_bytes()
        if (
            self.round_continuation is not None
            and hashlib.sha256(content).hexdigest()
            != self.round_continuation.parent_sha256
        ):
            raise ValueError(
                "Round continuation requires its calibrated parent checkpoint"
            )
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        for field in ("kind", "width", "layers"):
            if payload["config"][field] != getattr(self.config, field):
                raise ValueError(f"Initialization architecture mismatch: {field}")
        source_width = payload["config"].get("koi_width", 0)
        if source_width == self.config.koi_width:
            self.model.load_state_dict(payload["model"])
        elif source_width == 0 and self.config.koi_width > 0:
            initial = self.model.state_dict()
            base_keys = {k for k in initial if not k.startswith("koi_residual.")}
            if set(payload["model"]) != base_keys:
                raise ValueError(
                    "Initialization model keys do not match the base actor"
                )
            self.model.load_state_dict(initial | payload["model"])
        else:
            raise ValueError("Initialization architecture mismatch: koi_width")
        self.provenance["initialized_from"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "profile": payload["config"]["profile"],
            "counters": payload["counters"].copy(),
            "provenance": payload["provenance"],
        }

    def enable_rollout_capture(self):
        """Explicit execution change, retaining optimizer, games, RNG and budget."""
        previous = self.config.cuda_rollout
        if previous:
            return
        self.config = replace(self.config, cuda_rollout=True)
        self.provenance["execution_change"] = {
            "cuda_rollout": {"from": previous, "to": True},
            "retains_training_state": True,
        }

    def actor(self, obs, mask):
        with torch.autocast(
            device_type=self.env.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
            cache_enabled=not (self.config.cuda_graph or self.config.cuda_rollout),
        ):
            result = self.model(obs, mask)
        return tuple(value.float() for value in result)

    def central_critic(self, features):
        with torch.autocast(
            device_type=self.env.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
        ):
            result = self.critic(features)
        return result.float()

    def ppo_critic_value(self, features):
        with torch.autocast(
            device_type=self.env.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
            cache_enabled=not (self.config.cuda_graph or self.config.cuda_rollout),
        ):
            value = self.ppo_critic(features)
        return value.float()

    @torch.no_grad()
    def collect(self):
        self._check_round_continuation()
        if self.config.complete_matches:
            from .complete_matches import collect_complete_matches

            batch, totals, rows = collect_complete_matches(self)
            rounds, matches = totals.cpu().tolist()
            self.counters["rounds"] += rounds
            self.counters["matches"] += matches
            self.counters["decisions"] += len(batch["action"])
            self.counters["collection_rows"] += rows
            return batch
        if self.config.cuda_rollout:
            if self.captured_rollout is None:
                self.captured_rollout = CapturedRollout(
                    self.collect_device, self.env.state
                )
            batch, totals = self.captured_rollout()
        else:
            batch, totals = self.collect_device()
        counts = totals.cpu().tolist()
        rounds, matches = counts[:2]
        if self.config.opponent_fraction:
            self.counters["learner_decisions"] += int(batch["learner"].sum())
        if self.config.published_opponent:
            self.counters["published_decisions"] += int((1 - batch["learner"]).sum())
            self.counters["published_rounds"] += counts[2]
            self.counters["published_matches"] += counts[3]
        if self.config.policy_scope == "koi":
            self.counters["policy_decisions"] += int(
                batch["mask"][:, 48:].any(-1).sum()
            )
        self.counters["rounds"] += rounds
        self.counters["matches"] += matches
        self.counters["decisions"] += self.config.envs * self.config.horizon
        return batch

    @torch.no_grad()
    def collect_device(self):
        """Tensor-only collection; Python counters stay outside graph replay."""
        if self.config.opponent_fraction and self.frozen is None:
            raise ValueError("Load a frozen opponent before collection")
        if self.config.published_opponent:
            self.frozen.check_runtime()
        buffers = {
            k: []
            for k in (
                "obs",
                "mask",
                "target",
                "action",
                "old_logp",
                "value",
                "reward",
                "done",
                "player",
            )
        }
        if self.frozen is not None:
            buffers["learner"] = []
        if self.critic:
            buffers.update({"critic_obs": [], "q": [], "probs": []})
        if self.ppo_critic is not None:
            buffers["ppo_value_obs"] = []
        totals = torch.zeros(
            4 if self.config.published_opponent else 2,
            dtype=torch.int64,
            device=self.config.device,
        )
        for _ in range(self.config.horizon):
            obs, mask, target = self.observe(labels=True)
            player = self.env.state[:, PLAYER].clone()
            logits, value, _ = self.actor(obs, mask)
            action, logp = sample_action(logits)
            if self.config.deterministic_cards:
                koi = mask[:, 48:].any(-1)
                action = torch.where(koi, action, logits.argmax(-1))
                # Frozen card actions have probability one under this behavior.
                # Their rows never contribute to the PPO policy objective.
                logp = torch.where(koi, logp, 0.0)
            if self.ppo_critic is not None:
                features = ppo_value_input(obs, self.env.state, self.config.ppo_value)
                buffers["ppo_value_obs"].append(features)
                value = self.ppo_critic_value(features)
            if self.frozen is not None:
                learner = self.learner_rows()
                n = self.frozen_envs
                if self.config.published_opponent:
                    action = torch.where(
                        learner, action, self.frozen.actions(self.env, learner)
                    )
                else:
                    opponent_action, _ = sample_action(
                        self.frozen.logits(obs[:n], mask[:n])
                    )
                    action[:n] = torch.where(learner[:n], action[:n], opponent_action)
                # Frozen rows are excluded from PPO. Store a finite learner
                # log probability for the executed action on every row.
                logp = logits.log_softmax(-1).gather(1, action[:, None]).squeeze(-1)
                value = torch.where(learner, value, self.role_value(obs))
                buffers["learner"].append(learner.float())
            if self.critic:
                features = critic_input(obs, self.env.state)
                buffers["critic_obs"].append(features)
                buffers["q"].append(self.central_critic(features))
                buffers["probs"].append(logits.softmax(-1))
            result = self.env.step(action)
            reward = (
                result[:, 0].float()
                if self.config.reward == "match"
                else result[:, 3].float() / 32
            )
            reward = reward * (1 - 2 * player)
            for key, data in zip(
                buffers,
                (
                    obs,
                    mask,
                    target,
                    action,
                    logp,
                    value,
                    reward,
                    result[:, 2].bool(),
                    player,
                ),
            ):
                buffers[key].append(data)
            totals[:2] += result[:, 1:3].sum(0)
            if self.config.published_opponent:
                totals[2:] += result[: self.frozen_envs, 1:3].sum(0)
            self.env.reset_done()
        batch = {k: torch.stack(v) for k, v in buffers.items()}
        obs, mask = self.observe()
        _, next_value, _ = self.actor(obs, mask)
        if self.ppo_critic is not None:
            next_value = self.ppo_critic_value(
                ppo_value_input(obs, self.env.state, self.config.ppo_value)
            )
        if self.frozen is not None:
            next_value = torch.where(
                self.learner_rows(), next_value, self.role_value(obs)
            )
        if self.critic:
            next_q = self.central_critic(critic_input(obs, self.env.state))
            self.bootstrap = (obs, mask, next_q, self.env.state[:, PLAYER].clone())
            next_value = (self.actor(obs, mask)[0].softmax(-1) * next_q).sum(-1)
            expected = (batch["q"] * batch["probs"]).sum(-1)
            selected = batch["q"].gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
            advantages, returns = q_boost(
                batch["reward"],
                selected,
                expected,
                batch["done"],
                batch["player"],
                next_value,
                self.env.state[:, PLAYER],
                self.config.gae_lambda,
            )
        else:
            arguments = (
                batch["reward"],
                batch["value"],
                batch["done"],
                batch["player"],
                next_value,
                self.env.state[:, PLAYER],
                self.config.gae_lambda,
            )
            advantages, returns = (
                native_gae(self.env.backend, *arguments)
                if self.config.returns_backend == "native"
                else signed_gae(*arguments)
            )
        batch["advantage"], batch["return"] = advantages, returns
        return {k: v.flatten(0, 1) for k, v in batch.items()}, totals

    @torch.no_grad()
    def refresh_q_advantages(self, batch):
        """Recompute current-policy expectations with the frozen rollout critic."""
        values = []
        for start in range(0, len(batch["obs"]), self.config.minibatch):
            end = start + self.config.minibatch
            probs = self.actor(batch["obs"][start:end], batch["mask"][start:end])[
                0
            ].softmax(-1)
            values.append((probs * batch["q"][start:end]).sum(-1))
        shape = (self.config.horizon, self.config.envs)
        expected = torch.cat(values).reshape(shape)
        obs, mask, q, player = self.bootstrap
        next_value = (self.actor(obs, mask)[0].softmax(-1) * q).sum(-1)
        selected = (
            batch["q"].gather(-1, batch["action"][:, None]).squeeze(-1).reshape(shape)
        )
        advantage, _ = q_boost(
            batch["reward"].reshape(shape),
            selected,
            expected,
            batch["done"].reshape(shape),
            batch["player"].reshape(shape),
            next_value,
            player,
            self.config.gae_lambda,
        )
        advantage = advantage.flatten()
        batch["advantage"] = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )

    def minibatch_step(self, data):
        c = self.config
        obs, mask = data["obs"], data["mask"]
        logits, value, belief = self.actor(obs, mask)
        logprobs = logits.log_softmax(-1)
        logp = logprobs.gather(1, data["action"][:, None]).squeeze(-1)
        delta = logp - data["old_logp"]
        learner = data.get("learner")
        policy_weight = (
            mask[:, 48:].any(-1).float() if c.policy_scope == "koi" else learner
        )
        if policy_weight is not None:
            delta = torch.where(policy_weight.bool(), delta, 0.0)
        ratio = delta.exp()
        adv = data["advantage"]
        policy_terms = -torch.minimum(
            ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv
        )
        entropy_terms = -(logprobs.exp() * logprobs).sum(-1)
        if policy_weight is None:
            policy_loss = policy_terms.mean()
            entropy = entropy_terms.mean()
        else:
            count = policy_weight.sum().clamp_min(1)
            policy_loss = (policy_terms * policy_weight).sum() / count
            entropy = (entropy_terms * policy_weight).sum() / count
        if learner is not None:
            value = torch.where(learner.bool(), value, self.role_value(obs))
        value_loss = F.mse_loss(value, data["return"])
        unknown = obs[:, :528].reshape(-1, 48, 11)[:, :, 4]
        if learner is not None:
            unknown = unknown * learner[:, None]
        belief_loss = (
            F.binary_cross_entropy_with_logits(belief, data["target"], reduction="none")
            * unknown
        ).sum() / unknown.sum().clamp_min(1)
        loss = (
            policy_loss
            + c.value_weight * value_loss
            - c.entropy * entropy
            + c.belief_weight * belief_loss
        )
        if self.ppo_critic is not None:
            ppo_critic_loss = F.mse_loss(
                self.ppo_critic_value(data["ppo_value_obs"]), data["return"]
            )
            # Retain the actor's observation-only value head for deployment/search.
            # Each value loss has value_weight; the two critics share no parameters.
            loss = loss + c.value_weight * ppo_critic_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.trainable.parameters(), 0.5)
        self.optimizer.step()
        with torch.no_grad():
            kl_terms = (ratio - 1) - delta
            kl = (
                kl_terms.mean()
                if policy_weight is None
                else (kl_terms * policy_weight).sum() / policy_weight.sum().clamp_min(1)
            )
            metrics = [policy_loss, value_loss, entropy, belief_loss, kl, grad]
            if self.ppo_critic is not None:
                metrics.append(ppo_critic_loss)
            return torch.stack(metrics)

    def update(self):
        batch = self.collect()
        c = self.config
        advantage = batch["advantage"]
        learner = (
            batch["mask"][:, 48:].any(-1).float()
            if c.policy_scope == "koi"
            else batch.get("learner")
        )
        if learner is not None:
            count = learner.sum().clamp_min(1)
            mean = (advantage * learner).sum() / count
            variance = ((advantage - mean).square() * learner).sum() / count
            batch["advantage"] = torch.where(
                learner.bool(), (advantage - mean) / (variance.sqrt() + 1e-8), 0.0
            )
        else:
            batch["advantage"] = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )
        n = len(advantage)
        metrics = torch.zeros(6 + (self.ppo_critic is not None), device=c.device)
        count = 0
        for _ in range(c.epochs):
            if self.critic:
                self.refresh_q_advantages(batch)
            order = torch.randperm(n, device=c.device)
            for indices in order.split(c.minibatch):
                data = {
                    key: batch[key][indices]
                    for key in (
                        "obs",
                        "mask",
                        "action",
                        "old_logp",
                        "advantage",
                        "return",
                        "target",
                    )
                }
                if "learner" in batch:
                    data["learner"] = batch["learner"][indices]
                if self.ppo_critic is not None:
                    data["ppo_value_obs"] = batch["ppo_value_obs"][indices]
                if c.cuda_graph:
                    if self.captured_minibatch is None:
                        self.captured_minibatch = CapturedMinibatch(
                            self.minibatch_step, self.trainable, self.optimizer, data
                        )
                    metrics += self.captured_minibatch(data)
                else:
                    metrics += self.minibatch_step(data)
                count += 1
        critic_loss_mean = 0.0
        if self.critic:
            losses = []
            for _ in range(c.critic_epochs):
                for indices in torch.randperm(n, device=c.device).split(c.minibatch):
                    q = self.central_critic(batch["critic_obs"][indices])
                    selected = q.gather(1, batch["action"][indices, None]).squeeze(-1)
                    loss = F.mse_loss(selected, batch["return"][indices])
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
                    losses.append(loss.detach())
            critic_loss_mean = float(torch.stack(losses).mean())
        self.counters["updates"] += 1
        values = (metrics / count).detach().cpu().tolist()
        if self.ppo_critic is not None:
            critic_loss_mean = values.pop()
        if not all(np.isfinite(values + [critic_loss_mean])):
            raise FloatingPointError("Non-finite training metrics")
        return dict(
            zip(
                (
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "belief_loss",
                    "approx_kl",
                    "gradient_norm",
                    "critic_loss",
                ),
                values + [critic_loss_mean],
            )
        )

    def _check_round_continuation(self):
        if self.round_continuation is not None:
            self.round_continuation.check(self.config.round_continuation)
            if (
                self.provenance.get("round_continuation_sha256")
                != self.round_continuation.sha256
            ):
                raise ValueError("Round continuation checkpoint identity mismatch")
            if (
                self.provenance.get("initialized_from", {}).get("sha256")
                != self.round_continuation.parent_sha256
            ):
                raise ValueError(
                    "Round continuation requires initialization from its calibrated parent"
                )
        elif self.config.round_continuation is not None:
            raise ValueError("Round continuation cannot be enabled after construction")

    def save(self, path):
        self._check_round_continuation()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state,
            "counters": self.counters,
            "provenance": self.provenance,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        if self.opponent_value is not None:
            if self.frozen is None:
                raise ValueError("Cannot save without the configured frozen opponent")
            payload["opponent_value"] = self.opponent_value.state_dict()
            if self.config.published_opponent:
                payload["published_opponent"] = self.frozen.checkpoint()
            else:
                payload.update(
                    frozen_config=self.frozen.config,
                    frozen_model=self.frozen.model.state_dict(),
                )
        if self.critic:
            payload.update(
                critic=self.critic.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
            )
        if self.ppo_critic is not None:
            payload["ppo_critic"] = self.ppo_critic.state_dict()
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def restore(cls, path, allow_code_change=False):
        # Only load checkpoints produced locally by this project.
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        config = Config(**payload["config"])
        if config.published_opponent and "published_opponent" not in payload:
            raise ValueError("Published opponent checkpoint identity missing")
        obj = cls(
            config,
            published_snapshot=payload.get("published_opponent"),
        )
        if obj.provenance.get("round_continuation_sha256") != payload["provenance"].get(
            "round_continuation_sha256"
        ):
            raise ValueError("Round continuation checkpoint identity mismatch")
        if obj.config.published_opponent and (
            obj.provenance["frozen_opponent"]
            != payload["provenance"].get("frozen_opponent")
        ):
            raise ValueError("Published opponent checkpoint identity mismatch")
        obj.provenance["parent_checkpoint"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
            "source_sha256": payload["provenance"]["source_sha256"],
            "revision": payload["provenance"]["revision"],
        }
        for key in ("initialized_from", "execution_change", "frozen_opponent"):
            if key in payload["provenance"]:
                obj.provenance[key] = payload["provenance"][key]
        obj._check_round_continuation()
        if (
            payload["provenance"]["source_sha256"] != obj.provenance["source_sha256"]
            and not allow_code_change
        ):
            raise ValueError(
                "Source changed since checkpoint; use --allow-code-change for a labeled continuation"
            )
        obj.model.load_state_dict(payload["model"])
        if obj.ppo_critic is not None:
            obj.ppo_critic.load_state_dict(payload["ppo_critic"])
        if obj.opponent_value is not None:
            obj.opponent_value.load_state_dict(payload["opponent_value"])
            if not obj.config.published_opponent:
                obj._load_frozen(payload["frozen_config"], payload["frozen_model"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        if obj.critic:
            obj.critic.load_state_dict(payload["critic"])
            obj.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        obj.env.state.copy_(payload["env"])
        obj.counters = payload["counters"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path, default=Path("runs/ppo"))
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path)
    start.add_argument(
        "--initialize",
        type=Path,
        help="Load actor weights only; reset optimizer, deals, RNG and all counters",
    )
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument(
        "--frozen-opponent",
        type=Path,
        help="Fixed opponent for a new PPO experiment with opponent_fraction > 0",
    )
    parser.add_argument(
        "--enable-cuda-rollout",
        action="store_true",
        help="Explicitly enable rollout capture on resume, retaining all training state",
    )
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--target-matches", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="Stop cleanly after a completed update; zero disables the limit",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.updates < 0 or args.target_matches < 0:
        parser.error("Invalid update or checkpoint interval")
    if args.max_seconds < 0:
        parser.error("max-seconds must be nonnegative")
    if args.resume and args.frozen_opponent:
        parser.error("--resume restores its embedded frozen opponent")
    if args.resume and args.config:
        parser.error(
            "--resume restores its config; use --initialize for a new experiment"
        )
    if args.enable_cuda_rollout and not args.resume:
        parser.error("--enable-cuda-rollout requires --resume; new runs use config")
    if not args.resume and (args.run / "metrics.jsonl").exists():
        parser.error("Run already has metrics; use --resume or a new run directory")
    if args.resume:
        trainer = Trainer.restore(args.resume, args.allow_code_change)
        if args.enable_cuda_rollout:
            trainer.enable_rollout_capture()
    else:
        trainer = Trainer(
            Config(**json.loads(args.config.read_text(encoding="utf-8")))
            if args.config
            else Config()
        )
        if args.initialize:
            trainer.initialize_weights(args.initialize)
        if args.frozen_opponent:
            trainer.set_frozen_opponent(args.frozen_opponent)
        elif trainer.config.opponent_fraction and not trainer.config.published_opponent:
            parser.error("opponent_fraction requires --frozen-opponent for a new run")
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "config.json").write_text(
        json.dumps(asdict(trainer.config), indent=2) + "\n", encoding="utf-8"
    )
    (args.run / "provenance.json").write_text(
        json.dumps(trainer.provenance, indent=2) + "\n", encoding="utf-8"
    )
    start = time.perf_counter()
    initial = trainer.counters.copy()
    try:
        while (
            trainer.counters["matches"] < args.target_matches
            if args.target_matches
            else trainer.counters["updates"] < args.updates
        ):
            if (args.run / "STOP").exists() or (
                args.max_seconds and time.perf_counter() - start >= args.max_seconds
            ):
                print(
                    "Clean stop requested; saving the completed-update state.",
                    flush=True,
                )
                break
            metrics = trainer.update()
            elapsed = time.perf_counter() - start
            metrics.update(trainer.counters)
            metrics.update(
                elapsed_seconds=elapsed,
                decisions_per_second=(
                    trainer.counters["decisions"] - initial["decisions"]
                )
                / elapsed,
                matches_per_second=(trainer.counters["matches"] - initial["matches"])
                / elapsed,
            )
            line = json.dumps(metrics)
            print(line, flush=True)
            with (args.run / "metrics.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(line + "\n")
            if trainer.counters["updates"] % args.checkpoint_every == 0:
                trainer.save(args.run / "latest.pt")
                trainer.save(args.run / f"update_{trainer.counters['updates']:08d}.pt")
    except KeyboardInterrupt:
        print(
            "Interrupted; retaining the last completed-update checkpoint.", flush=True
        )
        return
    trainer.save(args.run / "latest.pt")


if __name__ == "__main__":
    main()
