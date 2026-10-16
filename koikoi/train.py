"""Restartable current-policy self-play PPO; returns span complete matches."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .env import BatchEnv
from .estimators import CentralQ, critic_input, native_gae, q_boost
from .layout import PLAYER
from .model import PolicyValue, sample_action


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

    def __post_init__(self):
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
            )
            < 1
        ):
            raise ValueError("Training sizes must be positive")
        if not 0 <= self.gae_lambda <= 1 or self.belief_weight < 0:
            raise ValueError("Invalid loss/trace configuration")


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
    def __init__(self, config: Config):
        self.config = config
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.env = BatchEnv(config.envs, config.device, config.profile, config.seed)
        self.model = PolicyValue(config.kind, config.width, config.layers).to(
            config.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, eps=1e-5
        )
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
        self.provenance = provenance()

    def initialize_weights(self, path):
        """Start a new experiment from actor weights, with fresh state and counters."""
        if any(self.counters.values()) or self.optimizer.state:
            raise ValueError("Weight initialization requires a fresh trainer")
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        for field in ("kind", "width", "layers"):
            if payload["config"][field] != getattr(self.config, field):
                raise ValueError(f"Initialization architecture mismatch: {field}")
        self.model.load_state_dict(payload["model"])
        self.provenance["initialized_from"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "profile": payload["config"]["profile"],
            "counters": payload["counters"].copy(),
            "provenance": payload["provenance"],
        }

    def actor(self, obs, mask):
        with torch.autocast(
            device_type=self.env.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
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

    @torch.no_grad()
    def collect(self):
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
        if self.critic:
            buffers.update({"critic_obs": [], "q": [], "probs": []})
        totals = torch.zeros(2, dtype=torch.int64, device=self.config.device)
        for _ in range(self.config.horizon):
            obs, mask, target = self.env.observe(labels=True)
            player = self.env.state[:, PLAYER].clone()
            logits, value, _ = self.actor(obs, mask)
            action, logp = sample_action(logits)
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
            totals += result[:, 1:3].sum(0)
            self.env.reset_done()
        batch = {k: torch.stack(v) for k, v in buffers.items()}
        obs, mask = self.env.observe()
        _, next_value, _ = self.actor(obs, mask)
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
        rounds, matches = totals.cpu().tolist()
        self.counters["rounds"] += rounds
        self.counters["matches"] += matches
        self.counters["decisions"] += self.config.envs * self.config.horizon
        return {k: v.flatten(0, 1) for k, v in batch.items()}

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

    def update(self):
        batch = self.collect()
        c = self.config
        advantage = batch["advantage"]
        batch["advantage"] = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )
        n = len(advantage)
        metrics = torch.zeros(6, device=c.device)
        count = 0
        for _ in range(c.epochs):
            if self.critic:
                self.refresh_q_advantages(batch)
            order = torch.randperm(n, device=c.device)
            for indices in order.split(c.minibatch):
                obs, mask = batch["obs"][indices], batch["mask"][indices]
                logits, value, belief = self.actor(obs, mask)
                logprobs = logits.log_softmax(-1)
                logp = logprobs.gather(1, batch["action"][indices, None]).squeeze(-1)
                ratio = (logp - batch["old_logp"][indices]).exp()
                adv = batch["advantage"][indices]
                policy_loss = -torch.minimum(
                    ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv
                ).mean()
                value_loss = F.mse_loss(value, batch["return"][indices])
                entropy = -(logprobs.exp() * logprobs).sum(-1).mean()
                unknown = obs[:, :528].reshape(-1, 48, 11)[:, :, 4]
                belief_loss = (
                    F.binary_cross_entropy_with_logits(
                        belief, batch["target"][indices], reduction="none"
                    )
                    * unknown
                ).sum() / unknown.sum().clamp_min(1)
                loss = (
                    policy_loss
                    + c.value_weight * value_loss
                    - c.entropy * entropy
                    + c.belief_weight * belief_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - batch["old_logp"][indices])).mean()
                    metrics += torch.stack(
                        [policy_loss, value_loss, entropy, belief_loss, kl, grad]
                    )
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

    def save(self, path):
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
        if self.critic:
            payload.update(
                critic=self.critic.state_dict(),
                critic_optimizer=self.critic_optimizer.state_dict(),
            )
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
        obj = cls(Config(**payload["config"]))
        obj.provenance["parent_checkpoint"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
            "source_sha256": payload["provenance"]["source_sha256"],
            "revision": payload["provenance"]["revision"],
        }
        if "initialized_from" in payload["provenance"]:
            obj.provenance["initialized_from"] = payload["provenance"][
                "initialized_from"
            ]
        if (
            payload["provenance"]["source_sha256"] != obj.provenance["source_sha256"]
            and not allow_code_change
        ):
            raise ValueError(
                "Source changed since checkpoint; use --allow-code-change for a labeled continuation"
            )
        obj.model.load_state_dict(payload["model"])
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
    if args.resume and args.config:
        parser.error(
            "--resume restores its config; use --initialize for a new experiment"
        )
    if not args.resume and (args.run / "metrics.jsonl").exists():
        parser.error("Run already has metrics; use --resume or a new run directory")
    if args.resume:
        trainer = Trainer.restore(args.resume, args.allow_code_change)
    else:
        trainer = Trainer(
            Config(**json.loads(args.config.read_text(encoding="utf-8")))
            if args.config
            else Config()
        )
        if args.initialize:
            trainer.initialize_weights(args.initialize)
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
