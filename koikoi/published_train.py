"""Explicit pretrained-score PPO using the existing signed-GAE/PPO machinery."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import BatchEnv
from .layout import PHASE
from .published_actor import PublishedActor
from .published_opponent import runtime_identity
from .train import Config, Trainer, provenance


@dataclass
class PublishedConfig(Config):
    kind: str = "published"
    profile: str = "paper8"
    envs: int = 128
    horizon: int = 128
    minibatch: int = 512
    lr: float = 0.000003
    cpu_threads: int = 2
    endgame: bool = True
    initial_variant: str = "wp"
    score_temperature: float = 0.05
    value_lr: float = 0.0001
    critic_warmup_updates: int = 20

    def __post_init__(self):
        super().__post_init__()
        if (
            self.kind != "published"
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
        ):
            raise ValueError(
                "Published PPO requires eager float32 pure-self-play paper8 match learning"
            )
        if self.initial_variant not in ("wp", "point"):
            raise ValueError("Published initial_variant must be wp or point")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.score_temperature, self.lr, self.value_lr)
        ):
            raise ValueError(
                "Published temperature and learning rates must be positive and finite"
            )
        if (
            type(self.critic_warmup_updates) is not int
            or self.critic_warmup_updates < 0
        ):
            raise ValueError("critic_warmup_updates must be a nonnegative integer")
        if any(
            type(x) is not int or x < 1
            for x in (
                self.envs,
                self.horizon,
                self.minibatch,
                self.epochs,
                self.critic_width,
                self.cpu_threads,
            )
        ):
            raise ValueError(
                "Published PPO dimensions and counts must be positive integers"
            )


def packed_observation(env, labels=False):
    """Pack only native public observations; never give the actor a raw state."""
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Published PPO requires default float32 observations")
    ordinary = env.observe(labels=labels)
    features, legal = env.published_observe()
    obs = torch.cat(
        [ordinary[0], env.state[:, PHASE : PHASE + 1].float(), features.flatten(1)],
        dim=1,
    )
    return (obs, legal, ordinary[2]) if labels else (obs, legal)


class PublishedPolicyValue(nn.Module):
    def __init__(self, config, snapshot=None, device=None):
        super().__init__()
        device = config.device if device is None else device
        self.actor = (
            PublishedActor.from_published(config.initial_variant, device)
            if snapshot is None
            else PublishedActor(snapshot, device)
        )
        if self.actor.initialization["variant"] != config.initial_variant:
            raise ValueError("Published actor variant/config mismatch")
        self.temperature = config.score_temperature
        self.value = nn.Sequential(
            nn.Linear(560, config.critic_width),
            nn.Tanh(),
            nn.Linear(config.critic_width, config.critic_width),
            nn.Tanh(),
            nn.Linear(config.critic_width, 1),
        ).to(device=device, dtype=torch.float32)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, obs, mask):
        if obs.ndim != 2 or obs.shape[1] != 15561:
            raise ValueError(
                "Published PPO requires packed public observations with 15561 columns"
            )
        scores = self.actor(obs[:, 561:].reshape(-1, 300, 50), obs[:, 560].int(), mask)
        logits = (scores / self.temperature).masked_fill(~mask, -1e9)
        value = self.value(obs[:, :560]).squeeze(-1)
        return logits, value, obs.new_zeros((len(obs), 48))


class PublishedTrainer(Trainer):
    """Reuse Trainer.collect_device/minibatch_step/update; specialize inputs/state."""

    def __init__(self, config: PublishedConfig, actor_snapshot=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Published PPO requires default float32")
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
        self.model = PublishedPolicyValue(config, actor_snapshot)
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
        self.counters = {
            key: 0
            for key in (
                "updates",
                "decisions",
                "rounds",
                "matches",
                "policy_updates",
                "warmup_updates",
                "warmup_decisions",
                "warmup_rounds",
                "warmup_matches",
                "optimizer_steps_value",
                "optimizer_steps_discard",
                "optimizer_steps_pick",
                "optimizer_steps_koi",
            )
        }
        self.provenance = provenance()
        self.provenance["published_initialization"] = copy.deepcopy(
            self.model.actor.initialization
        )
        self.runtime = runtime_identity(self.env.state.device)

    def _check_runtime(self):
        if json.dumps(asdict(self.config), sort_keys=True) != self._config_json:
            raise ValueError("Published PPO config changed during training")
        if runtime_identity(self.env.state.device) != self.runtime:
            raise ValueError("Published PPO runtime changed")
        if (
            self.provenance["published_initialization"]
            != self.model.actor.initialization
        ):
            raise ValueError("Published PPO initialization identity mismatch")
        if self.model.temperature != self.config.score_temperature:
            raise ValueError("Published PPO score temperature changed")

    def observe(self, labels=False):
        result = packed_observation(self.env, labels)
        if self.config.endgame:
            solved = self.env.solved_actions()
            result[1].logical_and_(
                (solved[:, None] < 0)
                | (torch.arange(50, device=self.env.device)[None] == solved[:, None])
            )
        return result

    def collect(self):
        self._check_runtime()
        return super().collect()

    def update(self):
        warmup = self.counters["updates"] < self.config.critic_warmup_updates
        self.model.actor.requires_grad_(not warmup)
        before = self.counters.copy()
        metrics = super().update()
        self.counters["warmup_updates" if warmup else "policy_updates"] += 1
        if warmup:
            for key in ("decisions", "rounds", "matches"):
                self.counters["warmup_" + key] += self.counters[key] - before[key]
        metrics["critic_only"] = warmup
        return metrics

    def minibatch_step(self, data):
        metrics = super().minibatch_step(data)
        self.counters["optimizer_steps_value"] += 1
        if self.counters["updates"] >= self.config.critic_warmup_updates:
            phases = data["obs"][:, 560]
            present = (
                torch.stack(
                    [
                        (phases == 0).any(),
                        ((phases == 1) | (phases == 2)).any(),
                        (phases == 3).any(),
                    ]
                )
                .cpu()
                .tolist()
            )
            for name, active in zip(("discard", "pick", "koi"), present):
                self.counters["optimizer_steps_" + name] += int(active)
        return metrics

    def initialize_weights(self, path):
        raise ValueError(
            "Published PPO initializes from pinned weights; use its embedded checkpoint to resume"
        )

    def save(self, path):
        self._check_runtime()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "published_ppo_v1",
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
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path, allow_code_change=False):
        content = Path(path).read_bytes()
        import io

        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload["format"] != "published_ppo_v1":
            raise ValueError("Unsupported published PPO checkpoint")
        obj = cls(PublishedConfig(**payload["config"]), payload["actor"])
        if obj.runtime != payload["runtime"]:
            raise ValueError("Published PPO runtime mismatch; exact resume rejected")
        if (
            obj.provenance["published_initialization"]
            != payload["provenance"]["published_initialization"]
        ):
            raise ValueError("Published PPO initialization identity mismatch")
        if (
            not allow_code_change
            and obj.provenance["source_sha256"]
            != payload["provenance"]["source_sha256"]
        ):
            raise ValueError(
                "Source changed since checkpoint; use --allow-code-change for a labeled continuation"
            )
        obj.provenance["parent_checkpoint"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "counters": payload["counters"].copy(),
            "source_sha256": payload["provenance"]["source_sha256"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    if args.updates < 0 or args.checkpoint_every < 1 or (args.resume and args.config):
        parser.error("Invalid update/checkpoint options or resume with config")
    if not args.resume and (args.run / "metrics.jsonl").exists():
        parser.error("Run already has metrics; use --resume or a new directory")
    trainer = (
        PublishedTrainer.restore(args.resume, args.allow_code_change)
        if args.resume
        else PublishedTrainer(
            PublishedConfig(**json.loads(args.config.read_text(encoding="utf-8")))
            if args.config
            else PublishedConfig()
        )
    )
    args.run.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("config", asdict(trainer.config)),
        ("provenance", trainer.provenance),
    ):
        (args.run / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    start = time.perf_counter()
    try:
        while trainer.counters["updates"] < args.updates:
            if (args.run / "STOP").exists():
                break
            metrics = (
                trainer.update()
                | trainer.counters
                | {"elapsed_seconds": time.perf_counter() - start}
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
