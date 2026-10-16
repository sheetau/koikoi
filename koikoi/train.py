"""Restartable current-policy self-play PPO; returns span complete matches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .env import BatchEnv
from .layout import PLAYER
from .model import PolicyValue, sample_action


@dataclass
class Config:
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
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unknown"
    return {"revision": revision, "source_sha256": digest.hexdigest(), "torch": torch.__version__,
            "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        torch.set_num_threads(config.cpu_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.env = BatchEnv(config.envs, config.device, config.profile, config.seed)
        self.model = PolicyValue(config.kind, config.width, config.layers).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.counters = {"updates": 0, "decisions": 0, "rounds": 0, "matches": 0}
        self.provenance = provenance()

    @torch.no_grad()
    def collect(self):
        buffers = {k: [] for k in ("obs", "mask", "target", "action", "old_logp", "value", "reward", "done", "player")}
        totals = torch.zeros(2, dtype=torch.int64, device=self.config.device)
        for _ in range(self.config.horizon):
            obs, mask, target = self.env.observe(labels=True)
            player = self.env.state[:, PLAYER].clone()
            logits, value, _ = self.model(obs, mask)
            action, logp = sample_action(logits)
            result = self.env.step(action)
            reward = result[:, 0].float() if self.config.reward == "match" else result[:, 3].float()/32
            reward = reward * (1 - 2*player)
            for key, data in zip(buffers, (obs, mask, target, action, logp, value, reward, result[:, 2].bool(), player)):
                buffers[key].append(data)
            totals += result[:, 1:3].sum(0)
            self.env.reset_done()
        batch = {k: torch.stack(v) for k, v in buffers.items()}
        obs, mask = self.env.observe()
        _, next_value, _ = self.model(obs, mask)
        advantages, returns = signed_gae(batch["reward"], batch["value"], batch["done"], batch["player"],
                                          next_value, self.env.state[:, PLAYER], self.config.gae_lambda)
        batch["advantage"], batch["return"] = advantages, returns
        rounds, matches = totals.cpu().tolist()
        self.counters["rounds"] += rounds
        self.counters["matches"] += matches
        self.counters["decisions"] += self.config.envs * self.config.horizon
        return {k: v.flatten(0, 1) for k, v in batch.items()}

    def update(self):
        batch = self.collect()
        c = self.config
        advantage = batch["advantage"]
        batch["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        n = len(advantage)
        metrics = torch.zeros(6, device=c.device)
        count = 0
        for _ in range(c.epochs):
            order = torch.randperm(n, device=c.device)
            for indices in order.split(c.minibatch):
                obs, mask = batch["obs"][indices], batch["mask"][indices]
                logits, value, belief = self.model(obs, mask)
                logprobs = logits.log_softmax(-1)
                logp = logprobs.gather(1, batch["action"][indices, None]).squeeze(-1)
                ratio = (logp - batch["old_logp"][indices]).exp()
                adv = batch["advantage"][indices]
                policy_loss = -torch.minimum(ratio*adv, ratio.clamp(1-c.clip, 1+c.clip)*adv).mean()
                value_loss = F.mse_loss(value, batch["return"][indices])
                entropy = -(logprobs.exp()*logprobs).sum(-1).mean()
                unknown = obs[:, :528].reshape(-1, 48, 11)[:, :, 4]
                belief_loss = (F.binary_cross_entropy_with_logits(belief, batch["target"][indices], reduction="none") * unknown).sum() / unknown.sum().clamp_min(1)
                loss = policy_loss + c.value_weight*value_loss - c.entropy*entropy + c.belief_weight*belief_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio-1) - (logp-batch["old_logp"][indices])).mean()
                    metrics += torch.stack([policy_loss, value_loss, entropy, belief_loss, kl, grad])
                count += 1
        self.counters["updates"] += 1
        values = (metrics/count).detach().cpu().tolist()
        if not all(np.isfinite(values)):
            raise FloatingPointError("Non-finite training metrics")
        return dict(zip(("policy_loss", "value_loss", "entropy", "belief_loss", "approx_kl", "gradient_norm"), values))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": 1, "config": asdict(self.config), "model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "env": self.env.state,
                   "counters": self.counters, "provenance": self.provenance,
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def restore(cls, path, allow_code_change=False):
        # Only load checkpoints produced locally by this project.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        obj = cls(Config(**payload["config"]))
        if payload["provenance"]["source_sha256"] != obj.provenance["source_sha256"] and not allow_code_change:
            raise ValueError("Source changed since checkpoint; use --allow-code-change for a labeled continuation")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--target-matches", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    if args.resume:
        trainer = Trainer.restore(args.resume, args.allow_code_change)
    else:
        trainer = Trainer(Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config())
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "config.json").write_text(json.dumps(asdict(trainer.config), indent=2)+"\n", encoding="utf-8")
    (args.run / "provenance.json").write_text(json.dumps(trainer.provenance, indent=2)+"\n", encoding="utf-8")
    start = time.perf_counter()
    initial = trainer.counters.copy()
    try:
        while (trainer.counters["matches"] < args.target_matches if args.target_matches else trainer.counters["updates"] < args.updates):
            metrics = trainer.update()
            elapsed = time.perf_counter()-start
            metrics.update(trainer.counters)
            metrics.update(elapsed_seconds=elapsed,
                           decisions_per_second=(trainer.counters["decisions"]-initial["decisions"])/elapsed,
                           matches_per_second=(trainer.counters["matches"]-initial["matches"])/elapsed)
            line = json.dumps(metrics)
            print(line, flush=True)
            with (args.run / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line+"\n")
            if trainer.counters["updates"] % args.checkpoint_every == 0:
                trainer.save(args.run / "latest.pt")
                trainer.save(args.run / f"update_{trainer.counters['updates']:08d}.pt")
    except KeyboardInterrupt:
        print("Interrupted; retaining the last completed-update checkpoint.", flush=True)
        return
    trainer.save(args.run / "latest.pt")


if __name__ == "__main__":
    main()
