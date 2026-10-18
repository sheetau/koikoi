"""Complete terminal-return PPO using renewed deals at public stopping contexts.

This separate format/module preserves the frozen fixed-root study. The collection
loop retains its terminal-return semantics;only root/deal sampling differs.
"""

import hashlib
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER, ROUND, TERMINAL
from koikoi.model import sample_action
from scripts.continuation_chance import integer
from scripts.fixed_microbatch_continuation import fixed_scores
from scripts.public_memory_continuation import ContinuationConfig, ContinuationTrainer
from scripts.renewed_root_sampler import RenewedRootSampler


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class RenewedContinuationConfig(ContinuationConfig):
    chance_seed: int | None = None
    chance_draws: int | None = None

    def __post_init__(self):
        super().__post_init__()
        start = integer(self.chance_seed, "chance_seed")
        count = integer(self.chance_draws, "chance_draws")
        if start < 0 or count < self.envs or count % self.envs or start + count > 2**31:
            raise ValueError("Expected a finite chance allocation for complete batches")


class RenewedContinuationTrainer(ContinuationTrainer):
    def __init__(self, config, *args, **kwargs):
        if not isinstance(config, RenewedContinuationConfig):
            raise TypeError("Expected RenewedContinuationConfig")
        super().__init__(config, *args, **kwargs)
        self.root_sampler = RenewedRootSampler(
            *self.bank,
            self.memory,
            sampling_seed=config.seed + 1,
            chance_seed=config.chance_seed,
            max_draws=config.chance_draws,
        )
        del self.bank_rng
        self.counters["fresh_conditional_deals"] = 0
        self.provenance.update(
            {
                "renewed_continuation_source_sha256": source_identity(),
                "root_sampler_source_sha256": self.root_sampler.identity[
                    "sampler_source"
                ],
                "chance_constructor_source_sha256": self.root_sampler.identity[
                    "deal_source"
                ],
            }
        )

    def _check_runtime(self):
        super()._check_runtime()
        expected_sources = {
            "renewed_continuation_source_sha256": source_identity(),
            "root_sampler_source_sha256": hashlib.sha256(
                Path("scripts/renewed_root_sampler.py").read_bytes()
            ).hexdigest(),
            "chance_constructor_source_sha256": hashlib.sha256(
                Path("scripts/continuation_chance.py").read_bytes()
            ).hexdigest(),
        }
        if any(self.provenance[k] != value for k, value in expected_sources.items()):
            raise ValueError("Renewed continuation source changed")
        if self.root_sampler.config != {
            "sampling_seed": self.config.seed + 1,
            "chance_seed": self.config.chance_seed,
            "max_draws": self.config.chance_draws,
        }:
            raise ValueError("Chance allocation changed")
        if not (
            self.root_sampler.cursor
            == self.counters["fresh_conditional_deals"]
            == self.counters["conditional_tail_completions"]
        ):
            raise ValueError("Chance sampler and continuation counters disagree")
        if not (
            torch.equal(self.root_sampler.memory, self.memory)
            and torch.equal(self.root_sampler.seats, self.bank[1])
            and torch.equal(self.root_sampler.opponents, self.bank[2])
        ):
            raise ValueError("Chance sampler metadata changed")

    @torch.no_grad()
    def collect(self):
        self._check_runtime()
        roots = self.root_sampler.sample(self.config.envs)
        states, seats, labels = (
            roots[k].to(self.env.device) for k in ("states", "seats", "opponents")
        )
        self.env.state.copy_(states)
        self.last_root_indices = roots["indices"].clone()
        self.last_chance_seeds = roots["chance_seeds"].clone()
        buffers = {
            k: [] for k in ("obs", "mask", "action", "old_logp", "value", "lane")
        }
        rewards = torch.zeros(self.env.count, device=self.env.device)
        completed = torch.zeros(
            self.env.count, dtype=torch.bool, device=self.env.device
        )
        lane_round_ends = torch.zeros(
            self.env.count, dtype=torch.int64, device=self.env.device
        )
        decisions = own_decisions = round_ends = 0
        for _ in range(self.config.horizon):
            active = self.env.state[:, PHASE] != TERMINAL
            if not bool(active.any()):
                break
            own = active & (self.env.state[:, PLAYER] == seats)
            obs, mask = self.observe()
            actions = torch.full(
                (self.env.count,), -1, dtype=torch.int64, device=self.env.device
            )
            lanes = torch.where(own)[0]
            if len(lanes):
                logits, values, _ = self.actor(obs[lanes], mask[lanes])
                selected, logp = sample_action(logits)
                actions[lanes] = selected
                for key, value in zip(
                    buffers, (obs[lanes], mask[lanes], selected, logp, values, lanes)
                ):
                    buffers[key].append(value)
            for label, actor in enumerate(self.opponents.values()):
                rows = torch.where(active & ~own & (labels == label))[0]
                if len(rows):
                    actions[rows] = fixed_scores(
                        actor,
                        obs[rows, 561:].reshape(-1, 300, 50),
                        obs[rows, 560].int(),
                        mask[rows],
                        64,
                    ).argmax(-1)
            result = self.env.step(actions, freeze_done=True)
            if bool((result[:, 0] == 99).any()):
                raise AssertionError("Illegal continuation action")
            ended = result[:, 2].bool()
            if bool((completed & ended).any()):
                raise AssertionError("Duplicate terminal event")
            rewards += result[:, 0].float() * (1 - 2 * seats)
            completed |= ended
            decisions += int(active.sum())
            own_decisions += len(lanes)
            lane_round_ends += result[:, 1].long()
            round_ends += int(result[:, 1].sum())
        if not bool(completed.all()) or not bool(
            (self.env.state[:, PHASE] == TERMINAL).all()
        ):
            raise RuntimeError(
                "Continuation horizon exhausted; refusing truncated training"
            )
        expected_rounds = self.env.state[:, ROUND].long() - states[:, ROUND].long() + 1
        if not torch.equal(lane_round_ends, expected_rounds):
            raise AssertionError(
                "Continuation round endings disagree with terminal rounds"
            )
        self.last_round_endings = lane_round_ends.clone()
        if not buffers["obs"]:
            raise RuntimeError("No learner decisions in this complete-tail batch")
        batch = {k: torch.cat(v) for k, v in buffers.items()}
        batch["return"] = rewards[batch["lane"]]
        batch["advantage"] = batch["return"] - batch["value"]
        batch["target"] = batch["obs"].new_zeros((len(batch["obs"]), 48))
        self.last_terminal_rewards = rewards.clone()
        self.counters["decisions"] += decisions
        self.counters["learner_decisions"] += own_decisions
        self.counters["opponent_decisions"] += decisions - own_decisions
        self.counters["conditional_tail_completions"] += self.env.count
        self.counters["conditional_round_endings"] += round_ends
        # Ordinary matches/rounds stay zero: these trajectories start at stored roots.
        self.counters["fresh_conditional_deals"] += self.env.count
        return batch

    def save(self, path):
        self._check_runtime()
        if self.prefix_actor is None:
            raise ValueError("A frozen prefix parent is required")
        payload = {
            "format": "renewed_public_continuation_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "prefix_actor": self.prefix_actor,
            "value": self.model.value.state_dict(),
            "memory": self.memory,
            "bank": self.bank,
            "root_sampler": self.root_sampler.state_dict(),
            "opponents": {n: a.checkpoint() for n, a in self.opponents.items()},
            "optimizer": self.optimizer.state_dict(),
            "counters": self.counters,
            "provenance": self.provenance,
            "runtime": self.runtime,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "renewed_public_continuation_v1":
            raise ValueError("Expected public-memory continuation checkpoint")
        obj = cls(
            RenewedContinuationConfig(**payload["config"]),
            *payload["bank"],
            payload["memory"],
            payload["actor"],
            payload["opponents"],
            initialize_memory=False,
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Continuation runtime mismatch")
        for key in (
            "source_sha256",
            "published_initialization",
            "tail_source_sha256",
            "root_bank_sha256",
            "tail_opponents",
            "public_memory_source_sha256",
            "public_memory_bank_sha256",
            "continuation_source_sha256",
            "renewed_continuation_source_sha256",
            "root_sampler_source_sha256",
            "chance_constructor_source_sha256",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Continuation source/bank/opponent mismatch")
        obj.provenance = payload["provenance"]
        obj.prefix_actor = payload["prefix_actor"]
        obj.model.value.load_state_dict(payload["value"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.counters = payload["counters"]
        obj.root_sampler.load_state_dict(payload["root_sampler"])
        obj._check_runtime()
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        return obj
