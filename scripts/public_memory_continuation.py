"""Complete multi-round PPO continuations after public opponent identification.

The public stopping rule determines training roots. Private packed states are
used solely to restart the simulator; actor conditioning uses public codes.
Collection reaches terminal match rewards without cross-player bootstrapping.
This module is isolated from the frozen final-round comparison.
"""

import hashlib
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import (
    BANKRUPT,
    DISCARD,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    STATE_SIZE,
    TERMINAL,
    TURN,
)
from koikoi.model import sample_action
from koikoi.published_actor import PublishedActor
from koikoi.published_train import PublishedTrainer
from scripts.final_round_training import bank_identity
from scripts.final_round_training import source_identity as tail_source_identity
from scripts.fixed_microbatch_continuation import fixed_scores
from scripts.public_memory_tail import (
    MemoryTailConfig,
    MemoryTailTrainer,
    memory_identity,
    validate_memory,
)
from scripts.public_memory_tail import (
    source_identity as memory_source_identity,
)


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_continuation_bank(states, seats, opponents, memory):
    if not isinstance(states, torch.Tensor) or states.dtype != torch.int32:
        raise ValueError("Continuation states must be int32 tensors")
    n = len(states)
    if n < 1 or states.shape != (n, STATE_SIZE):
        raise ValueError("Expected nonempty continuation state matrix")
    for name, values in (("seats", seats), ("opponents", opponents)):
        if values.dtype != torch.int64 or values.shape != (n,):
            raise ValueError(f"Invalid continuation {name}")
        if not bool(((values == 0) | (values == 1)).all()):
            raise ValueError(f"Invalid continuation {name} labels")
    validate_memory(memory, n)
    if not bool((memory >= 0).all()):
        raise ValueError("Continuation requires public identification")
    expected = {ROUNDS: 8, BANKRUPT: 1, TURN: 0, PHASE: DISCARD}
    if any(not bool((states[:, k] == v).all()) for k, v in expected.items()):
        raise ValueError("Expected live paper8 fresh-round continuation roots")
    if not bool(((states[:, ROUND] >= 1) & (states[:, ROUND] <= 7)).all()):
        raise ValueError("Continuation roots must start in rounds2--8")
    if not bool((states[:, POINTS : POINTS + 2] > 0).all()):
        raise ValueError("Bankrupt continuation root")


@dataclass
class ContinuationConfig(MemoryTailConfig):
    horizon: int = 448

    def __post_init__(self):
        super().__post_init__()
        if self.horizon < 448:
            raise ValueError("Complete continuations require horizon >=448")


class ContinuationTrainer(MemoryTailTrainer):
    """Reuse conditioning/PPO, with separate multi-round collection and format."""

    def __init__(
        self,
        config,
        states,
        seats,
        opponents,
        memory,
        actor_snapshot=None,
        opponent_snapshots=None,
        *,
        initialize_memory=True,
    ):
        if not isinstance(config, ContinuationConfig):
            raise TypeError("Expected ContinuationConfig")
        validate_continuation_bank(states, seats, opponents, memory)
        # Bypass only the final-round-specific bank validation and constructor.
        # Core PPO and observation behavior remain inherited and source-bound.
        PublishedTrainer.__init__(self, config, actor_snapshot)
        self.bank = tuple(v.detach().cpu().clone() for v in (states, seats, opponents))
        self.memory = memory.detach().cpu().clone()
        self.bank_rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        self.opponents = {
            name: (
                PublishedActor.from_published(name, config.device)
                if opponent_snapshots is None
                else PublishedActor(opponent_snapshots[name], config.device)
            )
            .requires_grad_(False)
            .eval()
            for name in ("wp", "point")
        }
        if any(a.initialization["variant"] != n for n, a in self.opponents.items()):
            raise ValueError("Opponent variant mismatch")
        self.prefix_actor = None
        self.provenance.update(
            {
                "tail_source_sha256": tail_source_identity(),
                "public_memory_source_sha256": memory_source_identity(),
                "continuation_source_sha256": source_identity(),
                "root_bank_sha256": bank_identity(*self.bank),
                "public_memory_bank_sha256": memory_identity(self.memory),
                "tail_opponents": {
                    n: a.checkpoint()["current"]["parameters_sha256"]
                    for n, a in self.opponents.items()
                },
            }
        )
        self.counters.update(
            learner_decisions=0,
            opponent_decisions=0,
            conditional_tail_completions=0,
            conditional_round_endings=0,
        )
        if initialize_memory:
            with torch.no_grad():
                for model in self.model.actor.models.values():
                    model.encoder_block.f1.weight[:, 2:4].zero_()

    def _check_runtime(self):
        super()._check_runtime()
        if self.provenance["continuation_source_sha256"] != source_identity():
            raise ValueError("Continuation implementation changed")

    @torch.no_grad()
    def collect(self):
        self._check_runtime()
        indices = torch.randint(
            len(self.bank[0]), (self.config.envs,), generator=self.bank_rng
        )
        states, seats, labels = (v[indices].to(self.env.device) for v in self.bank)
        self.env.state.copy_(states)
        self.last_root_indices = indices.clone()
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
        return batch

    def save(self, path):
        self._check_runtime()
        if self.prefix_actor is None:
            raise ValueError("A frozen prefix parent is required")
        payload = {
            "format": "public_memory_continuation_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "prefix_actor": self.prefix_actor,
            "value": self.model.value.state_dict(),
            "memory": self.memory,
            "bank": self.bank,
            "bank_rng": self.bank_rng.get_state(),
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
        if payload.get("format") != "public_memory_continuation_v1":
            raise ValueError("Expected public-memory continuation checkpoint")
        obj = cls(
            ContinuationConfig(**payload["config"]),
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
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Continuation source/bank/opponent mismatch")
        obj.provenance = payload["provenance"]
        obj.prefix_actor = payload["prefix_actor"]
        obj.model.value.load_state_dict(payload["value"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.counters = payload["counters"]
        obj.bank_rng.set_state(payload["bank_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        return obj
