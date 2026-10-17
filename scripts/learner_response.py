"""Isolated fixed-opponent PPO using returns between learner decisions.

Primitive simulator budgets and public actor inputs stay explicit. No opponent
critic is fitted. This is a capability, not evidence of stronger Koi-Koi play.
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

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PLAYER
from koikoi.model import sample_action
from koikoi.published_train import PublishedPolicyValue, PublishedTrainer
from scripts.learner_returns import learner_gae
from scripts.published_response import PublicMixture, ResponseConfig


def source_identities():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "learner_response.py",
            "learner_returns.py",
            "published_response.py",
        )
    }


@dataclass
class LearnerConfig(ResponseConfig):
    returns_backend: str = "torch"

    def __post_init__(self):
        super().__post_init__()
        if self.opponent_fraction != 1 or self.returns_backend != "torch":
            raise ValueError(
                "Learner-time returns require all fixed-opponent lanes and torch"
            )


class LearnerTrainer(PublishedTrainer):
    def __init__(self, config, actor_snapshot=None, opponent_snapshots=None):
        if not isinstance(config, LearnerConfig):
            raise TypeError("LearnerTrainer requires LearnerConfig")
        super().__init__(config, actor_snapshot)
        self.rows = torch.arange(config.envs, device=config.device)
        self.frozen_envs = config.envs
        self.frozen = PublicMixture(config, opponent_snapshots)
        self.counters.update(
            learner_decisions=0,
            valid_learner_decisions=0,
            dropped_learner_tails=0,
            published_decisions=0,
            published_rounds=0,
            published_matches=0,
            skipped_empty_minibatches=0,
        )
        self.opponent_work = {
            name: {"decisions": 0, "rounds": 0, "matches": 0}
            for name in self.frozen.agents
        }
        self.provenance["learner_sources"] = source_identities()
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
            raise ValueError("Learner initialization requires a published PPO parent")
        if any(
            payload["config"][k] != getattr(config, k) for k in ("profile", "endgame")
        ):
            raise ValueError("Learner parent rules/assistance mismatch")
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Learner parent initialization identity mismatch")
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
        if self.provenance["learner_sources"] != source_identities():
            raise ValueError("Learner-return implementation changed during training")

    @torch.no_grad()
    def collect(self):
        self._check_runtime()
        self.frozen.check_runtime()
        buffers = {
            name: []
            for name in (
                "obs",
                "mask",
                "target",
                "action",
                "old_logp",
                "value",
                "reward",
                "done",
                "player",
                "is_learner",
            )
        }
        counts = torch.zeros(
            (len(self.frozen.agents), 3), dtype=torch.int64, device=self.env.device
        )
        for _ in range(self.config.horizon):
            obs, mask, target = self.observe(labels=True)
            player = self.env.state[:, PLAYER].clone()
            own = self.learner_rows()
            ids = self.frozen.ids(self.env)
            logits, value, _ = self.actor(obs, mask)
            proposed, _ = sample_action(logits)
            action = torch.where(own, proposed, self.frozen.actions(self.env, own))
            logp = logits.log_softmax(-1).gather(1, action[:, None]).squeeze(-1)
            result = self.env.step(action)
            # The engine returns seat-zero reward. Convert with PRE-STEP roles;
            # an opponent terminal must keep the old episode's learner seat.
            acting_reward = result[:, 0].float() * (1 - 2 * player)
            reward = torch.where(own, acting_reward, -acting_reward)
            data = (
                obs,
                mask,
                target,
                action,
                logp,
                value,
                reward,
                result[:, 2].bool(),
                player,
                own,
            )
            for name, tensor in zip(buffers, data):
                buffers[name].append(tensor)
            for index in range(len(self.frozen.agents)):
                lanes = ids == index
                counts[index, 0] += ((~own) & lanes).sum()
                counts[index, 1:] += (result[:, 1:3] * lanes[:, None]).sum(0)
            self.env.reset_done()
        batch = {name: torch.stack(tensors) for name, tensors in buffers.items()}
        obs, mask = self.observe()
        next_value = self.actor(obs, mask)[1]
        adv, targets, valid = learner_gae(
            batch["reward"],
            batch["value"],
            batch["done"],
            batch["is_learner"],
            next_value,
            self.learner_rows(),
            self.config.gae_lambda,
        )
        batch.update(advantage=adv, **{"return": targets, "learner": valid.float()})
        work = counts.cpu().tolist()
        for name, row in zip(self.frozen.agents, work):
            for key, count in zip(("decisions", "rounds", "matches"), row):
                self.opponent_work[name][key] += count
        opponent_decisions, rounds, matches = [
            sum(row[i] for row in work) for i in range(3)
        ]
        own_count = int(batch["is_learner"].sum())
        valid_count = int(valid.sum())
        additions = {
            "decisions": self.config.envs * self.config.horizon,
            "rounds": rounds,
            "matches": matches,
            "learner_decisions": own_count,
            "valid_learner_decisions": valid_count,
            "dropped_learner_tails": own_count - valid_count,
            "published_decisions": opponent_decisions,
            "published_rounds": rounds,
            "published_matches": matches,
        }
        for name, count in additions.items():
            self.counters[name] += count
        return {name: tensor.flatten(0, 1) for name, tensor in batch.items()}

    def minibatch_step(self, data):
        # Trainer.update normalizes advantages with the valid learner mask.
        # Filter both policy and value samples, then use the unchanged PPO loss.
        # Empty minibatches do NOT advance Adam momentum or phase counters.
        valid = data["learner"].bool()
        if not bool(valid.any()):
            self.counters["skipped_empty_minibatches"] += 1
            return torch.zeros(6, device=self.env.device)
        selected = {
            name: value[valid] for name, value in data.items() if name != "learner"
        }
        return super().minibatch_step(selected)

    def save(self, path):
        self._check_runtime()
        payload = {
            "format": "learner_response_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "value": self.model.value.state_dict(),
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
    def restore(cls, path):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "learner_response_v1":
            raise ValueError("Unsupported learner-response checkpoint")
        obj = cls(
            LearnerConfig(**payload["config"]), payload["actor"], payload["opponents"]
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Learner-response runtime mismatch")
        for key in (
            "source_sha256",
            "learner_sources",
            "published_initialization",
            "response_opponents",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError(
                    "Learner-response source/initialization/opponent identity mismatch"
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


class LearnerAgent(NeuralAgent):
    """Same greedy public actor inference; no training opponent required."""

    def __init__(self, checkpoint):
        content = Path(checkpoint).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "learner_response_v1":
            raise ValueError("Unsupported learner-response policy checkpoint")
        config = LearnerConfig(**payload["config"])
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Learner-response policy initialization mismatch")
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
