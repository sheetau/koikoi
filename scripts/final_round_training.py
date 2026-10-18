"""Fixed-prefix, complete-final-round PPO; isolated from deployed policies.

Root banks contain natural simulator states, but the trainable actor receives
only the existing public/own observation. Opponent labels are routing metadata.
Each learner decision receives its own lane's terminal match reward, with no
opponent-observation bootstrap and no discarded unfinished tails.
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
from koikoi.published_train import (
    PublishedConfig,
    PublishedPolicyValue,
    PublishedTrainer,
)
from koikoi.reference import ReferenceEnv
from scripts.fixed_microbatch_continuation import fixed_scores


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prefix_to_final_round(case, parent, opponent, native):
    """Stop at the natural round-eight start or an earlier terminal state.

    The returned record retains every attempted source, including bankruptcy.
    Prefixes reaching round eight are NOT complete matches.
    """
    if case["seat"] not in (0, 1):
        raise ValueError("Invalid learner seat")
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]).to(native.device))
    actions = []
    rounds = completed = 0
    while env.state[PHASE] != TERMINAL and env.state[ROUND] < 7:
        if len(actions) >= 1024:
            raise AssertionError("Prefix exceeded decision bound")
        actor = parent if env.state[PLAYER] == case["seat"] else opponent
        action = int(actor.act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action], device=native.device))
        np.testing.assert_array_equal(actual[0].cpu().numpy(), result)
        np.testing.assert_array_equal(native.state[0].cpu().numpy(), env.state)
        actions.append(action)
        rounds += int(result[1])
        completed += int(result[2])
    reached = env.state[PHASE] != TERMINAL
    if reached and not (
        env.state[ROUND] == 7 and env.state[TURN] == 0 and env.state[PHASE] == DISCARD
    ):
        raise AssertionError("Expected an untouched natural final-round start")
    return (env.state.copy() if reached else None), {
        **case,
        "actions": actions,
        "reached_final_round": bool(reached),
        "endpoint": env.state.tolist(),
        "work": {
            "decisions": len(actions),
            "round_endings": rounds,
            "completed_matches": completed,
        },
    }


def validate_bank(states, seats, opponents):
    if not isinstance(states, torch.Tensor) or states.dtype != torch.int32:
        raise ValueError("Root states must be int32 tensors")
    n = len(states)
    if n < 1 or states.shape != (n, STATE_SIZE):
        raise ValueError("Root bank must be a nonempty packed-state matrix")
    for name, values in (("seats", seats), ("opponents", opponents)):
        if values.dtype != torch.int64 or values.shape != (n,):
            raise ValueError(f"Invalid {name} vector")
        if not bool(((values == 0) | (values == 1)).all()):
            raise ValueError(f"Invalid {name} labels")
    expected = {ROUND: 7, ROUNDS: 8, BANKRUPT: 1, TURN: 0, PHASE: DISCARD}
    if any(
        not bool((states[:, key] == value).all()) for key, value in expected.items()
    ):
        raise ValueError("Roots must be live paper8 final-round start states")
    if not bool((states[:, POINTS : POINTS + 2] > 0).all()):
        raise ValueError("Bankrupt root is not live")


def bank_identity(states, seats, opponents):
    digest = hashlib.sha256()
    for value in (states, seats, opponents):
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class TailConfig(PublishedConfig):
    # horizon is a safety bound, not a truncation/bootstrapping horizon.
    horizon: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.horizon < 48 or not self.endgame:
            raise ValueError(
                "Final-round learning requires horizon >=48 and assistance"
            )


class TailTrainer(PublishedTrainer):
    def __init__(
        self,
        config,
        states,
        seats,
        opponents,
        actor_snapshot=None,
        opponent_snapshots=None,
    ):
        if not isinstance(config, TailConfig):
            raise TypeError("Expected TailConfig")
        validate_bank(states, seats, opponents)
        super().__init__(config, actor_snapshot)
        self.bank = tuple(v.detach().cpu().clone() for v in (states, seats, opponents))
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
        self.provenance["tail_source_sha256"] = source_identity()
        self.provenance["root_bank_sha256"] = bank_identity(*self.bank)
        self.provenance["tail_opponents"] = {
            n: a.checkpoint()["current"]["parameters_sha256"]
            for n, a in self.opponents.items()
        }
        self.counters.update(
            learner_decisions=0,
            opponent_decisions=0,
            conditional_tail_completions=0,
            conditional_round_endings=0,
        )

    @classmethod
    def from_parent(cls, config, path, states, seats, opponents):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_ppo_v1":
            raise ValueError("Expected published PPO parent")
        if any(
            payload["config"][k] != getattr(config, k) for k in ("profile", "endgame")
        ):
            raise ValueError("Parent rule/assistance mismatch")
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Parent initialization identity mismatch")
        obj = cls(config, states, seats, opponents, payload["actor"])
        obj.model.value.load_state_dict(payload["value"])
        obj.prefix_actor = copy.deepcopy(payload["actor"])
        obj.provenance["weight_initialization"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "path": str(path),
            "counters": payload["counters"].copy(),
        }
        return obj

    def _check_runtime(self):
        super()._check_runtime()
        if self.provenance["tail_source_sha256"] != source_identity():
            raise ValueError("Tail implementation changed during training")
        if self.provenance["root_bank_sha256"] != bank_identity(*self.bank):
            raise ValueError("Root bank changed during training")

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
                raise AssertionError("Illegal tail action")
            ended = result[:, 2].bool()
            if bool((completed & ended).any()):
                raise AssertionError("Duplicate terminal event")
            rewards += result[:, 0].float() * (1 - 2 * seats)
            completed |= ended
            decisions += int(active.sum())
            own_decisions += len(lanes)
            round_ends += int(result[:, 1].sum())
        if not bool(completed.all()) or not bool(
            (self.env.state[:, PHASE] == TERMINAL).all()
        ):
            raise RuntimeError("Tail horizon exhausted; refusing truncated training")
        if round_ends != self.env.count:
            raise AssertionError("Expected exactly one round ending per tail")
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
            raise ValueError("Saving requires an identified frozen prefix parent")
        payload = {
            "format": "final_round_tail_v1",
            "config": asdict(self.config),
            "actor": self.model.actor.checkpoint(),
            "prefix_actor": self.prefix_actor,
            "value": self.model.value.state_dict(),
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
        if payload.get("format") != "final_round_tail_v1":
            raise ValueError("Expected final-round tail checkpoint")
        obj = cls(
            TailConfig(**payload["config"]),
            *payload["bank"],
            payload["actor"],
            payload["opponents"],
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Tail runtime mismatch")
        for key in (
            "source_sha256",
            "published_initialization",
            "tail_source_sha256",
            "root_bank_sha256",
            "tail_opponents",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Tail source/bank/opponent identity mismatch")
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


class SnapshotAgent(NeuralAgent):
    """Strict CPU greedy adapter; receives only native public actor features."""

    def __init__(self, config, snapshot):
        self.model = PublishedPolicyValue(config, snapshot, "cpu").eval()
        self.published_env = BatchEnv(1, "cpu", "paper8")
        self.device, self.deterministic = "cpu", True
        self.profile, self.belief_trained = "paper8", False
        self.requires_endgame, self.solved_decisions = config.endgame, 0


class TailAgent:
    """Use the unchanged parent for rounds 1--7 and adapted actor in round 8."""

    def __init__(self, path):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "final_round_tail_v1":
            raise ValueError("Expected final-round tail policy")
        config = TailConfig(**payload["config"])
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        self.prefix = SnapshotAgent(config, payload["prefix_actor"])
        self.tail = SnapshotAgent(config, payload["actor"])

    def act(self, env):
        return (self.tail if env.state[ROUND] == 7 else self.prefix).act(env)
