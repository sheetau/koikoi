"""Final-round adaptation conditioned only on earlier public revelations.

An isolated experiment: original actors, native features and playable policies
are unchanged. Actor rows 2/3, unused in the published encoding, carry two
public candidate bits. Both real-memory and masked controls initialize their
corresponding input weights to zero. The ordinary observation critic is shared
in design, not in parameters, between arms and receives no opponent identity.
"""

import copy
import hashlib
import io
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import PLAYER, ROUND
from koikoi.published_train import packed_observation
from scripts.final_round_training import SnapshotAgent, TailConfig, TailTrainer
from scripts.revealed_opponent_models import RevealedModelSet


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def memory_identity(memory):
    return hashlib.sha256(memory.cpu().contiguous().numpy().tobytes()).hexdigest()


def candidate_code(candidates):
    """Do not guess an identity for ambiguous or inconsistent public evidence."""
    if not set(candidates) <= {"wp", "point"}:
        raise ValueError("Unexpected public candidate class")
    return {"wp": 0, "point": 1}[next(iter(candidates))] if len(candidates) == 1 else -1


def validate_memory(memory, count):
    if (
        not isinstance(memory, torch.Tensor)
        or memory.dtype != torch.int64
        or memory.shape != (count,)
        or not bool(((memory >= -1) & (memory <= 1)).all())
    ):
        raise ValueError("Expected int64 public candidate codes -1/0/1")


def condition_observation(obs, codes, enabled=True):
    """Return packed public observations with two dedicated actor input rows.

    No packed simulator state, opponent routing label or hidden cards accepted.
    Input rows must be unused; refuse accidental double conditioning.
    """
    if obs.ndim != 2 or obs.shape[1] != 15561:
        raise ValueError("Expected packed published observations")
    validate_memory(codes, len(obs))
    if codes.device != obs.device:
        raise ValueError("Memory and observations must share a device")
    result = obs.clone()
    features = result[:, 561:].reshape(-1, 300, 50)
    if bool(features[:, 2:4].any()):
        raise ValueError("Reserved actor feature rows are not empty")
    if enabled:
        features[:, 2] = (codes == 0).float()[:, None]
        features[:, 3] = (codes == 1).float()[:, None]
    return result


@dataclass
class MemoryTailConfig(TailConfig):
    public_memory: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.public_memory) is not bool:
            raise ValueError("public_memory must be boolean")


class MemoryTailTrainer(TailTrainer):
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
        if not isinstance(config, MemoryTailConfig):
            raise TypeError("Expected MemoryTailConfig")
        validate_memory(memory, len(states))
        super().__init__(
            config, states, seats, opponents, actor_snapshot, opponent_snapshots
        )
        self.memory = memory.detach().cpu().clone()
        self.provenance["public_memory_source_sha256"] = source_identity()
        self.provenance["public_memory_bank_sha256"] = memory_identity(self.memory)
        if initialize_memory:
            with torch.no_grad():
                for model in self.model.actor.models.values():
                    model.encoder_block.f1.weight[:, 2:4].zero_()

    @classmethod
    def from_parent(cls, config, path, states, seats, opponents, memory):
        content = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "published_ppo_v1":
            raise ValueError("Expected original published PPO parent")
        if any(
            payload["config"][k] != getattr(config, k) for k in ("profile", "endgame")
        ):
            raise ValueError("Parent rule/assistance mismatch")
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Parent initialization identity mismatch")
        obj = cls(config, states, seats, opponents, memory, payload["actor"])
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
        if self.provenance["public_memory_source_sha256"] != source_identity():
            raise ValueError("Public memory implementation changed")
        if self.provenance["public_memory_bank_sha256"] != memory_identity(self.memory):
            raise ValueError("Public memory root bank changed")

    def observe(self, labels=False):
        result = super().observe(labels)
        indices = self.last_root_indices
        codes = self.memory[indices].to(self.env.device)
        seats = self.bank[1][indices].to(self.env.device)
        codes = torch.where(self.env.state[:, PLAYER] == seats, codes, -1)
        obs = condition_observation(result[0], codes, self.config.public_memory)
        return (obs, *result[1:])

    def save(self, path):
        self._check_runtime()
        if self.prefix_actor is None:
            raise ValueError("A frozen prefix parent is required")
        payload = {
            "format": "public_memory_tail_v1",
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
        if payload.get("format") != "public_memory_tail_v1":
            raise ValueError("Expected public-memory tail checkpoint")
        obj = cls(
            MemoryTailConfig(**payload["config"]),
            *payload["bank"],
            payload["memory"],
            payload["actor"],
            payload["opponents"],
            initialize_memory=False,
        )
        if obj.runtime != payload["runtime"]:
            raise ValueError("Memory-tail runtime mismatch")
        for key in (
            "source_sha256",
            "published_initialization",
            "tail_source_sha256",
            "root_bank_sha256",
            "tail_opponents",
            "public_memory_source_sha256",
            "public_memory_bank_sha256",
        ):
            if obj.provenance[key] != payload["provenance"][key]:
                raise ValueError("Memory-tail source/bank/opponent mismatch")
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


def load_public_memory(base, diagnostic, parent):
    """Join audited training records by case/root address, never by true label."""
    base, diagnostic = Path(base), Path(diagnostic)
    audit = json.loads(
        (diagnostic / "independent_audit.json").read_text(encoding="utf-8")
    )
    if audit["status"] != "passed":
        raise ValueError("Successful independent identification audit required")
    for path, digest in audit["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed identification input: {path}")
    source = [
        json.loads(s)
        for s in (base / f"parent{parent}/source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records = [
        json.loads(s)
        for s in (diagnostic / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected = [r for r in records if r["parent"] == parent]
    if len(source) != len(selected):
        raise ValueError("Source/identification record count mismatch")
    codes = []
    for index, (row, record) in enumerate(zip(source, selected)):
        if record["case_index"] != index or any(
            row[k] != record[k] for k in ("seat", "seed")
        ):
            raise ValueError("Source/identification address mismatch")
        if row["reached_final_round"]:
            if row["root_index"] != len(codes):
                raise ValueError("Root order mismatch")
            codes.append(candidate_code(record["candidates"]))
    return torch.tensor(codes, dtype=torch.int64)


class MemoryTailAgent:
    """Explicit public-event lifecycle; freeze memory at the round-eight boundary.

    A caller must start every match and submit opponent public snapshots after
    each action in rounds 1--7. No ground-truth opponent identity is accepted.
    """

    def __init__(self, path, predictors):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "public_memory_tail_v1":
            raise ValueError("Expected public-memory tail checkpoint")
        self.config = MemoryTailConfig(**payload["config"])
        self.prefix = SnapshotAgent(self.config, payload["prefix_actor"])
        self.tail = SnapshotAgent(self.config, payload["actor"])
        if set(predictors) != {"wp", "point"}:
            raise ValueError("Expected both fixed candidate predictors")
        self.predictors = predictors
        self.tracker = None
        self.tail_code = None

    def start_match(self, match_key, learner_seat):
        self.tracker = RevealedModelSet(match_key, learner_seat, self.predictors)
        self.tail_code = None

    def observe_opponent(self, snapshot, action):
        if self.tracker is None:
            raise ValueError("Start the match before submitting public events")
        if self.tail_code is not None or round(float(snapshot.features[534]) * 12) >= 7:
            raise ValueError("This policy fixes memory before the final round")
        return self.tracker.observe(snapshot, action)

    @torch.no_grad()
    def act(self, env):
        if self.tracker is None or int(env.state[PLAYER]) != self.tracker.viewer:
            raise ValueError("Start the match and act only on the learner turn")
        if env.state[ROUND] < 7:
            if self.tail_code is not None:
                raise ValueError("A new match requires start_match")
            return self.prefix.act(env)
        if self.tail_code is None:
            self.tail_code = candidate_code(self.tracker.candidates)
        solved = solved_koi_action(env)
        if solved is not None:
            return solved
        self.tail.published_env.state.copy_(torch.from_numpy(env.state[None]))
        obs, mask = packed_observation(self.tail.published_env)
        obs = condition_observation(
            obs, torch.tensor([self.tail_code]), self.config.public_memory
        )
        logits, _, _ = self.tail.model(obs, mask)
        return int(logits.argmax(-1))
