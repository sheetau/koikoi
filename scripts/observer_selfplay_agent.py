"""Greedy CPU inference for isolated observer-self-play checkpoints."""

import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.published_train import PublishedPolicyValue
from scripts.observer_selfplay import ObserverConfig, ObserverValue


class ObserverAgent(NeuralAgent):
    """Use the unchanged public actor path; no inactive-view critic at inference."""

    def __init__(self, checkpoint):
        content = Path(checkpoint).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        if payload.get("format") != "observer_selfplay_v1":
            raise ValueError("Unsupported observer self-play policy checkpoint")
        config = ObserverConfig(**payload["config"])
        if (
            payload["provenance"]["published_initialization"]
            != payload["actor"]["initialization"]
        ):
            raise ValueError("Observer self-play policy initialization mismatch")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        self.training_counters = payload["counters"]
        self.training_provenance = payload["provenance"]
        self.training_config = payload["config"]
        self.model = PublishedPolicyValue(config, payload["actor"], "cpu")
        self.model.value = ObserverValue(
            nn.Linear(561, config.critic_width), *list(self.model.value.children())[1:]
        )
        self.model.value.load_state_dict(payload["value"])
        self.model.eval()
        self.published_env = BatchEnv(1, "cpu", "paper8")
        self.device, self.deterministic = "cpu", True
        self.profile, self.belief_trained = "paper8", False
        self.requires_endgame = config.endgame
        self.solved_decisions = 0
