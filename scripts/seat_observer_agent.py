"""Singleton greedy CPU inference for isolated public/masked-seat checkpoints."""

import hashlib
import io
from pathlib import Path

import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from scripts.seat_observer_selfplay import load_policy


class SeatObserverAgent(NeuralAgent):
    """Use the existing public observation and endgame-assisted action path."""

    def __init__(self, checkpoint):
        content = Path(checkpoint).read_bytes()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = load_policy(io.BytesIO(content))
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        self.training_counters = payload["counters"]
        self.training_provenance = payload["provenance"]
        self.training_config = payload["config"]
        self.published_env = BatchEnv(1, "cpu", "paper8")
        self.device, self.deterministic = "cpu", True
        self.profile, self.belief_trained = "paper8", False
        self.requires_endgame = self.training_config["endgame"]
        self.solved_decisions = 0
