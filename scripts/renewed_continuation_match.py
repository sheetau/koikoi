"""Strict online adapter for renewed-chance training checkpoints."""

import torch

from scripts.final_round_training import SnapshotAgent
from scripts.public_memory_continuation_match import ContinuationAgent
from scripts.renewed_continuation import RenewedContinuationConfig


class RenewedContinuationAgent(ContinuationAgent):
    """Reuse the public stopping rule;load only actors and inference configuration.

    Training roots,chance schedules,optimizer state and simulator opponent labels
    are not retained by the agent. Opponent identity must be recovered anew from
    the actual match's publicly available events.
    """

    def __init__(self, path, predictors):
        if set(predictors) != {"wp", "point"}:
            raise ValueError("Expected both fixed candidate predictors")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "renewed_public_continuation_v1":
            raise ValueError("Expected renewed public continuation checkpoint")
        self.config = RenewedContinuationConfig(**payload["config"])
        self.prefix = SnapshotAgent(self.config, payload["prefix_actor"])
        self.tail = SnapshotAgent(self.config, payload["actor"])
        self.predictors = predictors
        self.tracker = None
        self.activation = self.context = None
        self.next_decision = 0
