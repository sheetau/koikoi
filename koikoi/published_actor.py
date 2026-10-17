"""Differentiable pretrained score models; no PPO distribution is implicit.

The caller provides the native public/own-only features, public phase and mask.
Dropout stays disabled during optimization so stored action scores are repeatable.
This actor is deliberately separate from the frozen comparison/teacher adapters.
"""

import copy

import torch
from torch import nn

from .layout import DISCARD, KOI, PICK_DRAW, PICK_HAND
from .upstream_batch import BatchPublishedAgent, parameter_digest


class PublishedActor(nn.Module):
    def __init__(self, checkpoint, device="cpu"):
        super().__init__()
        if checkpoint.get("format") != "published_actor_v1":
            raise ValueError("Unsupported published actor checkpoint format")
        current, initial = checkpoint["current"], checkpoint["initialization"]
        for key in ("variant", "upstream_revision", "model_hashes"):
            if current[key] != initial[key]:
                raise ValueError("Published actor initialization identity mismatch")
        if current["endgame"]:
            raise ValueError("Published actor uses caller-supplied endgame masks")
        original = BatchPublishedAgent.from_checkpoint(current, device)
        self.initialization = copy.deepcopy(initial)
        self.models = nn.ModuleDict(
            {
                str(phase): model.requires_grad_(True)
                for phase, model in original.models.items()
            }
        )
        self.train(False)

    @classmethod
    def from_published(cls, variant="wp", device="cpu"):
        original = BatchPublishedAgent(variant, device).checkpoint()
        initial = {
            key: copy.deepcopy(original[key])
            for key in (
                "variant",
                "upstream_revision",
                "model_hashes",
                "parameters_sha256",
            )
        }
        return cls(
            {
                "format": "published_actor_v1",
                "initialization": initial,
                "current": original,
            },
            device,
        )

    def train(self, mode=True):
        """Keep upstream dropout off; this does not disable autograd."""
        return super().train(False)

    def checkpoint(self):
        weights = {
            int(phase): model.state_dict() for phase, model in self.models.items()
        }
        current = {
            "variant": self.initialization["variant"],
            "endgame": False,
            "upstream_revision": self.initialization["upstream_revision"],
            "model_hashes": copy.deepcopy(self.initialization["model_hashes"]),
            "parameters_sha256": parameter_digest(weights),
            "weights": weights,
        }
        return {
            "format": "published_actor_v1",
            "initialization": copy.deepcopy(self.initialization),
            "current": current,
        }

    def forward(self, features, phases, legal):
        """Return masked raw Q scores, not probabilities or log probabilities.

        Native published features have shape[N,300,50]. Terminal rows return
        all-minus-infinity scores and must be removed before a policy loss.
        Public endgame assistance is supplied by narrowing ``legal`` beforehand.
        """
        n = len(features)
        if (
            features.shape != (n, 300, 50)
            or phases.shape != (n,)
            or legal.shape != (n, 50)
        ):
            raise ValueError("Invalid published actor input shapes")
        if features.dtype != torch.float32 or legal.dtype != torch.bool:
            raise ValueError(
                "Published actor requires float32 features and boolean mask"
            )
        if not (
            features.device
            == phases.device
            == legal.device
            == next(self.parameters()).device
        ):
            raise ValueError("Published actor inputs and weights must share a device")
        scores = features.new_full((n, 50), -torch.inf)
        with torch.autocast(device_type=features.device.type, enabled=False):
            for phase in (DISCARD, PICK_HAND, KOI):
                selected = phases == phase
                if phase == PICK_HAND:
                    selected |= phases == PICK_DRAW
                indices = torch.where(selected)[0]
                if not len(indices):
                    continue
                width = 50 if phase == KOI else 48
                values = self.models[str(phase)](
                    features[indices, :, :width].contiguous()
                )
                if phase == KOI:
                    scores[indices, 48:] = values
                else:
                    scores[indices, :48] = values
        return scores.masked_fill(~legal, -torch.inf)
