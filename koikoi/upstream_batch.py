"""Explicit batched published policy; strict CPU PublishedAgent stays unchanged.

Float32 features/weights, no ambient autocast. TF32 runtime flags belong to the
caller and must be recorded for numerical comparisons. Batch/device arithmetic
may change near-tied actions; this adapter is not an exact CPU policy certificate.
"""

import hashlib

import torch

from .layout import DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND
from .upstream import UPSTREAM_PATH, UPSTREAM_REV, PublishedAgent


class BatchPublishedAgent:
    def __init__(self, variant="wp", device="cuda", endgame=False):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        devices = [self.device.index] if self.device.type == "cuda" else []
        # Teacher construction must not perturb learner action/initialization RNG.
        with torch.random.fork_rng(devices=devices):
            original = PublishedAgent(variant)
            self.models = {
                phase: original.models[phase]
                .to(device=self.device, dtype=torch.float32)
                .eval()
                .requires_grad_(False)
                for phase in (DISCARD, PICK_HAND, KOI)
            }
        self.variant, self.endgame = variant, endgame
        self.upstream_revision = UPSTREAM_REV
        self.model_hashes = {
            name: hashlib.sha256(
                (UPSTREAM_PATH / f"model_agent/{name}_rl_{variant}.pt").read_bytes()
            ).hexdigest()
            for name in ("discard", "pick", "koikoi")
        }

    @torch.no_grad()
    def logits(self, env):
        if env.state.device != self.device:
            raise ValueError("Published policy and environment must share a device")
        features, legal = env.published_observe()
        logits = torch.full(
            (env.count, 50), -torch.inf, dtype=torch.float32, device=self.device
        )
        phases = env.state[:, PHASE]
        with torch.autocast(device_type=self.device.type, enabled=False):
            for phase, model in self.models.items():
                selected = phases == phase
                if phase == PICK_HAND:
                    selected |= phases == PICK_DRAW
                indices = torch.where(selected)[0]
                if not indices.numel():
                    continue
                width = 50 if phase == KOI else 48
                output = model(features[indices, :, :width].contiguous())
                if phase == KOI:
                    logits[indices, 48:] = output
                else:
                    logits[indices, :48] = output
        return logits.masked_fill(~legal, -torch.inf)

    @torch.no_grad()
    def act(self, env):
        logits = self.logits(env)
        actions = logits.argmax(-1).to(torch.int32)
        actions = torch.where(torch.isfinite(logits).any(-1), actions, -1)
        if self.endgame:
            solved = env.solved_actions()
            actions = torch.where(solved >= 0, solved, actions)
        return actions
