"""Explicit batched published policy; strict CPU PublishedAgent stays unchanged.

Float32 features/weights, no ambient autocast. TF32 runtime flags belong to the
caller and must be recorded for numerical comparisons. Batch/device arithmetic
may change near-tied actions; this adapter is not an exact CPU policy certificate.
"""

import hashlib
import importlib

import torch

from .layout import DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND
from .upstream import UPSTREAM_PATH, UPSTREAM_REV, PublishedAgent, load_source


def parameter_digest(weights):
    """Hash tensor names, shapes, types and exact bytes independently of pickle."""
    digest = hashlib.sha256()
    for phase, state in sorted(weights.items()):
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{phase}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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

    def checkpoint(self):
        weights = {phase: model.state_dict() for phase, model in self.models.items()}
        return {
            "variant": self.variant,
            "endgame": self.endgame,
            "upstream_revision": self.upstream_revision,
            "model_hashes": self.model_hashes.copy(),
            "parameters_sha256": parameter_digest(weights),
            "weights": weights,
        }

    @classmethod
    def from_checkpoint(cls, payload, device):
        """Restore embedded tensors; only pinned upstream source is imported."""
        if payload["upstream_revision"] != UPSTREAM_REV:
            raise ValueError("Published source revision mismatch")
        if payload["variant"] not in ("wp", "point"):
            raise ValueError("Expected wp or point")
        if parameter_digest(payload["weights"]) != payload["parameters_sha256"]:
            raise ValueError("Published parameter digest mismatch")
        load_source()
        net = importlib.import_module("koikoinet2L")
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        if obj.device.type == "cuda" and obj.device.index is None:
            obj.device = torch.device("cuda", torch.cuda.current_device())
        devices = [obj.device.index] if obj.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            obj.models = {}
            for phase, model_type in (
                (DISCARD, net.DiscardModel),
                (PICK_HAND, net.PickModel),
                (KOI, net.KoiKoiModel),
            ):
                model = model_type().to(device=obj.device, dtype=torch.float32)
                model.load_state_dict(payload["weights"][phase])
                obj.models[phase] = model.eval().requires_grad_(False)
        obj.variant, obj.endgame = payload["variant"], payload["endgame"]
        obj.upstream_revision = payload["upstream_revision"]
        obj.model_hashes = payload["model_hashes"].copy()
        return obj

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
