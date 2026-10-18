"""Frozen rollout inference with fixed-size phase-head calls, including padding.

This identifies a different numerical execution policy from dynamic batching.
It does not claim universal floating-point invariance or equality to CPU roots.
Historical likelihoods and root fallbacks remain separate strict CPU policies.
"""

import torch

from koikoi.layout import DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND, PLAYER, TERMINAL
from scripts.history_conditioned_search import MixtureContinuation


@torch.no_grad()
def fixed_scores(actor, features, phases, legal, microbatch=64):
    """Keep the model's input shape fixed even for a partially filled last call."""
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    n = len(features)
    if features.shape != (n, 300, 50) or phases.shape != (n,) or legal.shape != (n, 50):
        raise ValueError("Invalid published actor inputs")
    if features.dtype != torch.float32 or legal.dtype != torch.bool:
        raise ValueError("Expected float32 features and boolean legal masks")
    if actor.training or any(model.training for model in actor.models.values()):
        raise ValueError("Fixed rollout scores require evaluation mode")
    scores = features.new_full((n, 50), -torch.inf)
    with torch.autocast(device_type=features.device.type, enabled=False):
        for phase in (DISCARD, PICK_HAND, KOI):
            selected = phases == phase
            if phase == PICK_HAND:
                selected |= phases == PICK_DRAW
            indices = torch.where(selected)[0]
            width = 50 if phase == KOI else 48
            for start in range(0, len(indices), microbatch):
                subset = indices[start : start + microbatch]
                padded = features.new_zeros((microbatch, 300, width))
                padded[: len(subset)] = features[subset, :, :width]
                values = actor.models[str(phase)](padded)[: len(subset)]
                if phase == KOI:
                    scores[subset, 48:] = values
                else:
                    scores[subset, :48] = values
    return scores.masked_fill(~legal, -torch.inf)


class FixedMicrobatchContinuation(MixtureContinuation):
    def __init__(self, parent_actor, device, microbatch=64):
        if type(microbatch) is not int or microbatch < 1:
            raise ValueError("microbatch must be a positive integer")
        super().__init__(parent_actor, device)
        self.microbatch = microbatch

    @torch.no_grad()
    def __call__(self, env):
        if self.labels is None or len(self.labels) != env.count:
            raise ValueError("Continuation labels do not match the paired workspace")
        features, legal = env.published_observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=env.device)[None] == solved[:, None]
        )
        active = env.state[:, PHASE] != TERMINAL
        routes = torch.where(env.state[:, PLAYER] == self.root_player, 2, self.labels)
        actions = torch.full((env.count,), -1, device=env.device, dtype=torch.int64)
        for route, actor in enumerate(self.actors):
            selected = torch.where(active & (routes == route))[0]
            if len(selected):
                actions[selected] = fixed_scores(
                    actor,
                    features[selected],
                    env.state[selected, PHASE],
                    legal[selected],
                    self.microbatch,
                ).argmax(-1)
        return actions
