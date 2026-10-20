"""Outcome-coarsened auxiliary learning for published actors (experimental).

The score head learns only representations. Deployment and PPO still use the
original action head. This module does not change rewards or launch training.
"""

import math

import torch
from torch import nn

from koikoi.layout import DISCARD, KOI, PICK_DRAW, PICK_HAND


def margin_classes(margins):
    """Compress even terminal margins; preserve win/draw/loss exactly."""
    if not torch.isfinite(margins).all() or not torch.equal(
        margins / 2, (margins / 2).round()
    ):
        raise ValueError("Expected finite, even terminal point margins")
    return ((margins.clamp(-64, 64) + 64) / 2).long()


def outcome_losses(logits, targets):
    """Per-row negative log likelihoods: full = outcome + within-outcome.

    Coarsening uses exactly the same 65 logits and parameters in both arms.
    The within-outcome term is zero for draws (the unique zero-margin class).
    It does not guarantee preservation of predicted or actual win probability.
    """
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or len(logits) == 0
        or logits.shape[1] != 65
        or targets.shape != (len(logits),)
        or targets.dtype != torch.long
        or targets.device != logits.device
        or not torch.isfinite(logits).all()
        or not ((targets >= 0) & (targets < 65)).all()
    ):
        raise ValueError("Expected finite [N,65] logits and legal integer targets")
    signs = torch.arange(65, device=logits.device).sub(32).sign()
    group = signs[None] == (targets - 32).sign()[:, None]
    group_mass = logits.masked_fill(~group, -torch.inf).logsumexp(-1)
    selected = logits.gather(1, targets[:, None]).squeeze(-1)
    return logits.logsumexp(-1) - group_mass, group_mass - selected


def policy_kl(reference_log_probs, scores, legal, temperature=0.05):
    """KL(reference || current) on the same public legal support, per row."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected a positive finite temperature")
    if (
        scores.shape != reference_log_probs.shape
        or scores.shape != legal.shape
        or legal.dtype != torch.bool
        or not legal.any(-1).all()
        or not torch.isfinite(scores[legal]).all()
        or not torch.isfinite(reference_log_probs[legal]).all()
    ):
        raise ValueError("Invalid policy probabilities, scores or support")
    old = reference_log_probs.detach().masked_fill(~legal, -torch.inf)
    if not torch.allclose(old.logsumexp(-1), torch.zeros_like(old[:, 0]), atol=1e-5):
        raise ValueError("Reference log probabilities are not normalized")
    current = (scores / temperature).masked_fill(~legal, -torch.inf).log_softmax(-1)
    # Mask before subtracting: illegal -inf minus -inf would poison gradients.
    delta = old.masked_fill(~legal, 0) - current.masked_fill(~legal, 0)
    return (old.exp() * delta).sum(-1)


class ScoreAuxiliary(nn.Module):
    """Three auxiliary heads share the inherited actor's public encoders.

    Pool only legal action-token embeddings. Hand/draw capture share the same
    inherited encoder and auxiliary head. No hidden state, target or future
    score enters forward(). The original score head is evaluated unchanged.
    """

    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        parameter = next(actor.parameters())
        self.heads = nn.ModuleDict()
        # Linear initializes on CPU before transfer. Do not advance sampling RNG
        # merely by attaching a head whose initialized weights are all zero.
        with torch.random.fork_rng(devices=[]):
            for phase in (DISCARD, PICK_HAND, KOI):
                width = actor.models[str(phase)].out.in_channels
                head = nn.Linear(width, 65).to(parameter)
                nn.init.zeros_(head.weight)
                # Equal initial loss/draw/win mass, uniform within each sign group.
                nn.init.constant_(head.bias, -math.log(32))
                with torch.no_grad():
                    head.bias[32] = 0
                self.heads[str(phase)] = head

    def forward(self, features, phases, legal):
        n = len(features)
        if (
            n == 0
            or features.shape != (n, 300, 50)
            or phases.shape != (n,)
            or legal.shape != (n, 50)
            or features.dtype != torch.float32
            or legal.dtype != torch.bool
            or not legal.any(-1).all()
            or not ((phases >= DISCARD) & (phases <= KOI)).all()
        ):
            raise ValueError("Expected active public published observations")
        if phases.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer phases")
        if not (
            features.device
            == phases.device
            == legal.device
            == next(self.parameters()).device
        ):
            raise ValueError("Inputs and model must share a device")
        card_rows = phases != KOI
        if legal[card_rows, 48:].any() or legal[~card_rows, :48].any():
            raise ValueError("Legal support does not match phase")
        scores = features.new_full((n, 50), -torch.inf)
        auxiliary = features.new_empty((n, 65))
        with torch.autocast(device_type=features.device.type, enabled=False):
            for phase in (DISCARD, PICK_HAND, KOI):
                active = phases == phase
                if phase == PICK_HAND:
                    active |= phases == PICK_DRAW
                ids = torch.where(active)[0]
                if not len(ids):
                    continue
                model = self.actor.models[str(phase)]
                width = 50 if phase == KOI else 48
                encoded = model.encoder_block(features[ids, :, :width].contiguous())
                tokens = encoded[:, :, [0, 1]] if phase == KOI else encoded
                start, stop = (48, 50) if phase == KOI else (0, 48)
                scores[ids, start:stop] = model.out(tokens).squeeze(1)
                weights = legal[ids, start:stop].to(tokens.dtype)
                pooled = (tokens * weights[:, None]).sum(-1) / weights.sum(-1)[:, None]
                auxiliary[ids] = self.heads[str(phase)](pooled)
        return scores.masked_fill(~legal, -torch.inf), auxiliary

    def auxiliary_loss(self, features, phases, legal, targets, reference, kind):
        if kind not in ("win", "score"):
            raise ValueError("Expected win or score auxiliary objective")
        scores, logits = self(features, phases, legal)
        coarse, within = outcome_losses(logits, targets)
        clone = policy_kl(reference, scores, legal)
        auxiliary = coarse if kind == "win" else coarse + within
        return (auxiliary + clone).mean(), {
            "outcome_nll": coarse.detach().mean(),
            "within_outcome_nll": within.detach().mean(),
            "clone_kl": clone.detach().mean(),
        }
