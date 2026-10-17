"""Shared action scorer using frozen public expert representations."""

from types import SimpleNamespace

import torch
from torch import nn

from koikoi.arbitration import ArbitrationAgent, ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import DISCARD, KOI, PICK_DRAW, PICK_HAND
from scripts.train_paired_arbitration import setup
from scripts.train_score_distribution import ScoreSelector, validate_roots


def pack_features(public, embeddings, scores):
    count = len(public)
    if (
        public.shape != (count, 562)
        or embeddings.shape != (count, 2, 2, 256)
        or scores.shape != (count, 2, 2)
    ):
        raise ValueError("Misaligned public expert features")
    if not all(torch.isfinite(x).all() for x in (public, embeddings, scores)):
        raise ValueError("Nonfinite public expert features")
    branches = torch.cat((embeddings.flatten(2), scores / 16), dim=-1)
    return torch.cat((public, branches.flatten(1)), dim=-1)


class FeatureScoreSelector(ScoreSelector):
    def __init__(self, kind="categorical", width=128):
        nn.Module.__init__(self)
        if kind not in (
            "categorical",
            "mean",
            "win",
            "paired_win",
            "paired_mean",
            "preference",
        ):
            raise ValueError("Unknown target type")
        self.kind = kind
        # Both experts describe the same proposed action; scorer weights are shared.
        self.body = nn.Sequential(
            nn.Linear(546, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.head = nn.Linear(width, 65 if kind == "categorical" else 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def branch_outputs(self, public):
        if public.ndim != 2 or public.shape[1] != 1590:
            raise ValueError("Expected packed public expert features")
        branches = public[:, 562:].reshape(-1, 2, 514)
        context = public[:, None, 528:560].expand(-1, 2, -1)
        return self.head(self.body(torch.cat((branches, context), dim=-1)))


@torch.no_grad()
def observe_features(bank, env):
    """Compute expert proposals and encoder features with one pass per expert."""
    ordinary, _ = env.observe()
    if not (ordinary[:, 528:532].sum(-1) == 1).all():
        raise ValueError("Feature selector requires active roots")
    phases = ordinary[:, 528:532].argmax(-1)
    features, legal = env.published_observe()
    solved = env.solved_actions()
    legal &= (solved[:, None] < 0) | (
        torch.arange(50, device=env.device)[None] == solved[:, None]
    )
    count = len(ordinary)
    encoded_actions = features.new_zeros((count, 2, 50, 256))
    all_scores = features.new_full((count, 2, 50), -torch.inf)
    with torch.autocast(device_type=features.device.type, enabled=False):
        for expert, actor in enumerate(bank.experts):
            for phase in (DISCARD, PICK_HAND, KOI):
                active = phases == phase
                if phase == PICK_HAND:
                    active |= phases == PICK_DRAW
                ids = torch.where(active)[0]
                if not len(ids):
                    continue
                model = actor.models[str(phase)]
                width = 50 if phase == KOI else 48
                encoded = model.encoder_block(features[ids, :, :width].contiguous())
                head_inputs = encoded[:, :, [0, 1]] if phase == KOI else encoded
                raw = model.out(head_inputs).squeeze(1)
                start, stop = (48, 50) if phase == KOI else (0, 48)
                all_scores[ids, expert, start:stop] = raw
                encoded_actions[ids, expert, start:stop] = head_inputs.transpose(1, 2)
    all_scores.masked_fill_(~legal[:, None, :], -torch.inf)
    proposals = all_scores.argmax(-1)
    scores = all_scores.gather(2, proposals[:, None, :].expand(-1, 2, -1)).transpose(
        1, 2
    )
    embeddings = encoded_actions.gather(
        2, proposals[:, None, :, None].expand(-1, 2, -1, 256)
    ).transpose(1, 2)
    public = torch.cat((ordinary, proposals.float()), dim=-1)
    mask = torch.zeros_like(legal).scatter_(1, proposals, True) & legal
    return pack_features(public, embeddings, scores), mask


class FeatureScoreDistributionAgent(ArbitrationAgent):
    def __init__(self, checkpoint, device="cpu"):
        payload, self.checkpoint_sha256 = read_checkpoint(checkpoint)
        if payload.get("format") != "paired_feature_score_selector_v1":
            raise ValueError("Wrong feature score-selector checkpoint format")
        config = payload["network_config"]
        setup(config["seed"])
        self.config = SimpleNamespace(**config)
        self.bank = ExpertBank(payload["experts"], device)
        self.model = (
            FeatureScoreSelector(config["kind"], config["width"]).to(device).eval()
        )
        self.model.load_state_dict(payload["model"])
        self.env = BatchEnv(1, device, "paper8")
        self.counters, self.provenance = payload["counters"], payload["provenance"]

    @torch.no_grad()
    def batch_act(self, env):
        packed, mask = observe_features(self.bank, env)
        validate_roots(packed[:, :562])
        return self.model(packed, mask)[0].argmax(-1)
