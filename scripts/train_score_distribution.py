"""Experimental score-increment supervision on the existing paired dataset.

Categorical heads decode win probabilities at the current score threshold.
A shared-architecture mean-regression control chooses expected point gain.
This prototype does not launch or alter any fixed comparative study.
"""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from koikoi.arbitration import ArbitrationAgent, ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.train import provenance
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import setup


def lead(public):
    return (public[:, 536] - public[:, 537]) * 64


def validate_roots(public):
    if public.ndim != 2 or public.shape[1] != 562 or not torch.isfinite(public).all():
        raise ValueError("Expected finite public observations and proposals")
    if not (
        ((public[:, 535] * 12 - 8).abs() < 1e-6).all() and (public[:, 543] == 1).all()
    ):
        raise ValueError("Expected paper8 round count and bankruptcy termination")
    points = public[:, 536:538] * 64
    if not torch.equal(points, points.round()):
        raise ValueError("Scores must be integral")
    if not ((points > 0).all() and (points.sum(-1) == 60).all()):
        raise ValueError("Support requires active paper8 roots with total score60")
    proposals = public[:, 560:]
    if not (
        torch.equal(proposals, proposals.round())
        and ((proposals >= 0) & (proposals < 50)).all()
    ):
        raise ValueError("Invalid proposal IDs")


def target_indices(public, differences):
    """Clip only tails whose sign cannot change at any active paper8 lead."""
    validate_roots(public)
    if differences.shape != (len(public), 2) or not torch.isfinite(differences).all():
        raise ValueError("Expected two finite final point differences per root")
    increments = differences - lead(public)[:, None]
    if not torch.equal(increments / 2, (increments / 2).round()):
        raise ValueError("Score increments must be even integers")
    clipped = increments.clamp(-64, 64)
    if not torch.equal((clipped + lead(public)[:, None]).sign(), differences.sign()):
        raise ValueError("Tail clipping changed a win/draw/loss outcome")
    return ((clipped + 64) / 2).long(), increments


def decode_returns(logits, current_lead):
    """Expected {-1,0,1} return, equivalent to twice win score minus one."""
    support = torch.arange(-64, 65, 2, device=logits.device, dtype=logits.dtype)
    payoffs = (support[None, :] + current_lead[:, None]).sign()
    return (logits.softmax(-1) * payoffs[:, None, :]).sum(-1)


class ScoreSelector(nn.Module):
    def __init__(self, kind="categorical", width=128):
        super().__init__()
        if kind not in ("categorical", "mean", "win", "paired_win", "paired_mean"):
            raise ValueError("Unknown target type")
        self.kind = kind
        self.body = nn.Sequential(
            nn.Linear(610, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh()
        )
        self.head = nn.Linear(width, 65 if kind == "categorical" else 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def branch_outputs(self, public):
        # A shared scorer conditions on the actual proposed action, not expert ID.
        proposals = public[:, 560:].long()
        features = public[:, None, :560].expand(-1, 2, -1)
        inputs = torch.cat((features, F.one_hot(proposals, 50).float()), dim=-1)
        return self.head(self.body(inputs))

    def values(self, public, outputs=None):
        if outputs is None:
            outputs = self.branch_outputs(public)
        if self.kind == "categorical":
            return decode_returns(outputs, lead(public))
        if self.kind == "win":
            return 2 * outputs.squeeze(-1).sigmoid() - 1
        # Pairwise objectives identify only score differences, not absolute values.
        return outputs.squeeze(-1)

    def forward(self, public, mask):
        values = self.values(public)
        proposals = public[:, 560:562].long()
        logits = public.new_full((len(public), 50), -1e9)
        # Duplicate actions receive exactly the same shared-scorer value.
        logits = logits.scatter(1, proposals, values).masked_fill(~mask, -1e9)
        return (
            logits,
            public.new_zeros(len(public)),
            public.new_zeros((len(public), 48)),
        )


def loss_for(model, public, indices, increments):
    outputs = model.branch_outputs(public)
    if model.kind in ("paired_win", "paired_mean"):
        target = (
            (increments + lead(public)[:, None]).sign()
            if model.kind == "paired_win"
            else increments / 64
        )
        return F.mse_loss(
            outputs[:, 1, 0] - outputs[:, 0, 0], target[:, 1] - target[:, 0]
        )
    if model.kind == "categorical":
        return F.cross_entropy(outputs.flatten(0, 1), indices.flatten())
    if model.kind == "win":
        wins = ((increments + lead(public)[:, None]).sign() + 1) / 2
        return F.binary_cross_entropy_with_logits(outputs.squeeze(-1), wins)
    return F.mse_loss(outputs.squeeze(-1), increments / 64)


def fit(
    dataset,
    target,
    kind="categorical",
    seed=19410001,
    device="cpu",
    epochs=20,
    feature_cache=None,
):
    if Path(target).exists() or epochs < 1:
        raise ValueError("Use a fresh output and positive fitting budget")
    setup(seed)
    data, digest = read_checkpoint(dataset)
    if data["format"] != "paired_expert_targets_v1":
        raise ValueError("Wrong paired dataset format")
    public = data["public_observations"].to(device)
    differences = data["point_differences"].to(device)
    indices, increments = target_indices(public, differences)
    if not torch.equal(differences.sign().float(), data["outcomes"].to(device)):
        raise ValueError("Point labels and win labels disagree")
    validation = match_split(data["match_groups"]).to(device)
    train_ids, valid_ids = torch.where(~validation)[0], torch.where(validation)[0]
    if not len(train_ids) or not len(valid_ids):
        raise ValueError("Need separate training and validation source matches")
    cache_info = None
    if feature_cache is None:
        model = ScoreSelector(kind).to(device)
    else:
        from scripts.feature_score_selector import FeatureScoreSelector, pack_features

        cache, cache_digest = read_checkpoint(feature_cache)
        if (
            cache.get("format") != "paired_public_expert_features_v1"
            or cache["dataset_sha256"] != digest
            or cache["verified_roots"] != len(public)
            or cache["new_source_replay_work"] != data["source_work"]
            or cache["new_branch_decisions"] != 0
        ):
            raise ValueError("Feature cache does not match paired dataset")
        public = pack_features(
            public, cache["embeddings"].to(device), cache["scores"].to(device)
        )
        model = FeatureScoreSelector(kind).to(device)
        cache_info = {
            "path": str(feature_cache),
            "sha256": cache_digest,
            "source_replay_work": cache["new_source_replay_work"],
            "feature_batch": cache["feature_batch"],
            "feature_device": cache["device"],
        }
    initial_body_digest = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.body.parameters())
    ).hexdigest()
    order_seed = seed ^ 0x5A5A5A
    order_generator = torch.Generator(device=device).manual_seed(order_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history, steps = [], 0
    for epoch in range(epochs):
        total = 0.0
        order = train_ids[
            torch.randperm(len(train_ids), device=device, generator=order_generator)
        ]
        model.train()
        for ids in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, public[ids], indices[ids], increments[ids])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite score-supervision loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            steps += 1
            total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            validation_loss = sum(
                float(loss_for(model, public[ids], indices[ids], increments[ids]))
                * len(ids)
                for ids in valid_ids.split(1024)
            ) / len(valid_ids)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train_ids),
                "validation_loss": validation_loss,
            }
        )
        print(history[-1], flush=True)
    payload = {
        "format": "paired_feature_score_selector_v1"
        if cache_info
        else "paired_score_selector_v1",
        "network_config": {"kind": kind, "width": 128, "seed": seed},
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "experts": data["experts"],
        "parent": data["parent"],
        "dataset": {"path": str(dataset), "sha256": digest},
        "training_config": {
            "device": str(torch.device(device)),
            "epochs": epochs,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "minibatch_roots": 1024,
            "selection": "fixed final epoch",
            "minibatch_order_seed": order_seed,
            "initial_shared_body_sha256": initial_body_digest,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "support": "even increments -64..64; endpoint classes include tails",
            "loss": {
                "categorical": "cross entropy",
                "mean": "MSE of increment/64",
                "win": "binary cross entropy with half credit for draws",
                "paired_win": "MSE of paired signed-win difference",
                "paired_mean": "MSE of paired remaining-margin difference/64",
            }[kind],
        },
        "counters": {
            "training_roots": len(train_ids),
            "validation_roots": len(valid_ids),
            "optimizer_steps": steps,
            "reused_source_work": data["source_work"],
            "reused_branch_work": data["branch_work"],
            "new_simulation_decisions": 0,
            "tail_labels": int((increments.abs() > 64).sum()),
        },
        "history": history,
        "provenance": provenance(),
        "script_sha256": sha(__file__),
    }
    if cache_info is not None:
        payload["feature_cache"] = cache_info
        payload["feature_model_script_sha256"] = sha(
            "scripts/feature_score_selector.py"
        )
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload


class ScoreDistributionAgent(ArbitrationAgent):
    def __init__(self, checkpoint, device="cpu"):
        payload, self.checkpoint_sha256 = read_checkpoint(checkpoint)
        if payload.get("format") != "paired_score_selector_v1":
            raise ValueError("Wrong score-selector checkpoint format")
        config = payload["network_config"]
        setup(config["seed"])
        self.config = SimpleNamespace(**config)
        self.bank = ExpertBank(payload["experts"], device)
        self.model = ScoreSelector(config["kind"], config["width"]).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.env = BatchEnv(1, device, "paper8")
        self.counters, self.provenance = payload["counters"], payload["provenance"]

    @torch.no_grad()
    def batch_act(self, env):
        public, mask = self.bank.observe(env)
        validate_roots(public[mask.any(-1)])
        return self.model(public, mask)[0].argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("categorical", "mean", "win", "paired_win", "paired_mean"),
        default="categorical",
    )
    parser.add_argument("--seed", type=int, default=19410001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-cache", type=Path)
    args = parser.parse_args()
    fit(
        args.dataset,
        args.output,
        args.kind,
        args.seed,
        args.device,
        args.epochs,
        feature_cache=args.feature_cache,
    )


if __name__ == "__main__":
    main()
