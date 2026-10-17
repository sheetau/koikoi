"""Direct regression of paired complete-match expert advantages.

Experimental training prototype; inference uses the same public arbitration
architecture, with a distinct checkpoint format identifying its learning method.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from koikoi.arbitration import (
    ArbitrationAgent,
    ArbitrationConfig,
    ArbitrationPolicyValue,
    ExpertBank,
    read_checkpoint,
)
from koikoi.env import BatchEnv
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.paired_rollout_targets import (
    PublicContinuation,
    collect_root_batch,
    match_split,
    paired_outcomes,
)


def setup(seed, threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def script_hashes():
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path("scripts/paired_rollout_targets.py"))
    }


def collect(parent_path, target, seed, roots=65536, batch=256, device="cuda"):
    if roots < batch or roots % batch or Path(target).exists():
        raise ValueError("Use a fresh dataset path and an integral number of batches")
    setup(seed)
    parent, digest = read_checkpoint(parent_path)
    if parent.get("format") != "published_ppo_v1":
        raise ValueError("Expected adapted published parent")
    config = ArbitrationConfig(seed=seed, device=device)
    point = PublishedActor.from_published("point", "cpu").checkpoint()
    bank = ExpertBank([parent["actor"], point], device)
    env = BatchEnv(128, device, "paper8", seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    continuation = PublicContinuation(bank.experts[0])
    source = dict.fromkeys(("decisions", "completed_matches", "round_endings"), 0)
    branches = dict.fromkeys(("decisions", "completed_branches", "round_endings"), 0)
    observations, groups, wins, points = [], [], [], []
    start = time.perf_counter()
    for offset in range(0, roots, batch):
        states, public, identifiers, used = collect_root_batch(
            env, bank, batch, generator
        )
        outcomes, differences, work = paired_outcomes(
            states, public[:, 560:].long(), continuation
        )
        observations.append(public.cpu())
        groups.append(identifiers.cpu())
        wins.append(outcomes.cpu())
        points.append(differences.cpu())
        for key, value in used.items():
            source[key] += value
        for key, value in asdict(work).items():
            branches[key] += value
        if (offset + batch) % (batch * 8) == 0:
            print(
                json.dumps(
                    {
                        "roots": offset + batch,
                        "source": source,
                        "branches": branches,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    payload = {
        "format": "paired_expert_targets_v1",
        "public_observations": torch.cat(observations),
        "match_groups": torch.cat(groups),
        "outcomes": torch.cat(wins),
        "point_differences": torch.cat(points),
        "experts": bank.snapshots(),
        "parent_value": parent["value"],
        "parent": {
            "path": str(parent_path),
            "sha256": digest,
            "counters": parent["counters"],
        },
        "network_config": asdict(config),
        "source_work": source,
        "branch_work": branches,
        "provenance": provenance(),
        "script_hashes": script_hashes(),
        "collection": {
            "seed": seed,
            "roots": roots,
            "batch": batch,
            "envs": 128,
            "behavior_parent_probability": 0.9,
            "continuation": "greedy parent both seats",
            "elapsed_seconds": time.perf_counter() - start,
        },
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload


def prediction(model, public):
    proposals = public[:, 560:].long()
    mask = torch.zeros((len(public), 50), dtype=torch.bool, device=public.device)
    mask.scatter_(1, proposals, True)
    logits = model(public, mask)[0]
    values = logits.gather(1, proposals)
    return values[:, 1] - values[:, 0]


def fit(dataset, target, context="full", seed=19310001, device="cuda", epochs=20):
    if Path(target).exists() or epochs < 1:
        raise ValueError("Use a fresh checkpoint path and positive fitting budget")
    setup(seed)
    payload, dataset_digest = read_checkpoint(dataset)
    if payload.get("format") != "paired_expert_targets_v1":
        raise ValueError("Unsupported training dataset")
    public = payload["public_observations"].to(device)
    outcomes = payload["outcomes"].to(device)
    targets = outcomes[:, 1] - outcomes[:, 0]
    validation = match_split(payload["match_groups"]).to(device)
    train_indices, validation_indices = (
        torch.where(~validation)[0],
        torch.where(validation)[0],
    )
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Need disjoint training and validation source matches")
    config = ArbitrationConfig(context=context, seed=seed, device=device)
    model = ArbitrationPolicyValue(config).to(device)
    model.value.load_state_dict(payload["parent_value"])
    model.value.requires_grad_(False)
    last = model.actor if context == "phase" else model.actor[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        last.bias[1] = -0.01
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    steps = 0
    for epoch in range(epochs):
        order = train_indices[torch.randperm(len(train_indices), device=device)]
        loss_sum = 0.0
        model.train()
        for indices in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(prediction(model, public[indices]), targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite regression loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            predicted = torch.cat(
                [
                    prediction(model, public[indices])
                    for indices in validation_indices.split(1024)
                ]
            )
            actual = targets[validation_indices]
            row = {
                "epoch": epoch + 1,
                "train_mse": loss_sum / len(train_indices),
                "validation_mse": float(F.mse_loss(predicted, actual)),
                "validation_zero_mse": float(actual.square().mean()),
                "validation_point_fraction": float((predicted > 0).float().mean()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    checkpoint = {
        "format": "paired_rollout_arbitration_v1",
        "network_config": asdict(config),
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "experts": payload["experts"],
        "parent": payload["parent"],
        "dataset": {"path": str(dataset), "sha256": dataset_digest},
        "training_config": {
            "algorithm": "paired complete-match advantage regression",
            "loss": "MSE",
            "epochs": epochs,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "minibatch": 1024,
            "selection": "fixed final epoch, no validation selection",
            "context": context,
        },
        "counters": {
            "roots": len(public),
            "training_roots": len(train_indices),
            "validation_roots": len(validation_indices),
            "optimizer_steps": steps,
            "source_work": payload["source_work"],
            "branch_work": payload["branch_work"],
        },
        "provenance": provenance(),
        "script_hashes": script_hashes(),
        "history": history,
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return checkpoint


class PairedRolloutAgent(ArbitrationAgent):
    def __init__(self, checkpoint, device="cpu"):
        payload, self.checkpoint_sha256 = read_checkpoint(checkpoint)
        if payload.get("format") != "paired_rollout_arbitration_v1":
            raise ValueError("Unsupported paired-rollout checkpoint")
        setup(payload["network_config"]["seed"])
        self.config = ArbitrationConfig(**payload["network_config"])
        self.bank = ExpertBank(payload["experts"], device)
        self.model = ArbitrationPolicyValue(self.config).to(device).eval()
        self.model.load_state_dict(payload["model"])
        self.env = BatchEnv(1, device, "paper8")
        self.counters, self.provenance = payload["counters"], payload["provenance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "fit"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roots", type=int, default=65536)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--context", choices=("full", "phase", "no_match"), default="full"
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.input, args.output, args.seed, args.roots, args.batch, args.device)
    else:
        fit(args.input, args.output, args.context, args.seed, args.device, args.epochs)


if __name__ == "__main__":
    main()
