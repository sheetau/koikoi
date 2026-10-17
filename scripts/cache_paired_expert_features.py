"""Recover public expert features by exact replay of an existing root stream.

This adds source-trajectory work, but neither fits a model nor repeats branches.
Saved outcomes and source-match splits are unchanged. Hidden states stay inside
the simulator; only public neural features, scores and a public dealer bit leave it.
"""

import argparse
import time
from pathlib import Path

import torch

from koikoi.arbitration import ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, DISCARD, KOI, PHASE, PICK_DRAW, PICK_HAND
from koikoi.train import provenance
from scripts.paired_rollout_targets import collect_root_batch
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import setup


@torch.no_grad()
def expert_features(bank, features, phases, legal, proposals):
    """Return [root, proposal, expert, feature] and corresponding raw scores."""
    count = len(features)
    if proposals.shape != (count, 2) or not legal.gather(1, proposals).all():
        raise ValueError("Expected two legal proposals")
    embeddings = features.new_empty((count, 2, 2, 256))
    scores = features.new_empty((count, 2, 2))
    if not ((phases >= DISCARD) & (phases <= KOI)).all():
        raise ValueError("Expert features require active roots")
    with torch.autocast(device_type=features.device.type, enabled=False):
        for expert, actor in enumerate(bank.experts):
            for phase in (DISCARD, PICK_HAND, KOI):
                active = phases == phase
                if phase == PICK_HAND:
                    active |= phases == PICK_DRAW
                indices = torch.where(active)[0]
                if not len(indices):
                    continue
                width = 50 if phase == KOI else 48
                model = actor.models[str(phase)]
                encoded = model.encoder_block(features[indices, :, :width].contiguous())
                # The pinned koi head reads its two special tokens at columns0/1.
                selected = proposals[indices] - (48 if phase == KOI else 0)
                head_inputs = encoded[:, :, [0, 1]] if phase == KOI else encoded
                raw = model.out(head_inputs).squeeze(1)
                scores[indices, :, expert] = raw.gather(1, selected)
                embeddings[indices, :, expert] = encoded.gather(
                    2, selected[:, None, :].expand(-1, 256, -1)
                ).transpose(1, 2)
    return embeddings, scores


@torch.no_grad()
def replay(dataset, output, device=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh feature-cache path")
    data, digest = read_checkpoint(dataset)
    if data.get("format") != "paired_expert_targets_v1":
        raise ValueError("Expected paired target dataset")
    current = provenance()
    if data["provenance"]["source_sha256"] != current["source_sha256"]:
        raise ValueError("Source identity changed")
    for path, expected in data["script_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Original collection scripts changed")
    config = data["collection"]
    device = device or data["network_config"]["device"]
    if device != data["network_config"]["device"]:
        raise ValueError("Replay must use the original collection device")
    setup(config["seed"])
    bank = ExpertBank(data["experts"], device)
    env = BatchEnv(config["envs"], device, "paper8", seed=config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    work = dict.fromkeys(data["source_work"], 0)
    all_embeddings, all_scores, dealers = [], [], []
    started = time.perf_counter()
    for offset in range(0, config["roots"], config["batch"]):
        states, public, groups, used = collect_root_batch(
            env, bank, config["batch"], generator
        )
        end = offset + len(public)
        if not torch.equal(public.cpu(), data["public_observations"][offset:end]):
            raise ValueError(f"Root public inputs differ at batch offset{offset}")
        if not torch.equal(groups.cpu(), data["match_groups"][offset:end]):
            raise ValueError(f"Root source groups differ at batch offset{offset}")
        roots = BatchEnv(len(states), device, "paper8")
        roots.state.copy_(states)
        features, legal = roots.published_observe()
        embeddings, scores = expert_features(
            bank, features, states[:, PHASE], legal, public[:, 560:].long()
        )
        all_embeddings.append(embeddings.cpu())
        all_scores.append(scores.cpu())
        dealers.append(states[:, DEALER].to(device="cpu", dtype=torch.int8))
        for name, value in used.items():
            work[name] += value
    if work != data["source_work"]:
        raise ValueError("Source trajectory accounting differs")
    result = {
        "format": "paired_public_expert_features_v1",
        "dataset_sha256": digest,
        "script_sha256": sha(__file__),
        "provenance": current,
        "feature_batch": config["batch"],
        "device": device,
        "public_dealer": torch.cat(dealers),
        "embeddings": torch.cat(all_embeddings),
        "scores": torch.cat(all_scores),
        "verified_roots": config["roots"],
        "new_source_replay_work": work,
        "new_branch_decisions": 0,
        "reused_branch_work": data["branch_work"],
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Exact public/group alignment with original root stream; frozen expert features at recorded batch size. No model fitting, new branch outcomes or strength estimate. Floating-point scores can depend on inference batch shape.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(args.dataset, args.output)
    print({k: v for k, v in result.items() if not isinstance(v, torch.Tensor)})


if __name__ == "__main__":
    main()
