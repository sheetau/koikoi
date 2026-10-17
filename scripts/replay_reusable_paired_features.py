"""Exact public-feature replay with one reusable native environment.

Retains the original batch shape and source replay checks. Separate from the
original helper so currently bound research protocols remain unchanged.
"""

import time
from pathlib import Path

import torch

from koikoi.arbitration import ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, PHASE
from koikoi.train import provenance
from scripts.cache_paired_expert_features import expert_features
from scripts.paired_rollout_targets import collect_root_batch
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import setup


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
    roots = BatchEnv(config["batch"], device, "paper8")
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
        "feature_helper_sha256": sha("scripts/cache_paired_expert_features.py"),
        "workspace": "one reusable public-feature BatchEnv",
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
