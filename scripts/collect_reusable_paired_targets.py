"""Bounded-resource paired collection with the original source/action protocol.

The original collector creates a new native backend for every branch batch.
On Windows its DLL directory registrations survive discarded backends. This
collector keeps one branch environment and overwrites its complete state before
each independent batch. No observations, transitions or targets are changed.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.arbitration import ArbitrationConfig, ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.paired_rollout_targets import (
    PublicContinuation,
    RolloutWork,
    collect_root_batch,
)
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import script_hashes, setup


class PairedWorkspace:
    def __init__(self, roots, device):
        if roots < 1:
            raise ValueError("Root count must be positive")
        self.roots = roots
        self.env = BatchEnv(2 * roots, device, "paper8", debug=True)

    @torch.no_grad()
    def __call__(self, states, proposals, continuation, max_steps=512):
        if states.ndim != 2 or states.shape[1] != 208 or states.dtype != torch.int32:
            raise ValueError("Expected packed int32 simulator states")
        if proposals.shape != (len(states), 2):
            raise ValueError("Expected two proposals per root")
        if (states[:, PHASE] == TERMINAL).any():
            raise ValueError("Roots must be active")
        count = len(states)
        env = self.env
        if count != self.roots or states.device != env.state.device:
            raise ValueError("Workspace root count or device mismatch")
        env.state.copy_(states.repeat_interleave(2, dim=0))
        player = states[:, PLAYER].repeat_interleave(2)
        actions = proposals.reshape(-1)
        work = RolloutWork()
        for step in range(max_steps):
            active = env.state[:, PHASE] != TERMINAL
            if not active.any():
                break
            if step:
                actions = continuation(env)
            result = env.step(actions, freeze_done=True)
            work.decisions += int(active.sum())
            work.round_endings += int(result[:, 1].sum())
            work.completed_branches += int(result[:, 2].sum())
        if (env.state[:, PHASE] != TERMINAL).any():
            raise RuntimeError("Branch rollout exceeded the fixed completion bound")
        if work.completed_branches != 2 * count:
            raise RuntimeError("Branch completion accounting mismatch")
        points0 = env.state[:, POINTS] - env.state[:, POINTS + 1]
        points = points0 * (1 - 2 * player)
        wins = points.sign().float()
        return wins.reshape(count, 2), points.reshape(count, 2), work


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
    workspace = PairedWorkspace(batch, device)
    start = time.perf_counter()
    for offset in range(0, roots, batch):
        states, public, identifiers, used = collect_root_batch(
            env, bank, batch, generator
        )
        outcomes, differences, work = workspace(
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
        "script_hashes": {**script_hashes(), __file__: sha(__file__)},
        "collection": {
            "seed": seed,
            "roots": roots,
            "batch": batch,
            "envs": 128,
            "behavior_parent_probability": 0.9,
            "continuation": "greedy parent both seats",
            "elapsed_seconds": time.perf_counter() - start,
            "branch_workspace": "one reusable BatchEnv; complete state overwritten per batch",
        },
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return payload
