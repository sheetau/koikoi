"""Two continuation targets on identical learner roots and public features."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.arbitration import ArbitrationConfig, ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, PHASE, PLAYER
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.cache_paired_expert_features import expert_features
from scripts.collect_reusable_paired_targets import PairedWorkspace
from scripts.opponent_paired_roots import OpponentContinuation, collect_roots
from scripts.paired_rollout_targets import PublicContinuation
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import setup

CONDITIONS = ("parent_only", "opponent_preserving")


@torch.no_grad()
def collect(parent_path, base, seed, roots=65536, batch=256, device="cuda", envs=128):
    base = Path(base)
    if base.exists() or roots < batch or roots % batch or envs < 1:
        raise ValueError("Use a fresh directory and positive integral batch budget")
    setup(seed)
    parent, digest = read_checkpoint(parent_path)
    if parent.get("format") != "published_ppo_v1":
        raise ValueError("Expected adapted published parent")
    point = PublishedActor.from_published("point", "cpu").checkpoint()
    wp = PublishedActor.from_published("wp", device).eval().requires_grad_(False)
    bank = ExpertBank([parent["actor"], point], device)
    env = BatchEnv(envs, device, "paper8", seed=seed)
    root_env = BatchEnv(batch, device, "paper8")
    workspace = PairedWorkspace(batch, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    control = PublicContinuation(bank.experts[0])
    columns = {
        key: []
        for key in ("public", "groups", "opponents", "dealers", "embeddings", "scores")
    }
    labels = {name: {"wins": [], "points": []} for name in CONDITIONS}
    branch_work = {
        name: dict.fromkeys(("decisions", "completed_branches", "round_endings"), 0)
        for name in CONDITIONS
    }
    source = None
    started = time.perf_counter()
    for offset in range(0, roots, batch):
        states, public, groups, opponent_ids, work = collect_roots(
            env, bank, wp, batch, generator
        )
        if source is None:
            source = dict.fromkeys(work, 0)
        for key, value in work.items():
            source[key] += value
        root_env.state.copy_(states)
        # Check alignment before labels are produced; hidden state is not exported.
        torch.testing.assert_close(
            root_env.observe()[0], public[:, :560], rtol=0, atol=0
        )
        features, legal = root_env.published_observe()
        embeddings, scores = expert_features(
            bank, features, states[:, PHASE], legal, public[:, 560:].long()
        )
        for key, value in (
            ("public", public),
            ("groups", groups),
            ("opponents", opponent_ids),
            ("dealers", states[:, DEALER].to(torch.int8)),
            ("embeddings", embeddings),
            ("scores", scores),
        ):
            columns[key].append(value.cpu())
        opponent_policy = OpponentContinuation(
            bank.experts[0],
            wp,
            bank.experts[1],
            states[:, PLAYER].repeat_interleave(2),
            opponent_ids.repeat_interleave(2),
        )
        for name, policy in zip(CONDITIONS, (control, opponent_policy), strict=True):
            wins, points, used = workspace(states, public[:, 560:].long(), policy)
            labels[name]["wins"].append(wins.cpu())
            labels[name]["points"].append(points.cpu())
            for key, value in asdict(used).items():
                branch_work[name][key] += value
        if (offset + batch) % (batch * 8) == 0:
            print(
                json.dumps(
                    {
                        "roots": offset + batch,
                        "source": source,
                        "branches": branch_work,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    columns = {key: torch.cat(values) for key, values in columns.items()}
    identity = provenance()
    scripts = [
        __file__,
        "scripts/opponent_paired_roots.py",
        "scripts/collect_reusable_paired_targets.py",
        "scripts/paired_rollout_targets.py",
        "scripts/cache_paired_expert_features.py",
        "scripts/train_paired_arbitration.py",
    ]
    common = {
        "format": "paired_expert_targets_v1",
        "public_observations": columns["public"],
        "match_groups": columns["groups"],
        "opponent_ids": columns["opponents"],
        "experts": bank.snapshots(),
        "parent_value": parent["value"],
        "parent": {
            "path": str(parent_path),
            "sha256": digest,
            "counters": parent["counters"],
        },
        "opponent_wp": wp.checkpoint(),
        "network_config": asdict(
            ArbitrationConfig(seed=seed, device=device, envs=envs)
        ),
        "source_work": source,
        "provenance": identity,
        "script_hashes": {str(path): sha(path) for path in scripts},
    }
    receipts = {}
    for name in CONDITIONS:
        folder = base / name
        folder.mkdir(parents=True)
        dataset = folder / "targets.pt"
        payload = {
            **common,
            "outcomes": torch.cat(labels[name]["wins"]),
            "point_differences": torch.cat(labels[name]["points"]),
            "branch_work": branch_work[name],
            "collection": {
                "seed": seed,
                "roots": roots,
                "batch": batch,
                "envs": envs,
                "behavior_parent_probability": 0.9,
                "continuation": name,
                "source_policy": "learner 90/10 parent/Point; opponent WP/Point crossed by lane and episode",
                "elapsed_seconds": time.perf_counter() - started,
                "shared_root_conditions": list(CONDITIONS),
                "accounting": "Source and public feature extraction shared once across both conditions; branch work is condition-specific. Opponent IDs are metadata only.",
            },
        }
        torch.save(payload, dataset)
        cache = {
            "format": "paired_public_expert_features_v1",
            "dataset_sha256": sha(dataset),
            "extraction": "at_collection",
            "script_sha256": sha(__file__),
            "feature_helper_sha256": sha("scripts/cache_paired_expert_features.py"),
            "provenance": identity,
            "feature_batch": batch,
            "device": device,
            "public_dealer": columns["dealers"],
            "embeddings": columns["embeddings"],
            "scores": columns["scores"],
            "verified_roots": roots,
            "aligned_source_work": source,
            "new_source_replay_work": dict.fromkeys(source, 0),
            "new_branch_decisions": 0,
            "reused_branch_work": branch_work[name],
            "elapsed_seconds": None,
            "scope": "Extracted once from exact source roots before branch outcomes. Shared feature bytes across two target conditions; no additional replay or hidden-state export. Extraction time included in collection, not separately timed.",
        }
        torch.save(cache, folder / "features.pt")
        receipts[name] = {
            "dataset_sha256": sha(dataset),
            "cache_sha256": sha(folder / "features.pt"),
            "branch_work": branch_work[name],
        }
    return {
        "conditions": receipts,
        "shared_source_work": source,
        "roots": roots,
        "opponent_root_counts": torch.bincount(
            columns["opponents"], minlength=2
        ).tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
