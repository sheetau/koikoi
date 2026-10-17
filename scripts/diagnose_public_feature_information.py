"""Check which released-model inputs are absent from stored relative observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, KOIS, PHASE, PLAYER, POINTS
from scripts.benchmark_final_latency import CORPUS, CORPUS_SHA, spread
from scripts.run_arbitration_study import sha


def exchange_player_labels(states):
    """Exchange public seat labels, preserving each player's relative position."""
    result = states.clone()
    locations = states[:, :48]
    result[:, :48] = torch.where(
        (locations == 0) | (locations == 1), 1 - locations, locations
    )
    result[:, :48] = torch.where(
        (locations == 3) | (locations == 4), 7 - locations, result[:, :48]
    )
    result[:, PLAYER] = 1 - states[:, PLAYER]
    result[:, DEALER] = 1 - states[:, DEALER]
    result[:, 48:56] = states[:, 56:64]
    result[:, 56:64] = states[:, 48:56]
    for start in (POINTS, KOIS):
        result[:, start : start + 2] = states[:, start : start + 2].flip(-1)
    return result


@torch.no_grad()
def diagnose(checkpoint, output, count=2048):
    if output.exists():
        raise ValueError("Use a fresh output file")
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("Cached corpus changed")
    torch.set_num_threads(2)
    payload, digest = read_checkpoint(checkpoint)
    bank = ExpertBank(payload["experts"], "cpu")
    with np.load(CORPUS, allow_pickle=False) as cached:
        indices = spread(np.flatnonzero(cached["sources"] >= 0), count)
        states = torch.from_numpy(cached["states"][indices].copy())
    assert torch.equal(exchange_player_labels(exchange_player_labels(states)), states)
    changed_rows = set()
    action_changes = [0, 0]
    max_score_changes = [0.0, 0.0]
    same_proposals_changed_scores = [0, 0]
    same_proposals_count = 0
    for original in states.split(64):
        paired = torch.cat((original, exchange_player_labels(original)))
        env = BatchEnv(len(paired), "cpu", "paper8")
        env.state.copy_(paired)
        public, ordinary_mask = env.observe()
        features, legal = env.published_observe()
        n = len(original)
        assert torch.equal(public[:n], public[n:])
        assert torch.equal(ordinary_mask[:n], ordinary_mask[n:])
        assert torch.equal(legal[:n], legal[n:])
        differences = (features[:n] != features[n:]).any(0).any(-1)
        changed_rows.update(torch.where(differences)[0].tolist())
        solved = env.solved_actions()
        assert torch.equal(solved[:n], solved[n:])
        legal &= (solved[:, None] < 0) | (
            torch.arange(50)[None] == solved[:, None]
        )
        scores = [actor(features, paired[:, PHASE], legal) for actor in bank.experts]
        proposals = torch.stack([q.argmax(-1) for q in scores], dim=-1)
        same = (proposals[:n] == proposals[n:]).all(-1)
        same_proposals_count += int(same.sum())
        for index, q in enumerate(scores):
            difference = torch.where(legal[:n], (q[:n] - q[n:]).abs(), 0)
            action_changes[index] += int(
                (proposals[:n, index] != proposals[n:, index]).sum()
            )
            max_score_changes[index] = max(
                max_score_changes[index], float(difference.max())
            )
            same_proposals_changed_scores[index] += int(
                (same & (difference > 1e-5).any(-1)).sum()
            )
    assert changed_rows == {50, 51}
    report = {
        "scope": "Cached random-play states and public seat-label exchange. No new matches, training or strength estimate. Ordinary relative observations omit absolute dealer identity used by the pinned feature encoder. This is not hidden information or a production-rule defect.",
        "script_sha256": sha(__file__),
        "corpus_sha256": CORPUS_SHA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": count,
        "indices": indices.tolist(),
        "ordinary_observations_masks_guards_equal": True,
        "published_feature_changed_rows": sorted(changed_rows),
        "expert_action_changes": action_changes,
        "expert_max_legal_score_changes": max_score_changes,
        "both_expert_proposals_unchanged": same_proposals_count,
        "same_562_input_changed_legal_scores": same_proposals_changed_scores,
        "score_difference_threshold": 1e-5,
        "interpretation": "Exact released-model features or scores cannot in general be reconstructed from the saved 562 inputs alone. Future cached expert-feature extraction must retain the needed public absolute seat bit or explicitly define and evaluate another representation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "indices"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2048)
    args = parser.parse_args()
    diagnose(args.checkpoint, args.output, args.count)


if __name__ == "__main__":
    main()
