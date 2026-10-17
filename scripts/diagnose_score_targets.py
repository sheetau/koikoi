"""Held-out source-match diagnostics for fixed score-selector endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from scripts.diagnose_arbitration import proposal_mask
from scripts.diagnose_paired_targets import cluster_interval, selected_point
from scripts.feature_score_selector import FeatureScoreSelector, pack_features
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha
from scripts.train_score_distribution import ScoreSelector


def decision_summary(chosen_point, outcomes, points, groups):
    selected_wins = np.where(chosen_point, outcomes[:, 1], outcomes[:, 0])
    selected_points = np.where(chosen_point, points[:, 1], points[:, 0])
    result = {}
    for index, expert in enumerate(("parent", "point")):
        wins = (selected_wins - outcomes[:, index]) / 2
        margins = selected_points - points[:, index]
        result[expert] = {
            "win_score_gain": float(wins.mean()),
            "win_source_match_ci95": cluster_interval(wins, groups),
            "point_margin_gain": float(margins.mean()),
            "point_source_match_ci95": cluster_interval(margins, groups),
        }
    return result


@torch.no_grad()
def diagnose(dataset, checkpoint, output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic path")
    torch.set_num_threads(2)
    data, dataset_digest = read_checkpoint(dataset)
    fitted, checkpoint_digest = read_checkpoint(checkpoint)
    if fitted["dataset"]["sha256"] != dataset_digest:
        raise ValueError("Training dataset changed")
    public = data["public_observations"]
    config = fitted["network_config"]
    if fitted["format"] == "paired_feature_score_selector_v1":
        cache, digest = read_checkpoint(fitted["feature_cache"]["path"])
        if digest != fitted["feature_cache"]["sha256"]:
            raise ValueError("Feature cache changed")
        public = pack_features(public, cache["embeddings"], cache["scores"])
        model = FeatureScoreSelector(config["kind"], config["width"])
    elif fitted["format"] == "paired_score_selector_v1":
        model = ScoreSelector(config["kind"], config["width"])
    else:
        raise ValueError("Unsupported score-selector checkpoint")
    model.load_state_dict(fitted["model"])
    model.eval()
    valid = match_split(data["match_groups"])
    public = public[valid]
    values, choices = [], []
    for batch in public.split(1024):
        mask = proposal_mask(batch[:, :562])
        actions = model(batch, mask)[0].argmax(-1)
        choices.append(selected_point(batch[:, :562], actions))
        values.append(model.values(batch))
    chosen = torch.cat(choices).numpy()
    predicted = torch.cat(values).numpy()
    outcomes = data["outcomes"][valid].numpy()
    points = data["point_differences"][valid].numpy()
    groups = data["match_groups"][valid].numpy()
    phases = public[:, 528:532].argmax(-1).numpy()
    sections = {}
    for name, active in {
        "all": np.ones(len(public), dtype=bool),
        **{f"phase{p}": phases == p for p in range(4)},
    }.items():
        if not active.any():
            continue
        section = {
            "roots": int(active.sum()),
            "source_matches": len(np.unique(groups[active], axis=0)),
            "point_choice_fraction": float(chosen[active].mean()),
            "single_action_comparisons": decision_summary(
                chosen[active], outcomes[active], points[active], groups[active]
            ),
        }
        if config["kind"] in ("categorical", "win", "paired_win"):
            delta = predicted[active, 1] - predicted[active, 0]
            target = outcomes[active, 1] - outcomes[active, 0]
            section["win_delta_mse"] = float(np.mean((delta - target) ** 2))
            section["win_delta_zero_mse"] = float(np.mean(target**2))
        sections[name] = section
    report = {
        "scope": "Held-out source-match roots; change one action then frozen continuation. Not repeated deployment strength. Development diagnostics may inform a future method choice; final match confirmation must use fresh deals.",
        "dataset_sha256": dataset_digest,
        "checkpoint_sha256": checkpoint_digest,
        "diagnostic_script_sha256": sha(__file__),
        "kind": config["kind"],
        "representation": "expert features"
        if "feature_cache" in fitted
        else "raw public",
        "final_training_loss": fitted["history"][-1],
        "sections": sections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.dataset, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
