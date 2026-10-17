"""Training-only threshold for a fixed 10% expert-switching diagnostic."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from scripts.compare_score_training_budgets import paired_change
from scripts.diagnose_arbitration import proposal_mask
from scripts.diagnose_paired_targets import selected_point
from scripts.diagnose_score_targets import decision_summary
from scripts.feature_score_selector import FeatureScoreSelector, pack_features
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha


def threshold_for_budget(training_differences, fraction=0.1):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Expected a fraction strictly between zero and one")
    if training_differences.ndim != 1 or not training_differences.numel():
        raise ValueError("Expected nonempty training differences")
    if not torch.isfinite(training_differences).all():
        raise ValueError("Nonfinite training differences")
    return max(
        0.0,
        float(
            torch.quantile(
                training_differences.double(), 1 - fraction, interpolation="higher"
            )
        ),
    )


@torch.no_grad()
def diagnose(checkpoint, output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    fitted, checkpoint_digest = read_checkpoint(checkpoint)
    if (
        fitted["format"] != "paired_feature_score_selector_v1"
        or fitted["network_config"]["kind"] != "paired_win"
    ):
        raise ValueError("Expected the fixed feature/paired-win scorer")
    dataset = Path(fitted["dataset"]["path"])
    cache_path = Path(fitted["feature_cache"]["path"])
    data, dataset_digest = read_checkpoint(dataset)
    cache, cache_digest = read_checkpoint(cache_path)
    if (
        dataset_digest != fitted["dataset"]["sha256"]
        or cache_digest != fitted["feature_cache"]["sha256"]
        or cache["dataset_sha256"] != dataset_digest
    ):
        raise ValueError("Dataset/cache identity changed")
    public = pack_features(
        data["public_observations"], cache["embeddings"], cache["scores"]
    )
    valid = match_split(data["match_groups"])
    if not valid.any() or not (~valid).any():
        raise ValueError("Need training and held-out source groups")
    model = FeatureScoreSelector("paired_win", fitted["network_config"]["width"])
    model.load_state_dict(fitted["model"])
    model.eval()
    values = torch.cat([model.values(rows) for rows in public.split(1024)])
    delta = values[:, 1] - values[:, 0]
    threshold = threshold_for_budget(delta[~valid])
    training_fraction = float((delta[~valid] > threshold).double().mean())
    assert training_fraction <= 0.1
    chosen = (delta[valid] > threshold).numpy()
    greedy = torch.cat(
        [
            selected_point(
                rows[:, :562], model(rows, proposal_mask(rows[:, :562]))[0].argmax(-1)
            )
            for rows in public[valid].split(1024)
        ]
    ).numpy()
    outcomes, points, groups = (
        data[key][valid].numpy()
        for key in ("outcomes", "point_differences", "match_groups")
    )

    def selected(mask):
        return np.where(mask, outcomes[:, 1], outcomes[:, 0]), np.where(
            mask, points[:, 1], points[:, 0]
        )

    restricted, unrestricted = selected(chosen), selected(greedy)
    rate = float(chosen.mean())
    random_mean = (
        (1 - rate) * outcomes[:, 0] + rate * outcomes[:, 1],
        (1 - rate) * points[:, 0] + rate * points[:, 1],
    )
    inputs = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            checkpoint,
            dataset,
            cache_path,
            Path("scripts/feature_score_selector.py"),
            Path("scripts/train_score_distribution.py"),
            Path("scripts/diagnose_score_targets.py"),
            Path("scripts/diagnose_paired_targets.py"),
            Path("scripts/compare_score_training_budgets.py"),
        )
    }
    assert inputs[str(checkpoint)] == checkpoint_digest
    report = {
        "scope": "Fixed10% training-score quantile, capped below by zero; strict greater-than switching. Held-out one-action diagnostics under frozen continuation, not full-match strength or an uncertainty bound. Same-rate randomized comparator is its analytical expectation on these roots, not simulated match play.",
        "inputs": inputs,
        "threshold": threshold,
        "training_switch_fraction": training_fraction,
        "validation_roots": int(valid.sum()),
        "validation_source_matches": len(np.unique(groups, axis=0)),
        "validation_switch_fraction": rate,
        "unrestricted_switch_fraction": float(greedy.mean()),
        "restricted": decision_summary(chosen, outcomes, points, groups),
        "unrestricted": decision_summary(greedy, outcomes, points, groups),
        "restricted_minus_unrestricted": paired_change(
            restricted, unrestricted, groups
        ),
        "restricted_minus_same_rate_random_expectation": paired_change(
            restricted, random_mean, groups
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "threshold",
                    "validation_switch_fraction",
                    "restricted",
                    "restricted_minus_unrestricted",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
