"""Held-out one-action improvement diagnostics, not deployed match strength."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_arbitration_study import sha
from scripts.diagnose_arbitration import proposal_mask
from scripts.paired_rollout_targets import match_split
from scripts.train_paired_arbitration import PairedRolloutAgent, prediction


def cluster_interval(values, groups, repetitions=5000):
    """Resample source matches; retain the root-weighted mean estimand."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if values.ndim != 1 or len(values) != len(groups) or not len(values):
        raise ValueError("Expected nonempty aligned root values and source groups")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite root values")
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    if len(counts) < 2:
        return None
    rng = np.random.default_rng(19730001)
    draws = []
    for offset in range(0, repetitions, 100):
        ids = rng.integers(
            len(counts), size=(min(100, repetitions - offset), len(counts))
        )
        draws.extend(totals[ids].sum(1) / counts[ids].sum(1))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def selected_point(public, actions):
    proposals = public[:, 560:].long()
    if not ((actions == proposals[:, 0]) | (actions == proposals[:, 1])).all():
        raise ValueError("Controller selected an action outside the expert proposals")
    if not (proposals[:, 0] != proposals[:, 1]).all():
        raise ValueError("Expected disagreement roots only")
    return actions == proposals[:, 1]


def summarize(predicted, choose_point, outcomes, groups, train_mean):
    predicted = np.asarray(predicted, dtype=float)
    choose_point = np.asarray(choose_point, dtype=bool)
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.shape != (len(predicted), 2) or choose_point.shape != predicted.shape:
        raise ValueError("Misaligned predictions, decisions and branch outcomes")
    delta = outcomes[:, 1] - outcomes[:, 0]
    chosen = np.where(choose_point, outcomes[:, 1], outcomes[:, 0])
    comparisons = {}
    for name, other in (("parent", outcomes[:, 0]), ("point", outcomes[:, 1])):
        gain = (chosen - other) / 2
        comparisons[name] = {
            "win_score_gain": float(gain.mean()),
            "source_match_bootstrap_ci95": cluster_interval(gain, groups),
        }
    bins = []
    edges = (-np.inf, -0.25, -0.1, 0.0, 0.1, 0.25, np.inf)
    for low, high in pairwise(edges):
        active = (predicted >= low) & (predicted < high)
        bins.append(
            {
                "lower_inclusive": None if not np.isfinite(low) else low,
                "upper_exclusive": None if not np.isfinite(high) else high,
                "roots": int(active.sum()),
                "predicted_delta_mean": float(predicted[active].mean())
                if active.any()
                else None,
                "observed_delta_mean": float(delta[active].mean())
                if active.any()
                else None,
            }
        )
    return {
        "roots": len(predicted),
        "source_matches": len(np.unique(groups, axis=0)),
        "point_choice_fraction": float(choose_point.mean()),
        "validation_mse": float(np.mean((predicted - delta) ** 2)),
        "zero_baseline_mse": float(np.mean(delta**2)),
        "training_mean_baseline_mse": float(np.mean((train_mean - delta) ** 2)),
        "fixed_training_mean": float(train_mean),
        "single_action_comparisons": comparisons,
        "calibration_bins": bins,
    }


def diagnose(dataset, checkpoints, output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    data = torch.load(dataset, map_location="cpu", weights_only=False)
    if data["format"] != "paired_expert_targets_v1":
        raise ValueError("Wrong dataset format")
    split = match_split(data["match_groups"])
    if not split.any() or split.all():
        raise ValueError("Need both training and held-out source matches")
    public = data["public_observations"][split]
    outcomes = data["outcomes"][split].numpy()
    groups = data["match_groups"][split].numpy()
    train_delta = data["outcomes"][~split, 1] - data["outcomes"][~split, 0]
    train_mean = float(train_delta.mean())
    phase = public[:, 528:532].argmax(-1).numpy()
    report = {
        "dataset": str(dataset),
        "dataset_sha256": sha(dataset),
        "diagnostic_script_sha256": sha(__file__),
        "scope": "Held-out source-match roots under the recorded behavior distribution. Exactly one expert action is changed, then both seats follow the frozen continuation. These gains do not measure deploying a selector repeatedly through a match. Bootstrap resamples whole source matches and retains root weighting. No endpoint, threshold, epoch or training decision is selected using this diagnostic.",
        "models": {},
    }
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["dataset"]["sha256"] != report["dataset_sha256"]:
            raise ValueError("Checkpoint belongs to another dataset")
        agent = PairedRolloutAgent(path)
        predictions, actions = [], []
        with torch.no_grad():
            for batch in public.split(1024):
                predictions.append(prediction(agent.model, batch))
                actions.append(agent.model(batch, proposal_mask(batch))[0].argmax(-1))
        predicted = torch.cat(predictions).numpy()
        point = selected_point(public, torch.cat(actions)).numpy()
        sections = {"all": summarize(predicted, point, outcomes, groups, train_mean)}
        for index in range(4):
            active = phase == index
            if active.any():
                sections[f"phase{index}"] = summarize(
                    predicted[active],
                    point[active],
                    outcomes[active],
                    groups[active],
                    train_mean,
                )
        report["models"][str(path)] = {
            "checkpoint_sha256": agent.checkpoint_sha256,
            "context": agent.config.context,
            "sections": sections,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.dataset, args.checkpoints, args.output)


if __name__ == "__main__":
    main()
