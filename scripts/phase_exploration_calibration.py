"""Fit a koi sampling temperature from public score distributions, not outcomes."""

import argparse
import math
from pathlib import Path

import numpy as np

from scripts.run_arbitration_study import read, sha, write


def choice_statistics(scores, legal, temperature):
    scores, legal = np.asarray(scores, dtype=float), np.asarray(legal)
    if (
        scores.ndim != 2
        or scores.shape != legal.shape
        or legal.dtype != bool
        or not len(scores)
    ):
        raise ValueError("Expected nonempty scores and boolean legal masks")
    if not np.isfinite(scores[legal]).all() or not (legal.sum(1) > 1).all():
        raise ValueError("Use finite legal scores at unforced decisions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Positive finite temperature required")
    masked = np.where(legal, scores, -np.inf)
    z = (masked - masked.max(1, keepdims=True)) / temperature
    weights = np.exp(z)
    normalizer = weights.sum(1, keepdims=True)
    probabilities = weights / normalizer
    entropy = np.log(normalizer[:, 0]) - (probabilities * np.where(legal, z, 0)).sum(1)
    return {
        "rows": len(scores),
        "mean_entropy": float(entropy.mean()),
        "mean_normalized_entropy": float((entropy / np.log(legal.sum(1))).mean()),
        "mean_greedy_probability": float(probabilities.max(1).mean()),
        "median_greedy_probability": float(np.median(probabilities.max(1))),
    }


def calibrate_temperature(scores, legal, target):
    if not math.isfinite(target) or not 0 < target < 1:
        raise ValueError("Target normalized entropy must lie strictly inside (0,1)")
    low, high = 1e-5, 100.0
    if not (
        choice_statistics(scores, legal, low)["mean_normalized_entropy"]
        <= target
        <= choice_statistics(scores, legal, high)["mean_normalized_entropy"]
    ):
        raise ValueError("Target is outside the attainable entropy range")
    for _ in range(70):
        middle = math.sqrt(low * high)
        if choice_statistics(scores, legal, middle)["mean_normalized_entropy"] < target:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def run(output):
    if output.exists():
        raise ValueError("Preserve prior calibration")
    folder = Path("runs/pretrained_policy_diagnostic")
    audit, report = read(folder / "audit.json"), read(folder / "report.json")
    assert audit["status"] == "passed"
    assert audit["report_sha256"] == sha(folder / "report.json")
    assert audit["data_sha256"] == report["scores_sha256"] == sha(folder / "scores.npz")
    inputs = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path("tests/test_phase_exploration_calibration.py"),
            Path("runs/phase_exploration_calibration_tests.xml"),
            folder / "audit.json",
            folder / "report.json",
            folder / "protocol.json",
            folder / "scores.npz",
        )
    }
    with np.load(folder / "scores.npz", allow_pickle=False) as data:
        scores, legal, phases, sources = [
            data[k].copy() for k in ("cuda_initial", "mask", "phases", "sources")
        ]
    eligible = (sources >= 0) & (legal.sum(1) > 1)
    train = eligible & (sources % 4 != 3)
    validation = eligible & (sources % 4 == 3)
    assert not set(sources[train]) & set(sources[validation])
    discard = train & (phases == 0)
    koi = train & (phases == 3)
    target = choice_statistics(scores[discard], legal[discard], 0.05)[
        "mean_normalized_entropy"
    ]
    calibrated = calibrate_temperature(scores[koi], legal[koi], target)
    groups = {}
    for split, selected in (("calibration", train), ("validation", validation)):
        groups[split] = {}
        for phase in range(4):
            choose = selected & (phases == phase)
            groups[split][str(phase)] = {
                "baseline": choice_statistics(scores[choose], legal[choose], 0.05),
                "koi_calibrated": choice_statistics(
                    scores[choose], legal[choose], calibrated if phase == 3 else 0.05
                ),
            }
    write(
        output,
        {
            "status": "complete",
            "inputs": inputs,
            "phase_temperatures": [0.05, 0.05, 0.05, calibrated],
            "target_normalized_entropy": target,
            "source_matches": {
                "calibration": len(set(sources[train])),
                "validation": len(set(sources[validation])),
            },
            "groups": groups,
            "new_neural_calls": 0,
            "new_simulation_decisions": 0,
            "training_updates": 0,
            "scope": "Exploratory score-only calibration on old random-policy public states; validation diagnoses score distribution, not learned strength or training visitation.",
        },
    )
    print(
        {
            "temperature": calibrated,
            "target": target,
            "validation_koi": groups["validation"]["3"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
