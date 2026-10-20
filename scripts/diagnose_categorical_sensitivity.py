"""Saved-root analysis of an exact bounded-outcome categorical gradient envelope."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(probabilities):
    """Maximum ||d E[q]/d logits||_2 over q in [-1,1]^k, for normalized rows."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not 1 <= p.shape[1] <= 8:
        raise ValueError("Expected rows with one through eight legal actions")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError("Probabilities must sum to one")
    # Complementary sign partitions have the same norm. Fix the first sign.
    bits = np.arange(2 ** (p.shape[1] - 1))[:, None]
    partition = np.concatenate(
        [np.ones((len(bits), 1)), (bits >> np.arange(p.shape[1] - 1)) & 1],
        axis=1,
    )
    mass = p @ partition.T
    squared = np.zeros_like(mass)
    for a in range(p.shape[1]):
        squared += p[:, a, None] ** 2 * (partition[None, :, a] - mass) ** 2
    return 2 * np.sqrt(squared.max(1))


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh diagnostic directory")
    source = Path("runs/branch_update_coverage")
    inputs = [source / f"parent{i}_roots.npz" for i in (1, 2)]
    protocol = {
        "scope": "Post-hoc finite-pool analysis; no new simulations or updates",
        "inputs": {str(path): sha(path) for path in inputs},
        "implementation_sha256": sha(__file__),
        "normalization": "Renormalize stored float32 legal probabilities in float64",
        "priorities": {
            "uniform": "1",
            "gini": "0.05 + sqrt(1 - sum(p**2))",
            "envelope": "0.05 + max_{q in [-1,1]^k} ||diag(p)q - p(p.q)||",
        },
        "metric": "mean(w) * mean(||K*g||**2 / w) / mean(||K*g||**2)",
        "interpretation": "Hypothetical resampling of saved root pools; not within-match variance or future performance",
    }
    output.mkdir(parents=True)
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    results = {}
    for index, path in enumerate(inputs, 1):
        data = np.load(path)
        legal = data["legal"]
        p = data["probabilities"].astype(np.float64)
        p[~legal] = 0
        normalization_error = float(np.abs(p.sum(1) - 1).max())
        p /= p.sum(1, keepdims=True)
        q = data["q"].astype(np.float64)
        assert np.isfinite(q[legal]).all()
        assert (np.abs(q[legal]) <= 1).all()
        gradient = p * (q - (p * q).sum(1, keepdims=True))
        bound = np.zeros(len(p))
        action_count = legal.sum(1)
        for count in np.unique(action_count):
            rows = action_count == count
            compact = p[rows][legal[rows]].reshape(-1, count)
            bound[rows] = envelope(compact)
        norm = np.linalg.norm(gradient, axis=1)
        assert (norm <= bound + 1e-12).all()
        gini = np.sqrt(np.maximum(0, 1 - (p * p).sum(1)))
        assert (bound <= gini + 1e-12).all()
        energy = (data["counts"] * norm) ** 2
        weights = {
            "uniform": np.ones(len(p)),
            "gini": 0.05 + gini,
            "envelope": 0.05 + bound,
        }
        results[f"parent{index}"] = {
            "roots": len(p),
            "max_legal_actions": int(action_count.max()),
            "max_probability_normalization_correction": normalization_error,
            "second_moment_ratio": {
                name: float(w.mean() * (energy / w).mean() / energy.mean())
                for name, w in weights.items()
            },
            "mean_exact_envelope": float(bound.mean()),
            "mean_gini_envelope": float(gini.mean()),
            "nonconstant_sampled_outcome_roots": int(
                np.count_nonzero(
                    np.where(legal, q, -2).max(1) != np.where(legal, q, 2).min(1)
                )
            ),
        }
    summary = {
        "protocol_sha256": sha(protocol_path),
        "results": results,
        "new_simulator_decisions": 0,
        "new_training_updates": 0,
        "new_independent_samples": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/categorical_sensitivity")
    run(parser.parse_args().output)
