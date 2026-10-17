"""Fixed fit/holdout study of public round-start continuation probabilities.

Reuses the audited complete-match corpus; does not simulate or train a policy.
The held-out matches were previously used in descriptive return diagnostics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("runs/round_boundary_calibration")
INPUT = Path("runs/match_return_diagnostic")
INPUT_SHA = "a107da18790ed4821a1e407295f2f2aac35c22618c26fcedcd783b84e93cc3bd"
RIDGE = 0.0001
FIELDS = (
    "seed",
    "round",
    "points0",
    "dealer0",
    "outcome0",
    "network_probability",
    "batch",
    "step",
    "env",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha():
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def extract(data, seed, batch):
    boundary = data["valid"].copy()
    boundary[1:] &= data["round"][1:] != data["round"][:-1]
    time, env = np.nonzero(boundary)
    assert np.all(data["phase"][boundary] == 0)
    assert len(time) == int(data["result"][:, :, 1].sum())
    delta = data["result"][:, :, 3]
    points_before = 30 + np.cumsum(delta, axis=0) - delta
    outcome = np.sign(data["final"][:, 106] - data["final"][:, 107])
    player = data["player"][boundary]
    return np.column_stack(
        (
            seed + env,
            data["round"][boundary],
            points_before[boundary],
            player == 0,
            (outcome[env] + 1) / 2,
            (data["value"][boundary].astype(float) * (1 - 2 * player) + 1) / 2,
            np.full(len(time), batch),
            time,
            env,
        )
    )


def features(rows):
    """Only public points and dealer, with a distinct fit for each round."""
    return np.column_stack(((rows[:, 2] - 30) / 30, 2 * rows[:, 3] - 1))


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def fit_logistic(x, y):
    """Convex regularized fractional-label cross entropy; fixed Newton solve."""
    assert len(x) and x.shape == (len(y), 2)
    assert np.isfinite(x).all() and np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    w = np.zeros(2, dtype=np.float64)

    def objective(beta):
        z = x @ beta
        return np.mean(np.logaddexp(0, z) - y * z) + RIDGE * np.dot(beta, beta) / 2

    for iteration in range(100):
        p = sigmoid(x @ w)
        gradient = x.T @ (p - y) / len(y) + RIDGE * w
        if np.max(np.abs(gradient)) < 1e-10:
            return w, {
                "iterations": iteration,
                "gradient_max": float(np.max(np.abs(gradient))),
                "objective": float(objective(w)),
            }
        hessian = x.T @ ((p * (1 - p))[:, None] * x) / len(y) + RIDGE * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        rate = 1.0
        old = objective(w)
        while objective(w - rate * step) > old - rate * 1e-4 * np.dot(gradient, step):
            rate /= 2
            assert rate > 1e-12, "Newton line search failed"
        w -= rate * step
    raise RuntimeError("Fixed logistic fit did not converge")


def cluster_interval(rows, delta):
    _, match = np.unique(rows[:, 0], return_inverse=True)
    sums = np.bincount(match, weights=delta)
    counts = np.bincount(match)
    rng = np.random.default_rng(99)
    samples = []
    for _ in range(5000):
        indices = rng.integers(0, len(counts), len(counts))
        samples.append(sums[indices].sum() / counts[indices].sum())
    return np.quantile(samples, [0.025, 0.975]).tolist()


def calibration_bins(y, p):
    bucket = np.minimum((p * 10).astype(int), 9)
    result = []
    for index in range(10):
        subset = bucket == index
        if subset.any():
            result.append(
                {
                    "bin": index,
                    "count": int(subset.sum()),
                    "predicted": float(p[subset].mean()),
                    "observed": float(y[subset].mean()),
                }
            )
    return result


def evaluate(rows, probability):
    y = rows[:, 4]
    predictions = {
        "public_logistic": probability,
        "network_with_cards": np.clip(rows[:, 5], 0, 1),
        "constant_half": np.full(len(rows), 0.5),
    }
    losses, results = {}, {}
    for name, p in predictions.items():
        losses[name] = (p - y) ** 2
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        results[name] = {
            "brier": float(losses[name].mean()),
            "cross_entropy": float(
                np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))
            ),
            "bins": calibration_bins(y, p),
        }
    differences = {}
    for other in ("network_with_cards", "constant_half"):
        delta = losses["public_logistic"] - losses[other]
        differences[other] = {
            "brier_difference": float(delta.mean()),
            "match_cluster_ci95": cluster_interval(rows, delta),
        }
    by_round = []
    for round_index in range(8):
        subset = rows[:, 1] == round_index
        by_round.append(
            {
                "round": round_index,
                "count": int(subset.sum()),
                "public_brier": float(losses["public_logistic"][subset].mean()),
                "network_brier": float(losses["network_with_cards"][subset].mean()),
            }
        )
    return {
        "rows": len(rows),
        "matches": len(np.unique(rows[:, 0])),
        "estimators": results,
        "public_minus_other": differences,
        "by_round": by_round,
        "network_probability_clipped": int(((rows[:, 5] < 0) | (rows[:, 5] > 1)).sum()),
    }


def main():
    assert sha(INPUT / "report.json") == INPUT_SHA
    input_report = json.loads((INPUT / "report.json").read_text(encoding="utf-8"))
    assert input_report["status"] == "complete"
    BASE.mkdir(exist_ok=False)
    report = {
        "producer_sha256": sha(__file__),
        "production_source_sha256": source_sha(),
        "input_report_sha256": INPUT_SHA,
        "protocol": {
            "fit_batches": [0, 1],
            "holdout_batches": [2, 3],
            "ridge": RIDGE,
            "features": ["(points0-30)/30", "2*dealer0-1"],
            "intercept": False,
            "separate_round_fits": 8,
            "bootstrap_matches": 5000,
            "bootstrap_seed": 99,
            "fields": FIELDS,
            "new_simulations": 0,
            "policy_updates": 0,
        },
        "cases": {},
    }
    (BASE / "protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    raw = {}
    for label, checkpoint, digest, _ in input_report["protocol"]["parents"]:
        assert sha(checkpoint) == digest
        rows = []
        for batch, meta in enumerate(input_report["cases"][label]["batches"]):
            path = INPUT / f"{label}_{batch}.npz"
            assert sha(path) == meta["raw_sha256"]
            with np.load(path, allow_pickle=False) as data:
                rows.append(extract(data, meta["seed"], batch))
        rows = np.concatenate(rows)
        train, test = rows[rows[:, 6] < 2], rows[rows[:, 6] >= 2]
        assert len(set(train[:, 0]) & set(test[:, 0])) == 0
        assert len(np.unique(train[:, 0])) == len(np.unique(test[:, 0])) == 2048
        weights, fits = [], []
        for round_index in range(8):
            subset = train[train[:, 1] == round_index]
            w, diagnostics = fit_logistic(features(subset), subset[:, 4])
            weights.append(w)
            fits.append(
                {
                    "round": round_index,
                    "rows": len(subset),
                    "coefficients": w.tolist(),
                    **diagnostics,
                }
            )
        weights = np.stack(weights)
        prediction = sigmoid(
            np.sum(features(test) * weights[test[:, 1].astype(int)], axis=1)
        )
        cells = {}
        for row in train:
            key = tuple(row[1:4].astype(int))
            cells[key] = cells.get(key, 0) + 1
        support = np.array([cells.get(tuple(row[1:4].astype(int)), 0) for row in test])
        result = {
            "parent_sha256": digest,
            "fit_rows": len(train),
            "fits": fits,
            "holdout": evaluate(test, prediction),
            "holdout_unseen_cells": int((support == 0).sum()),
            "holdout_cells_with_fewer_than20_fit_rows": int((support < 20).sum()),
            "round0_score_coefficient_identifiable": False,
        }
        report["cases"][label] = result
        raw.update(
            {
                f"{label}_rows": rows,
                f"{label}_weights": weights,
                f"{label}_prediction": prediction,
                f"{label}_fit_cell_counts": support,
            }
        )
        print(
            json.dumps(
                {
                    "case": label,
                    "holdout": result["holdout"]["public_minus_other"],
                    "unseen_rows": result["holdout_unseen_cells"],
                }
            ),
            flush=True,
        )
    np.savez_compressed(BASE / "data.npz", **raw)
    report.update(
        status="complete",
        data_sha256=sha(BASE / "data.npz"),
        limitations="Fit-disjoint held-out matches are reused diagnostic data, not untouched strength tests. "
        "Network baseline observes fresh private/public cards; public estimator intentionally integrates them out. "
        "Round0 only has points30, so its score slope is unidentified. Sparse/unseen contexts and policy drift limit deployment. "
        "Calibration does not demonstrate an improved policy or authorize replacing terminal rewards.",
    )
    (BASE / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
