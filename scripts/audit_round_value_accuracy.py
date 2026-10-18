"""Reconstruct all reported comparisons and three complete bootstrap backups."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_arbitration_study import sha, write


def audit(folder):
    output = folder / "independent_accuracy_audit.json"
    if output.exists():
        raise ValueError("Preserve the previous independent accuracy audit")
    report = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    inputs = report["inputs"].copy()
    inputs[str(Path(__file__))] = sha(__file__)
    inputs[str(folder / "analysis.json")] = sha(folder / "analysis.json")
    for name, digest in report["artifacts"].items():
        assert sha(folder / name) == digest
        inputs[str(folder / name)] = digest
    assert all(sha(p) == h for p, h in inputs.items())
    with np.load(folder / "bootstrap_weights.npz", allow_pickle=False) as data:
        weights, ref_weights = data["table_weights"], data["reference_weights"]
    rng = np.random.default_rng(62000001)
    expected_weights = np.stack(
        [rng.multinomial(128, np.full(128, 1 / 128), size=(8, 256)) for _ in range(2)]
    )
    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(
        ref_weights,
        np.random.default_rng(62000002).multinomial(
            512, np.full(512, 1 / 512), size=256
        ),
    )
    with np.load(folder / "bootstrap_values.npz", allow_pickle=False) as data:
        boot = data["values"]
    assert (
        boot.shape == (2, 2, 256, 8, 2, 2, 59, 2)
        and np.isfinite(boot).all()
        and (np.abs(boot) <= 1).all()
    )
    fitted, refs = [], []
    error = 0.0
    for parent in (1, 2):
        tables = []
        for table, name in enumerate(("A", "B")):
            with np.load(
                folder / f"parent{parent}_{name}" / "kernels.npz", allow_pickle=False
            ) as data:
                points, dealers, done, values = (
                    data[k] for k in ("points0", "next_dealer", "terminal", "values")
                )
            tables.append(values)
            # Fixed structural replicate indices; rerun scalar weighted backups.
            for b in (0, 127, 255):
                expected = np.zeros((8, 2, 2, 59, 2))
                for r in range(7, -1, -1):
                    for seat, model, p, d in np.ndindex(2, 2, 59, 2):
                        key = (r, seat, model, p, d)
                        total = 0.0
                        for n, weight in enumerate(weights[table, r, b]):
                            if not weight:
                                continue
                            k = (*key, n)
                            q = int(points[k])
                            if done[k]:
                                value = ((q > 30) - (q < 30)) * (1 - 2 * seat)
                            else:
                                value = expected[r + 1, seat, model, q - 1, dealers[k]]
                            total += int(weight) * value
                        expected[key] = total / 128
                error = max(
                    error, float(np.abs(expected - boot[parent - 1, table, b]).max())
                )
                np.testing.assert_allclose(
                    expected, boot[parent - 1, table, b], rtol=0, atol=2e-14
                )
        fitted.append(tables)
        with np.load(
            folder / f"reference_parent{parent}" / "outcomes.npz", allow_pickle=False
        ) as data:
            points = data["points0"]
            utility = (
                (points > 30).astype(int) - (points < 30).astype(int)
            ) * np.array([1, -1])[None, :, None, None, None]
            np.testing.assert_array_equal(data["signed_utility"], utility)
            refs.append(utility)
    fitted, refs = np.asarray(fitted), np.asarray(refs)
    contexts = [[r, p, d] for r in (0, 3, 7) for p in (24, 30, 36) for d in (0, 1)]
    assert len(report["reference_comparisons"]) == 576
    for row in report["reference_comparisons"]:
        parent, table, mode = (
            row["parent"] - 1,
            ("A", "B").index(row["table"]),
            ("native", "independent").index(row["mode"]),
        )
        seat, model, r, q, d = (
            row["seat"],
            ("point", "wp").index(row["opponent"]),
            row["round"],
            row["points0"],
            row["dealer"],
        )
        c = contexts.index([r, q, d])
        outcome = refs[parent, mode, seat, model, c]
        predicted = fitted[parent, table, r, seat, model, q - 1, d]
        sampled = boot[parent, table, :, r, seat, model, q - 1, d]
        ref_boot = ref_weights @ outcome / 512
        expected = [
            predicted,
            outcome.mean(),
            predicted - outcome.mean(),
            sampled.std(ddof=1),
            outcome.std(ddof=1) / np.sqrt(512),
        ]
        actual = [
            row[k]
            for k in (
                "table_value",
                "terminal_mean",
                "signed_error",
                "table_bootstrap_se",
                "reference_se",
            )
        ]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
        np.testing.assert_allclose(
            row["error_interval"],
            np.quantile(sampled - ref_boot, [0.025, 0.975]),
            rtol=0,
            atol=2e-14,
        )
    for row in report["chance_comparisons"]:
        p, seat, model = (
            row["parent"] - 1,
            row["seat"],
            ("point", "wp").index(row["opponent"]),
        )
        c = contexts.index([row["round"], row["points0"], row["dealer"]])
        delta = refs[p, 0, seat, model, c] - refs[p, 1, seat, model, c]
        assert row["native_minus_independent"] == delta.mean()
        np.testing.assert_allclose(
            row["interval"],
            np.quantile(ref_weights @ delta / 512, [0.025, 0.975]),
            rtol=0,
            atol=2e-14,
        )
    for row in report["reference_aggregates"]:
        selected = [
            x["signed_error"]
            for x in report["reference_comparisons"]
            if all(x[k] == row[k] for k in ("parent", "table", "mode"))
        ]
        assert len(selected) == 72
        expected = [
            sum(selected) / 72,
            sum(abs(x) for x in selected) / 72,
            (sum(x * x for x in selected) / 72) ** 0.5,
            max(abs(x) for x in selected),
        ]
        np.testing.assert_allclose(
            [row["observed_error"][k] for k in ("mean", "mae", "rmse", "max_abs")],
            expected,
            rtol=0,
            atol=2e-14,
        )
    assert all(sha(p) == h for p, h in inputs.items())
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "comparisons": 576,
            "chance_comparisons": 144,
            "full_weighted_backups": 12,
            "maximum_backup_difference": error,
            "new_simulation_decisions": 0,
            "scope": "Every per-context value/reference difference and interval recomputed; all eight raw error aggregates checked; three fixed whole-table bootstrap replicates per cell independently reconstructed. Remaining bootstrap backups,aggregate bootstrap spreads,and neural trajectories not rerun.",
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "sha256": sha(output),
                "maximum_backup_difference": error,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
