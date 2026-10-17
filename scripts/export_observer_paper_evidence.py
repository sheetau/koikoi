"""Bind the exact observer-bootstrap example to its CPU/CUDA diagnostics."""

from fractions import Fraction
from pathlib import Path

import numpy as np

from koikoi.train import provenance
from scripts.run_arbitration_study import read, sha, write

REPORTS = {
    "cpu": (
        "runs/observer_bootstrap_verified_cpu.json",
        "4e3e05b73b4e019bf9e68cf2b90afe0d61a2829b0e821dc303d1224b6a13aeab",
    ),
    "cuda": (
        "runs/observer_bootstrap_verified_cuda.json",
        "50ecb0247856ed6ddd19b91fb1db841fc52f0783682ea6fc59780fbb4306de7b",
    ),
}


def export():
    # Fraction arithmetic is independent of the diagnostic's tensor recurrence.
    means = [[Fraction(1, 5), Fraction(21, 100)], [Fraction(1), Fraction(-1)]]
    marginal = [sum(row[a] for row in means) / 2 for a in (0, 1)]
    lam = Fraction(19, 20)
    truth = [(row[1] - row[0]) / 8 for row in means]
    signed = [
        (lam * (row[1] - row[0]) + (1 - lam) * (marginal[1] - marginal[0])) / 8
        for row in means
    ]
    assert truth[0] == Fraction(1, 800) and signed[0] == Fraction(-161, 32000)
    threshold = -(marginal[1] - marginal[0]) / (
        (means[0][1] - means[0][0]) - (marginal[1] - marginal[0])
    )
    assert threshold == Fraction(199, 201)
    reports, inputs = {}, {}
    for device, (path, digest) in REPORTS.items():
        assert sha(path) == digest
        report = read(path)
        assert report["status"] == "passed"
        assert report["production_source_sha256"] == provenance()["source_sha256"]
        assert all(sha(p) == h for p, h in report["inputs"].items())
        inputs[path] = digest
        inputs.update(report["inputs"])
        result = report["result"]
        assert result["device"] == device
        assert result["lambda_for_correct_first_coordinate_must_exceed"] == str(
            threshold
        )
        np.testing.assert_allclose(
            result["payoff_means"],
            [[float(x) for x in row] for row in means],
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            result["true_gradient"], [float(x) for x in truth], atol=1e-15, rtol=0
        )
        for name, case in result["cases"].items():
            expected = signed if name == "opponent_observation_lambda095" else truth
            for key, tolerance in (
                ("analytic_expected_gradient", 1e-15),
                ("python_expected_gradient", 3e-8),
                ("native_expected_gradient", 3e-8),
            ):
                np.testing.assert_allclose(
                    case[key], [float(x) for x in expected], atol=tolerance, rtol=0
                )
        reports[device] = report
    destination = Path("docs/paper/generated/observer_evidence.json")
    write(
        destination,
        {
            "status": "passed",
            "exporter_sha256": sha(__file__),
            "inputs": inputs,
            "exact_true_gradient": [str(x) for x in truth],
            "exact_signed_lambda095_gradient": [str(x) for x in signed],
            "lambda_threshold_strict": str(threshold),
            "evidence": reports,
            "scope": "Exact finite-game expectation, before PPO clipping/empirical normalization. One reversed coordinate, not an opposite full gradient. Zero sampled games or training. No Koi-Koi improvement claim.",
        },
    )
    print({"status": "passed", "output": str(destination), "sha256": sha(destination)})


if __name__ == "__main__":
    export()
