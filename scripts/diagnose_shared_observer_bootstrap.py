"""Exact scalar-policy extension of the alternating-observer counterexample.

Both private signals share a nonzero policy parameter. The expected estimator
is a strict local descent direction, despite exact observation critics.
"""

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import torch

from koikoi.train import provenance
from scripts.diagnose_observer_bootstrap import calculate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact():
    alpha, lam = F(1, 100), F(19, 20)
    means = ((F(1, 5), F(21, 100)), (F(1), F(-1)))
    learner = tuple(sum(row) / 2 for row in means)
    opponent = tuple((means[0][a] + means[1][a]) / 2 for a in (0, 1))
    truth = ((means[0][1] - means[0][0]) - alpha * (means[1][1] - means[1][0])) / 8
    gradients = {}
    for name in (
        "opponent_observation_lambda095",
        "opponent_observation_lambda1",
        "learner_to_terminal_lambda095",
        "joint_information_lambda095",
    ):
        gradient, second_moment, total = F(0), F(0), F(0)
        for x in (0, 1):
            for action in (0, 1):
                for reward in (-1, 1):
                    probability = (1 + reward * means[x][action]) / 8
                    score = (F(action) - F(1, 2)) * (1 if x == 0 else -alpha)
                    if name == "opponent_observation_lambda095":
                        advantage = (
                            lam * reward + (1 - lam) * opponent[action] - learner[x]
                        )
                    elif name == "joint_information_lambda095":
                        advantage = (
                            lam * reward + (1 - lam) * means[x][action] - learner[x]
                        )
                    else:
                        advantage = reward - learner[x]
                    sample = score * advantage
                    total += probability
                    gradient += probability * sample
                    second_moment += probability * sample**2
        assert total == 1
        assert gradient == (
            F(-8301, 3200000)
            if name.endswith("lambda095") and name.startswith("opponent_")
            else truth
        )
        gradients[name] = {
            "gradient": gradient,
            "variance": second_moment - gradient**2,
        }
    assert truth == F(3, 800)
    threshold = 199 * (1 - alpha) / (201 * (1 + alpha))
    assert threshold == F(6567, 6767) and lam < threshold
    biased = gradients["opponent_observation_lambda095"]["gradient"]
    assert truth * biased == F(-24903, 2560000000) < 0
    return alpha, truth, threshold, gradients


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    source = provenance()["source_sha256"]
    paths = [Path(__file__), Path("scripts/diagnose_observer_bootstrap.py")]
    inputs = {str(path): sha(path) for path in paths}
    pinned = {
        "runs/observer_bootstrap_verified_cpu.json": "4e3e05b73b4e019bf9e68cf2b90afe0d61a2829b0e821dc303d1224b6a13aeab",
        "runs/observer_bootstrap_verified_cuda.json": "50ecb0247856ed6ddd19b91fb1db841fc52f0783682ea6fc59780fbb4306de7b",
    }
    prior = []
    for path, expected in pinned.items():
        assert sha(path) == expected, path
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        assert (
            report["status"] == "passed"
            and report["production_source_sha256"] == source
        )
        assert all(sha(file) == digest for file, digest in report["inputs"].items())
        inputs[path] = expected
        prior.append(report["result"])
    alpha, truth, threshold, rational = exact()
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    probabilities = torch.sigmoid(torch.stack((theta, -float(alpha) * theta)))
    means = theta.new_tensor([[0.20, 0.21], [1.0, -1.0]])
    objective = ((1 - probabilities) * means[:, 0] + probabilities * means[:, 1]).mean()
    direct = torch.autograd.grad(objective, theta)[0]
    torch.testing.assert_close(
        direct, direct.new_tensor(float(truth)), rtol=0, atol=1e-15
    )
    native_cpu = calculate("cpu")
    for result in [native_cpu, *prior]:
        for name, record in result["cases"].items():
            expected = float(rational[name]["gradient"])
            for key in (
                "analytic_expected_gradient",
                "python_expected_gradient",
                "native_expected_gradient",
            ):
                vector = record[key]
                assert abs(vector[0] - float(alpha) * vector[1] - expected) <= 3.1e-8
    assert all(sha(path) == digest for path, digest in inputs.items())
    assert provenance()["source_sha256"] == source
    result = {
        "status": "passed",
        "inputs": inputs,
        "production_source_sha256": source,
        "policy_logits": ["theta", "-theta/100"],
        "theta": 0,
        "true_gradient": str(truth),
        "direct_autograd_gradient": float(direct),
        "lambda_positive_gradient_threshold": str(threshold),
        "cases": {
            name: {key: str(value) for key, value in row.items()}
            for name, row in rational.items()
        },
        "true_objective_directional_derivative": str(
            truth * rational["opponent_observation_lambda095"]["gradient"]
        ),
        "scope": "Exact rational enumeration and scalar autograd; fresh Python/native CPU checks and chain-rule projection of pinned prior CPU/CUDA results. No fresh CUDA computation, sampled games, policy fitting or Koi-Koi strength claim. Expected score-function gradient before PPO clipping and empirical normalization.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha(output),
                "true_gradient": str(truth),
                "biased_gradient": str(
                    rational["opponent_observation_lambda095"]["gradient"]
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
