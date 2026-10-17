"""Exact two-step counterexample for bootstrapping across information sets.

Enumerates all private signals, actions and Bernoulli outcomes. No simulation,
fitting, Koi-Koi rule change or claim of a new general critic-bias theorem.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import torch

from koikoi.env import BatchEnv
from koikoi.estimators import native_gae
from koikoi.train import provenance, signed_gae
from scripts.run_arbitration_study import sha


def calculate(device="cpu"):
    # X is uniform and private to the learner. Its action A is public. The
    # opponent observes A, has one dummy action, then reward R in {-1,+1} ends
    # the game. E[R|X,A] is the table below; the initial policy is uniform.
    means = torch.tensor([[0.2, 0.21], [1.0, -1.0]], dtype=torch.float64, device=device)
    x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    action = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device)
    reward = torch.tensor([-1.0, 1.0] * 4, dtype=torch.float64, device=device)
    probability = 0.25 * (1 + reward * means[x, action]) / 2
    assert float(probability.sum()) == 1.0
    learner_value = means.mean(1)[x]
    public_action_value = means.mean(0)[action]
    # Check that the supplied opponent critic is the EXACT conditional mean
    # of its own payoff given its actual observation A under the current policy.
    for a in (0, 1):
        mask = action == a
        conditional = (
            -(probability[mask] * reward[mask]).sum() / probability[mask].sum()
        )
        torch.testing.assert_close(conditional, -means[:, a].mean(), rtol=0, atol=1e-15)

    theta = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)
    policy = torch.stack((1 - theta.sigmoid(), theta.sigmoid()), 1)
    objective = (policy * means).sum() / 2
    truth = torch.autograd.grad(objective, theta)[0]
    torch.testing.assert_close(
        truth, truth.new_tensor([1 / 800, -1 / 4]), rtol=0, atol=1e-15
    )
    backend = BatchEnv(8, device).backend
    rows = {}
    for name, lam, information in (
        ("opponent_observation_lambda095", 0.95, "opponent"),
        ("opponent_observation_lambda1", 1.0, "opponent"),
        ("learner_to_terminal_lambda095", 0.95, "compressed"),
        ("joint_information_lambda095", 0.95, "joint"),
    ):
        if information == "compressed":
            rewards = reward[None]
            values = learner_value[None]
            dones = torch.ones((1, 8), dtype=torch.bool, device=device)
            players = torch.zeros((1, 8), dtype=torch.int32, device=device)
        else:
            future = (
                public_action_value if information == "opponent" else means[x, action]
            )
            rewards = torch.stack((torch.zeros_like(reward), -reward))
            values = torch.stack((learner_value, -future))
            dones = torch.tensor([[False] * 8, [True] * 8], device=device)
            players = torch.tensor([[0] * 8, [1] * 8], dtype=torch.int32, device=device)
        arguments = (
            rewards,
            values,
            dones,
            players,
            torch.zeros(8, device=device),
            torch.zeros(8, dtype=torch.int32, device=device),
            lam,
        )
        ordinary, _ = signed_gae(*arguments)
        native, _ = native_gae(backend, *arguments)
        torch.testing.assert_close(native.double(), ordinary, rtol=0, atol=3e-7)
        # Score-function gradients have one coordinate for each private signal.
        score = (action.double() - 0.5)[:, None] * torch.nn.functional.one_hot(x, 2)
        samples = score * ordinary[0, :, None]
        gradient = (probability[:, None] * samples).sum(0)
        native_gradient = (probability[:, None] * score * native[0, :, None]).sum(0)
        variance = (probability[:, None] * (samples - gradient).square()).sum(0)
        bias = gradient - truth
        if information == "opponent" and lam < 1:
            targets = lam * means + (1 - lam) * means.mean(0)[None]
            analytic = (targets[:, 1] - targets[:, 0]) / 8
            target = Fraction(1, 2) * Fraction(-161, 16000)
            assert float(gradient[0]) < 0 < float(truth[0])
            torch.testing.assert_close(
                analytic[0], analytic.new_tensor(float(target)), rtol=0, atol=1e-15
            )
        else:
            analytic = truth
        # The production recurrence uses float32 sign/live multiplications even
        # when supplied double values; compare it with the analytic value using
        # numerical tolerance, not an arbitrary-precision equality assertion.
        torch.testing.assert_close(gradient, analytic, rtol=0, atol=3e-8)
        torch.testing.assert_close(native_gradient, analytic, rtol=0, atol=3e-8)
        rows[name] = {
            "analytic_expected_gradient": analytic.tolist(),
            "python_expected_gradient": gradient.tolist(),
            "native_expected_gradient": native_gradient.tolist(),
            "gradient_bias": bias.tolist(),
            "single_sample_gradient_variance": variance.tolist(),
        }
    return {
        "device": device,
        "payoff_means": means.tolist(),
        "true_gradient": truth.tolist(),
        "cases": rows,
        "lambda_for_correct_first_coordinate_must_exceed": str(Fraction(199, 201)),
    }


def run(output, device):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    paths = [
        Path(__file__),
        Path("koikoi/train.py"),
        Path("koikoi/estimators.py"),
        Path("koikoi/native/core.h"),
    ]
    inputs = {str(path): sha(path) for path in paths}
    source = provenance()["source_sha256"]
    result = calculate(device)
    assert all(sha(path) == digest for path, digest in inputs.items())
    assert provenance()["source_sha256"] == source
    report = {
        "status": "passed",
        "inputs": inputs,
        "production_source_sha256": source,
        "result": result,
        "scope": "Exact enumeration of an abstract two-step game, with no sampled matches or policy fitting. The existing signed-GAE implementation and native backend agree. Removing the intermediate opponent observation bootstrap repairs this example; no claim of unbiased finite-observation PPO in general or stronger Koi-Koi play follows.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"output": str(output), "sha256": sha(output), "result": result}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/observer_bootstrap_diagnostic.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.output, args.device)
