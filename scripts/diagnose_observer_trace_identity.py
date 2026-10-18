"""Exact finite-rollout trace identity and score-correlated bootstrap errors."""

import argparse
import hashlib
import itertools
import json
import random
from fractions import Fraction as F
from pathlib import Path

import torch

from koikoi.train import provenance
from scripts.observer_selfplay_returns import policy_advantages


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recurrence(rewards, values, players, lam):
    """Exact signed reverse recurrence in each current actor's payoff units."""
    carry = F(0)
    for t in range(len(rewards) - 1, -1, -1):
        sign = 1 if players[t] == players[t + 1] else -1
        own_reward = rewards[t] * (1 if players[t] == players[0] else -1)
        carry = own_reward + sign * values[t + 1] - values[t] + lam * sign * carry
    return carry


def identity(rewards, values, players, lam, future_return):
    returns = [future_return]
    for reward in reversed(rewards):
        returns.insert(0, reward + returns[0])
    root_values = [
        value * (1 if player == players[0] else -1)
        for value, player in zip(values, players)
    ]
    n = len(rewards)
    internal = (1 - lam) * sum(
        lam ** (k - 1) * (root_values[k] - returns[k]) for k in range(1, n)
    )
    boundary = lam ** (n - 1) * (root_values[n] - returns[n])
    return returns[0] - root_values[0] + internal + boundary


def pathwise():
    rng, cases = random.Random(26070001), 0
    for n in range(1, 7):
        for players in itertools.product((0, 1), repeat=n + 1):
            for _ in range(3):
                rewards = [F(rng.randrange(-9, 10), 7) for _ in range(n)]
                values = [F(rng.randrange(-11, 12), 13) for _ in range(n + 1)]
                for terminal in (False, True):
                    current = values.copy()
                    future = F(rng.randrange(-7, 8), 5)
                    if terminal:
                        current[-1] = future = F(0)
                    for lam in (F(0), F(37, 100), F(19, 20), F(1)):
                        left = recurrence(rewards, current, players, lam)
                        right = identity(rewards, current, players, lam, future)
                        assert left == right
                        cases += 1
    assert cases == 6048
    return cases


def toy():
    means = ((F(1, 5), F(21, 100)), (F(1), F(-1)))
    atoms = []
    covariance = F(0)
    true_gradient = F(0)
    for x, action, reward in itertools.product((0, 1), (0, 1), (-1, 1)):
        mu = means[x][action]
        probability = (1 + reward * mu) / 8
        score = (F(action) - F(1, 2)) * (1 if x == 0 else F(-1, 100))
        marginal = (means[0][action] + means[1][action]) / 2
        root = sum(means[x]) / 2
        atoms.append((probability, score, reward, root, mu, marginal))
        covariance += probability * score * (marginal - mu)
        true_gradient += probability * score * reward
    assert sum(atom[0] for atom in atoms) == 1
    assert true_gradient == F(3, 800)
    assert covariance == F(-20301, 160000)
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    probs = torch.sigmoid(torch.stack((theta, -theta / 100)))
    mu = theta.new_tensor([[0.2, 0.21], [1, -1]])
    objective = ((1 - probs) * mu[:, 0] + probs * mu[:, 1]).mean()
    assert (
        abs(float(torch.autograd.grad(objective, theta)[0]) - float(true_gradient))
        < 1e-15
    )
    cases, max_error = {}, 0.0
    for terminal in (False, True):
        horizon = 2 if terminal else 1
        rewards = torch.zeros(horizon, 8, dtype=torch.float64)
        values = torch.empty(horizon, 8, 2, dtype=torch.float64)
        dones = torch.zeros(horizon, 8, dtype=torch.bool)
        players = torch.zeros(horizon, 8, dtype=torch.int32)
        tail = torch.zeros(8, 2, dtype=torch.float64)
        next_players = (
            torch.zeros(8, dtype=torch.int32)
            if terminal
            else torch.ones(8, dtype=torch.int32)
        )
        weights = []
        for k, (probability, score, reward, root, mean, marginal) in enumerate(atoms):
            values[0, k] = torch.tensor([float(root), -0.1025], dtype=torch.float64)
            weights.append(float(probability * score))
            if terminal:
                rewards[1, k] = reward
                values[1, k] = torch.tensor(
                    [float(mean), -float(marginal)], dtype=torch.float64
                )
                dones[1, k] = True
                players[1, k] = 1
            else:
                tail[k] = torch.tensor(
                    [float(mean), -float(marginal)], dtype=torch.float64
                )
        weights = torch.tensor(weights, dtype=torch.float64)
        for lam in (F(0), F(37, 100), F(19, 20), F(1)):
            for mode in ("observer", "signed"):
                exact = F(0)
                for probability, score, reward, root, mean, marginal in atoms:
                    bootstrap = mean if mode == "observer" else marginal
                    advantage = (
                        lam * reward + (1 - lam) * bootstrap if terminal else bootstrap
                    ) - root
                    exact += probability * score * advantage
                bias = (
                    F(0)
                    if mode == "observer"
                    else covariance * (1 - lam if terminal else 1)
                )
                assert exact == true_gradient + bias
                actual = policy_advantages(
                    mode,
                    rewards,
                    values,
                    dones,
                    players,
                    tail,
                    next_players,
                    float(lam),
                )[0]
                gradient = float((weights * actual).sum())
                error = abs(gradient - float(exact))
                assert error < 3e-8
                max_error = max(max_error, error)
                key = f"{'terminal' if terminal else 'cutoff'}_{mode}_lambda{lam}"
                cases[key] = {
                    "exact_gradient": str(exact),
                    "exact_bias": str(bias),
                    "cpu_gradient": gradient,
                }
    assert cases["terminal_signed_lambda19/20"]["exact_gradient"] == "-8301/3200000"
    assert cases["cutoff_signed_lambda1"]["exact_gradient"] == "-19701/160000"
    return {
        "true_gradient": str(true_gradient),
        "cross_observer_score_covariance": str(covariance),
        "cases": cases,
        "max_cpu_gradient_error": max_error,
    }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    paths = [
        Path(__file__),
        Path("docs/OBSERVER_TRACE_IDENTITY.rst"),
        Path("scripts/observer_selfplay_returns.py"),
    ]
    inputs = {str(path): sha(path) for path in paths}
    source = provenance()["source_sha256"]
    protocol_path = Path("runs/observer_selfplay_study/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert source == protocol["source"]["source_sha256"]
    assert all(sha(path) == expected for path, expected in protocol["inputs"].items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "production_source_sha256": source,
        "live_protocol_sha256": sha(protocol_path),
        "pathwise_exact_cases": pathwise(),
        "toy": toy(),
        "scope": "Exact rational identity and eight-atom expected gradients; CPU return functions and direct autograd. No sampled games, training, search branches or CUDA computation. Standard GAE identity specialized to observer switching, not a new general policy-gradient theorem.",
    }
    assert all(sha(path) == expected for path, expected in inputs.items())
    assert all(sha(path) == expected for path, expected in protocol["inputs"].items())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha(output),
                "pathwise_exact_cases": result["pathwise_exact_cases"],
                "toy": result["toy"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
