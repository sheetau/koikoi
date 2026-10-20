"""Exact variable-length example for fresh shared policy innovations."""

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path


def probability(branch, history):
    # Uses only the root intervention and this branch's public action history.
    return F(1 + (2 * branch + 3 * len(history) + sum(history)) % 7, 8)


def terminal(branch, history):
    return (
        len(history) == 3
        or (branch == 0 and len(history) == 1 and history[0] == 0)
        or (branch == 1 and len(history) == 2 and history[1] == 1)
    )


def path_law(branch, history=()):
    """Direct probability-tree recursion, without uniforms or coupled sampling."""
    if terminal(branch, history):
        return {history: F(1)}
    p = probability(branch, history)
    result = {}
    for action, mass in ((0, 1 - p), (1, p)):
        for path, tail in path_law(branch, history + (action,)).items():
            result[path] = mass * tail
    return result


def sample_path(branch, uniforms):
    history = ()
    for u in uniforms:
        action = int(u < probability(branch, history))
        history += (action,)
        if terminal(branch, history):
            return history
    raise AssertionError("Incomplete path")


def shared_law(reuse=False):
    # All thresholds are multiples of 1/8; these cells integrate exactly.
    cells = [F(2 * i + 1, 16) for i in range(8)]
    result = defaultdict(F)
    for uniforms in itertools.product(cells, repeat=3):
        used = (uniforms[0],) * 3 if reuse else uniforms
        pair = tuple(sample_path(b, used) for b in (0, 1))
        result[pair] += F(1, 8**3)
    return dict(result)


def independent_law():
    return {
        (a, b): pa * pb
        for a, pa in path_law(0).items()
        for b, pb in path_law(1).items()
    }


def marginal(law, branch):
    result = defaultdict(F)
    for paths, mass in law.items():
        result[paths[branch]] += mass
    return dict(result)


def measurements(law, invert_second=False):
    moments = defaultdict(F)
    for paths, mass in law.items():
        # Hidden terminal bit affects reward only, never policy probabilities.
        for hidden, prior in ((0, F(3, 4)), (1, F(1, 4))):
            rewards = [
                1 if (sum(path) + branch + hidden) % 2 == 0 else -1
                for branch, path in enumerate(paths)
            ]
            if invert_second:
                rewards[1] *= -1
            delta = rewards[0] - rewards[1]
            weight = mass * prior
            moments["mean_delta"] += weight * delta
            moments["second_delta"] += weight * delta**2
    moments["variance_delta"] = moments["second_delta"] - moments["mean_delta"] ** 2
    # Root probabilities (1/4,3/4); derivative w.r.t. the first root logit.
    moments["expected_root_gradient"] = F(3, 16) * moments["mean_delta"]
    moments["root_logit_gradient_variance_trace"] = (
        2 * F(3, 16) ** 2 * moments["variance_delta"]
    )
    return dict(moments)


def verify():
    independent = independent_law()
    fresh = shared_law()
    reused = shared_law(reuse=True)
    for branch in (0, 1):
        assert marginal(fresh, branch) == path_law(branch)
        assert marginal(independent, branch) == path_law(branch)
    assert any(marginal(reused, b) != path_law(b) for b in (0, 1))
    rows = {}
    differences = []
    for invert in (False, True):
        results = {
            name: measurements(law, invert)
            for name, law in (
                ("independent", independent),
                ("fresh_shared", fresh),
                ("reused_within_branch", reused),
            )
        }
        expected = results["independent"]["expected_root_gradient"]
        assert results["fresh_shared"]["expected_root_gradient"] == expected
        differences.append(
            results["fresh_shared"]["variance_delta"]
            - results["independent"]["variance_delta"]
        )
        rows[str(invert)] = {
            name: {key: str(value) for key, value in values.items()}
            for name, values in results.items()
        }
    assert differences[0] * differences[1] < 0
    return {
        "status": "passed",
        "scope": "Exact finite example, not Koi-Koi performance",
        "fresh_shared_marginals_equal_direct_tree": True,
        "reused_innovation_marginals_equal_direct_tree": False,
        "coupling_variance_changes_have_opposite_signs": True,
        "reward_inversion_cases": rows,
        "new_koikoi_simulator_decisions": 0,
        "new_training_updates": 0,
        "new_independent_samples": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/policy_coupling_conditions.json")
    path = Path(parser.parse_args().output)
    if path.exists():
        raise ValueError("Do not overwrite a completed check")
    report = verify()
    report["implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
