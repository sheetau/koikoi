"""Cached-support comparison and exact finite-budget marginal MSE."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from koikoi.reference import ReferenceEnv
from scripts.enumerate_history_support import context
from scripts.mixed_history_sampler import chain_marginals, draw_hand_chain
from scripts.run_arbitration_study import read, sha, write


def expected_errors(pool, size, support, budget=128, global_probability=0.2):
    """Conditional on initialization success; exact finite-state covariance sum."""
    total = math.comb(len(pool), size)
    hands = [frozenset(h) for h in support]
    n = len(hands)
    adjacency = np.array([[len(a ^ b) == 2 for b in hands] for a in hands], dtype=float)
    local = np.eye(n) + (adjacency - np.diag(adjacency.sum(axis=1))) / (
        size * (len(pool) - size)
    )
    global_kernel = np.ones((n, n)) / total + np.eye(n) * (1 - n / total)
    kernel = global_probability * global_kernel + (1 - global_probability) * local
    np.testing.assert_allclose(kernel.sum(axis=1), 1, atol=1e-13)
    np.testing.assert_array_equal(kernel, kernel.T)
    indicator = np.array([[c in h for c in pool] for h in hands], dtype=float)
    centered = indicator - indicator.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(kernel)
    energy = np.square(vectors.T @ centered).sum(axis=1) / n / len(pool)
    covariance = np.array([np.dot(energy, eigenvalues**lag) for lag in range(budget)])
    probability = n / total
    success = 1 - (1 - probability) ** budget
    independent = (
        covariance[0]
        * sum(
            math.comb(budget, k)
            * probability**k
            * (1 - probability) ** (budget - k)
            / k
            for k in range(1, budget + 1)
        )
        / success
    )
    mixed = 0.0
    for length in range(1, budget + 1):
        variance = (
            length * covariance[0]
            + 2 * sum((length - lag) * covariance[lag] for lag in range(1, length))
        ) / length**2
        mixed += (
            probability * (1 - probability) ** (budget - length) * variance / success
        )
    return {
        "initialization_failure_probability": 1 - success,
        "independent_conditional_mse": float(independent),
        "mixed_conditional_mse": float(mixed),
    }


def independent(pool, size, support, rng, budget):
    accepted, checked = [], set()
    for _ in range(budget):
        hand = tuple(sorted(int(c) for c in rng.choice(pool, size, replace=False)))
        checked.add(hand)
        if hand in support:
            accepted.append(hand)
    if not accepted:
        return None, 0, len(checked)
    cards = np.zeros(48)
    for hand in accepted:
        cards[list(hand)] += 1
    return cards / len(accepted), len(set(accepted)), len(checked)


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    base = Path("runs/sampled_history_exact_check")
    audit_path = base / "arithmetic_verification.json"
    assert (
        sha(audit_path)
        == "c9a9b51f4f337bfc9bf70396ca14604679f0fa920cf5593b2cafda7d205e6bf9"
    )
    audit = read(audit_path)
    assert audit["status"] == "passed"
    inputs = dict(audit["inputs"])
    for path in (
        audit_path,
        Path(__file__),
        Path("scripts/mixed_history_sampler.py"),
        Path("tests/test_mixed_history_sampler.py"),
        Path("runs/mixed_history_sampler_tests.xml"),
        Path("tests/test_mixed_history_expectation.py"),
        Path("runs/mixed_history_expectation_tests.xml"),
        Path("docs/MIXED_HISTORY_SAMPLER.rst"),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "budget": 128,
            "replicates": 256,
            "global_probability": 0.2,
            "seed": "1701 + 1000000*parent + 1000*case_index + replicate",
        },
    )
    source_steps, summaries = 0, []
    for row in read(base / "report.json")["results"]:
        path = Path(
            f"runs/final_round_training_study/parent{row['parent']}/source.jsonl"
        )
        source = json.loads(
            path.read_text(encoding="utf-8").splitlines()[row["case_index"]]
        )
        env = ReferenceEnv(source["seed"], "paper8")
        for action in source["actions"][: row["decision"]]:
            assert env.legal()[action]
            env.step(action)
            source_steps += 1
        pool, size, _ = context(env.state)
        batch = read(base / row["support_file"])
        identified = row["exact_beliefs"]["identified"]
        model = next(iter(identified["model_probabilities"]))
        column = batch["model_names"].index(model)
        support = {
            tuple(h)
            for h, flags in zip(batch["hands"], batch["compatible"])
            if flags[column]
        }
        exact = np.array(identified["card_probabilities"])
        records = []
        for replicate in range(256):
            seed = 1701 + 1000000 * row["parent"] + 1000 * row["case_index"] + replicate
            baseline, baseline_distinct, baseline_queries = independent(
                pool, size, support, np.random.default_rng(seed), 128
            )
            chain = draw_hand_chain(
                pool, size, support.__contains__, np.random.default_rng(seed)
            )
            assert (baseline is not None) == bool(chain.states)
            result = {
                "replicate": replicate,
                "seed": seed,
                "success": bool(chain.states),
                "independent_queries": baseline_queries,
                "mixed_queries": len(chain.checked),
                "independent_distinct": baseline_distinct,
                "mixed_distinct": len(set(chain.states)),
            }
            if chain.states:
                mixed = np.array(chain_marginals(chain)["card_probabilities"])
                result.update(
                    independent_mse=float(
                        np.square(baseline[pool] - exact[pool]).mean()
                    ),
                    mixed_mse=float(np.square(mixed[pool] - exact[pool]).mean()),
                )
            records.append(result)
        successful = [r for r in records if r["success"]]
        summary = {k: row[k] for k in ("parent", "opponent", "case_index")}
        summary.update(
            replicates=256,
            failures=256 - len(successful),
            analytic=expected_errors(pool, size, sorted(support)),
            empirical={
                name: float(np.mean([r[name] for r in successful]))
                for name in (
                    "independent_mse",
                    "mixed_mse",
                    "independent_distinct",
                    "mixed_distinct",
                )
            },
            oracle_queries={
                name: sum(r[name] for r in records)
                for name in ("independent_queries", "mixed_queries")
            },
        )
        filename = f"parent{row['parent']}_{row['opponent']}.json"
        write(output / filename, {"records": records, "summary": summary})
        summary.update(record_file=filename, record_sha256=sha(output / filename))
        summaries.append(summary)
        print(json.dumps({"completed_cases": len(summaries)}), flush=True)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output / "report.json",
        {
            "status": "complete",
            "inputs": inputs,
            "results": summaries,
            "source_prefix_replay_decisions": source_steps,
            "proposal_attempts": 4 * 2 * 256 * 128,
            "new_neural_calls": 0,
            "new_evaluation_deals": 0,
            "future_rollouts": 0,
        },
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
